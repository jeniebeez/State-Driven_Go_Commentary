# sgf_rag_multi-agentic-llm.py
import os
os.environ["PYTORCH_JIT_DISABLE"] = "1"
os.environ["TORCH_CUDA_FUSER_DISABLE"] = "1"
from typing import List, Dict, Any, Set

import torch
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
except Exception as e: pass

import json
import time
import gc  
import re
import chromadb
import requests
from transformers import pipeline

from go_feature_extraction import GoFeatureExtractor
from go_tactic_analysis import get_symmetric_coords

CYAN, GREEN, YELLOW, RED, RESET, BOLD = "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[0m", "\033[1m"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "go_knowledge_base")
SGF_PROMPT_DIR = os.path.join(CURRENT_DIR, "source-qipuNsgf", "sgf_prompt")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "eval-results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sgf_rag_multi-agentic-llm.json")

API_URL = os.getenv("API_URL", "http://localhost:8000/v1/chat/completions")
COLLECTION_NAME = "kogo_joseki_dict_v2"

def get_active_model_name(api_url: str) -> str:
    try:
        base_url = api_url.rsplit('/', 2)[0] + "/models"
        res = requests.get(base_url, timeout=5).json()
        if "data" in res and len(res["data"]) > 0:
            return res["data"][0]["id"]
    except Exception:
        pass
    return os.getenv("MODEL_NAME", "qwen3.2-27b")

MODEL_NAME = get_active_model_name(API_URL)
print(f"        [系統] 動態檢測到本地端模型名稱為: {MODEL_NAME}", flush=True)

def robust_json_loads(s: str) -> Any:
    s = re.sub(r'^```json\s*|```$', '', s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    first_brace = s.find('{')
    if first_brace != -1:
        brace_count = 0
        in_string = False
        escape = False
        for idx in range(first_brace, len(s)):
            char = s[idx]
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = s[first_brace:idx+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
    return json.loads(s)  # fallback to raise the original error

def strip_thinking_process(text: str) -> str:
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()
    if "Thinking Process:" in text:
        text = text.split("Thinking Process:")[-1].strip()
    text = re.sub(r'</?think>', '', text).strip()
    text = re.sub(r'^```[a-zA-Z]*\n|```$', '', text).strip()
    return text

class GoTrieNode:
    def __init__(self):
        self.children = {}
        self.kjd_id = None

class GoJosekiTrie:
    def __init__(self):
        self.root = GoTrieNode()

    def build_from_chroma(self, collection):
        print(f"        [Trie] 正在從本地端 ChromaDB 中拉取 Metadata 進行字典樹建構...", flush=True)
        all_records = collection.get(include=["metadatas"])
        if not all_records or not all_records['ids']: return
        for chunk_id, metadata in zip(all_records['ids'], all_records['metadatas']):
            moves_str = metadata.get("moves_string", "")
            if not moves_str: continue
            tokens = moves_str.split("_")
            current_node = self.root
            for token in tokens:
                if token not in current_node.children:
                    current_node.children[token] = GoTrieNode()
                current_node = current_node.children[token]
            current_node.kjd_id = chunk_id

    def search_maximal_prefix_ids(self, track_a_string: str) -> Set[str]:
        if not track_a_string: return set()
        tokens = track_a_string.split("_")
        matched_ids = set()
        current_node = self.root
        for token in tokens:
            if token in current_node.children:
                current_node = current_node.children[token]
                if current_node.kjd_id is not None:
                    matched_ids.add(current_node.kjd_id)
            else:
                break
        return matched_ids

class ResearcherAgent:
    def __init__(self, db_path=DB_PATH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.trie = GoJosekiTrie()
        self.trie.build_from_chroma(self.collection)
        
        # Harness 外部工作記憶快取容器 (Harness State Containers)
        self.Ct = []  # Concentrated Cache (精選集)：儲存當前加載的 KJD 節點 ID 序列 (維持 DFS 順序)
        self.Dt = {}  # Document Cache (文檔快取)：儲存 ID 對應的真實文檔內容與結構化 metadata
        self.Vt = {}  # Verification Cache (驗證快取)：儲存 Go Sensei 產生的各個戰術 claim 及其 NLI 驗證結果
        self.Ht = []  # Harness Trajectory (決策軌跡記錄)：儲存 RAG 管線實時工具觸發步驟，用於論文評估導出
        self.step_counter = 0
        print(f"        正在載入原生英文 NLI 驗證模型...", flush=True)
        self.nli_model = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device="cpu")

    def reset_state(self):
        # 重置所有工作記憶與軌跡容器，防止跨題譜資料污染
        self.Ct, self.Dt, self.Vt, self.Ht = [], {}, {}, []
        self.step_counter = 0

    def execute_hierarchical_trie_retrieval(self, track_a_string: str, track_b_string: str) -> str:
        self.step_counter += 1
        
        # 4-fold 對稱座標標準化（將開局方向旋轉至右上角，僅用於 RAG 字典樹檢索，不污染原始軌跡）
        rotated_track_a = track_a_string
        if track_a_string:
            tokens = track_a_string.split("_")
            if tokens:
                first_token = tokens[0]
                start_idx = first_token.find("[")
                end_idx = first_token.find("]")
                if start_idx != -1 and end_idx != -1:
                    first_coord = first_token[start_idx+1:end_idx]
                    x0, y0 = parse_weiqi_coord(first_coord)
                    
                    sym_index = 0
                    if x0 >= 9 and y0 >= 9:    # 右上角
                        sym_index = 0
                    elif x0 < 9 and y0 >= 9:  # 左上角 -> 旋轉 270 度 以映射至右上角
                        sym_index = 6
                    elif x0 < 9 and y0 < 9:    # 左下角 -> 旋轉 180 度 以映射至右上角
                        sym_index = 3
                    elif x0 >= 9 and y0 < 9:   # 右下角 -> 旋轉 90 度 以映射至右上角
                        sym_index = 5
                        
                    rotated_tokens = []
                    COLS = "ABCDEFGHJKLMNOPQRST"
                    for tok in tokens:
                        s_idx = tok.find("[")
                        e_idx = tok.find("]")
                        if s_idx != -1 and e_idx != -1:
                            color = tok[:s_idx]
                            coord = tok[s_idx+1:e_idx]
                            tx, ty = parse_weiqi_coord(coord)
                            rx, ry = get_symmetric_coords(tx, ty, sym_index)
                            rotated_coord = f"{COLS[rx]}{ry+1}"
                            rotated_tokens.append(f"{color}[{rotated_coord}]")
                        else:
                            rotated_tokens.append(tok)
                    rotated_track_a = "_".join(rotated_tokens)
                    
        matched_leaf_ids = self.trie.search_maximal_prefix_ids(rotated_track_a)
        if not matched_leaf_ids:
            return "Deterministic RAG Alert: No matching joseki branch found in KJD."

        final_id_set = set()
        res_leaf = self.collection.get(ids=list(matched_leaf_ids), include=["metadatas"])
        for chunk_id, meta in zip(res_leaf['ids'], res_leaf['metadatas']):
            final_id_set.add(chunk_id)
            for pid in json.loads(meta.get("parents_ids", "[]")):
                final_id_set.add(pid)

        res_all = self.collection.get(ids=list(final_id_set), include=["documents", "metadatas"])
        for chunk_id, doc, meta in zip(res_all['ids'], res_all['documents'], res_all['metadatas']):
            self.Dt[chunk_id] = {"text": doc, "metadata": meta}

        children_map = {cid: [] for cid in final_id_set}
        parent_map = {}
        root_nodes = []

        for cid in final_id_set:
            pids = json.loads(self.Dt[cid]["metadata"].get("parents_ids", "[]"))
            closest_parent = None
            for pid in reversed(pids):
                if pid in final_id_set:
                    closest_parent = pid
                    break
            if closest_parent:
                children_map[closest_parent].append(cid)
                parent_map[cid] = closest_parent
            else:
                root_nodes.append(cid)

        for pid in children_map:
            children_map[pid].sort(key=lambda x: (self.Dt[x]["metadata"].get("move_count", 0), x))
        root_nodes.sort(key=lambda x: (self.Dt[x]["metadata"].get("move_count", 0), x))

        ordered_ids = []
        def dfs_traverse(node_id):
            ordered_ids.append(node_id)
            for child_id in children_map[node_id]:
                dfs_traverse(child_id)

        for r in root_nodes:
            dfs_traverse(r)
        
        self.Ct = ordered_ids
        track_b_list = track_b_string.split("_") if track_b_string else []

        markdown_output = "### [Go Joseki Library: Hierarchical Variation Tree Summary]\n"
        
        for cid in self.Ct:
            meta = self.Dt[cid]["metadata"]
            m_count = meta.get("move_count", 0)
            text_content = self.Dt[cid]["text"].strip()
            
            if m_count == 0 or not track_b_list:
                board_seq_str = "none"
            else:
                board_seq_str = " -> ".join(track_b_list[:m_count])
            
            if m_count == 0:
                markdown_output += f"#### [DocID: {cid}]\n"
                markdown_output += f"- Board Sequence: {board_seq_str}\n"
                markdown_output += f"- {text_content}\n\n---\n\n"
            else:
                markdown_output += f"##### [DocID: {cid}]\n"
                markdown_output += f"- Board Sequence: {board_seq_str}\n"
                
                parent_id = parent_map.get(cid)
                if parent_id:
                    sisters = children_map[parent_id]
                    if len(sisters) > 1:
                        idx = sisters.index(cid) + 1
                        markdown_output += f"- Lineage: Sister Branch {idx}/{len(sisters)} (Split from {parent_id})\n"
                
                markdown_output += f"- {text_content}\n\n"

        VIRTUAL_TREE_ID = "concentrated_tree_context"
        self.Dt[VIRTUAL_TREE_ID] = {
            "text": markdown_output,
            "metadata": {"type": "virtual_integrated_context"}
        }

        self.Ht.append({"step": self.step_counter, "tool": "hierarchical_trie_retrieval", "status": "SUCCESS", "result": f"Hydrated tree cached under '{VIRTUAL_TREE_ID}'."})
        return markdown_output

    def execute_verify_tool(self, claim: str) -> str:
        """終極融合演算法：線性時空路徑剪枝（方案 1）+ 語意功能性 NLI 驗證"""
        self.step_counter += 1
        
        if not self.Ct:
            obs = "Verification Failed: Memory cache is empty. Please run trie retrieval first."
            self.Ht.append({"step": self.step_counter, "tool": "verify", "status": "FAILED_CACHE", "result": obs})
            return obs

        leaf_id = max(self.Ct, key=lambda cid: self.Dt[cid]["metadata"].get("move_count", 0))
        pids = json.loads(self.Dt[leaf_id]["metadata"].get("parents_ids", "[]"))
        
        linear_path_ids = [pid for pid in pids if pid in self.Dt] + [leaf_id]
        
        # Use the leaf node's comment directly as the premise to prevent context dilution/noise in NLI.
        # Fallback to full path if leaf comment is empty.
        linear_story_premise = self.Dt[leaf_id]["text"].strip()
        if not linear_story_premise or "Kogo's Joseki Dictionary" in linear_story_premise:
            pure_comments = []
            for cid in linear_path_ids:
                text = self.Dt[cid]["text"].strip()
                if text and "Kogo's Joseki Dictionary" not in text:
                    pure_comments.append(text)
            linear_story_premise = " ".join(pure_comments)
            
        if not linear_story_premise:
            linear_story_premise = "No comprehensive text found along the matched linear path."

        try:
            result = self.nli_model(
                text=linear_story_premise,
                text_pair=claim,
                truncation="only_first",
                max_length=512
            )
            label = result[0]['label']
            confidence = result[0]['score']
        except Exception as e:
            result = self.nli_model(f"{linear_story_premise[:1500]} [SEP] {claim}", truncation=True, max_length=512)
            label = result[0]['label']
            confidence = result[0]['score']

        self.Vt[claim] = {"label": label, "score": confidence}
        print(f" (Claim: {claim})", end="", flush=True)
        
        if label == "entailment":
            status = "VERIFIED_TRUE"
            obs = f"NLI Verification Result: PASS (Entailed) (Confidence: {confidence:.2f})"
        elif label == "neutral":
            status = "VERIFIED_NEUTRAL"
            obs = f"NLI Verification Result: WARN (Neutral) (Confidence: {confidence:.2f})"
        else:
            status = "VERIFIED_FALSE"
            obs = f"NLI Verification Result: FAIL (Contradicted) (Confidence: {confidence:.2f})"
            
        self.Ht.append({"step": self.step_counter, "tool": "verify", "status": status, "claim": claim, "result": obs})
        return obs

    def render_state(self, current_tokens: int, threshold: int, for_terminal: bool = False) -> str:
        state_str = "\n[當前外部工作記憶快取 (Harness State)]\n"
        state_str += f"- 精選集 (Ct) 當前加載標籤: " + ", ".join(self.Ct) + "\n"
        state_str += "- 📚 記憶庫當前快取整合文檔狀態:\n"
        if "concentrated_tree_context" in self.Dt:
            if for_terminal:
                state_str += f"  * [concentrated_tree_context]: 結構化定式變化樹已成功生成並鎖定快取區。\n"
            else:
                state_str += f"  * [concentrated_tree_context]:\n{self.Dt['concentrated_tree_context']['text']}\n"
        else:
            state_str += "  (目前精選池無文檔，等待 Go Sensei 發起檢索工具)\n"
        return state_str

# --- 【終極修正】Go Sensei 提示詞：雙重平衡範例，強制規範首回合檢索 ---
PM_SYSTEM_PROMPT = """You are the Lead AI Go Sensei (Researcher Agent) in a Go Teaching System, operating entirely in English.
Your sole responsibility is to orchestrate external deterministic tools to analyze the student's opening layout.

[CRITICAL INDEPENDENT OPERATION RULE]
- ABSOLUTE PROHIBITION: Do NOT output any raw conversational text introduction like 'Thinking Process:', 'Analysis:', or fillers outside XML tags. You MUST start your response immediately with the XML tags.

[DECISION WORKFLOW POLICY]
1. **Turn 1**: You MUST invoke the `hierarchical_trie_retrieval` tool first to populate the cache memory. Do NOT call `verify` on your first turn because the memory cache is currently empty.
2. **Turn 2**: After receiving the loaded tree structure text inside the Harness State, synthesize a conceptual tactical claim in pure English, and invoke the `verify` tool to validate your tactical hypothesis.
   - [CRITICAL CLAIM RULE]: The synthesized claim MUST be written in pure natural English. ABSOLUTELY FORBIDDEN to include any coordinate lists (e.g. R17, Q16), color codes (e.g. B[R17], W[Q16]), coordinate chains with underscores (e.g. B[R17]_W[Q16]...), or DocIDs. Keep the claim concise, conceptual, and focused on strategic patterns (e.g., whether the sequence is a trick play, represents a balanced outcome, or allows Black to harass White's stones).
   - [NLI ALIGNMENT RULE]: To ensure NLI entailment, try to use direct factual phrasing that closely matches the document text in the Harness State (e.g., "Black may harass the two White stones, or play elsewhere" or "White's move is a trick play"), rather than adding high-level post-processing commentary (like "represents a balanced outcome" or "in this joseki variation") which the NLI model classifies as neutral.
3. **Turn 3**: Once verified, invoke `conclude_search` to complete the workflow.

XML Tags Format Restriction:
You MUST respond strictly using the dual-tag format (<reasoning> followed by <action>). Start immediately with <reasoning>.

Example for Turn 1 (Mandatory Retrieval):
<reasoning>This is the first turn. The memory cache is empty, so I must first invoke the trie retrieval tool to load the historical joseki variation nodes.</reasoning>
<action>{"tool": "hierarchical_trie_retrieval", "params": {}}</action>

Example for Turn 2 (Verification):
<reasoning>The tree is now fully populated in the context. I will synthesize an abstractive tactical claim and pass it to the NLI verification tool.</reasoning>
<action>{"tool": "verify", "params": {"claim": "The played variation represents a standard joseki branch responding to the shoulder hit."}}</action>
"""

PM_PROMPT_TEMPLATE = """With the following real-time layout data:
- Core Active Zone: {active_zone}

[Current Student Board Layout Sequence (Track B)]
{track_b_string}

[Structured RAG Engine Cache Status]
{harness_state}

Please execute your logical Decision Policy via the next tool action."""

# --- 【終極修正】Lecturer 提示詞：消滅括號佔位符，嚴防鏡像抄襲 ---
LECTURER_SYSTEM_PROMPT = """You are a Grandmaster Professional Go Player and Core Lecturer. Your task is to compile a professional, elegant opening review log in English by combining three key inputs:
1. **Board Move Sequence (`Localized Human Tactic Sequence`)**: The actual sequence of moves played on the board, described in order.
2. **Harness State (`Deduplicated RAG Output`)**: The audited KJD joseki variations database, containing standard trees, trick plays, and correct follow-up responses.
3. **AI Evaluation (`AI Evaluation Matrix`)**: KataGo's calculated winrate and score lead metrics.

### [UNDERSTANDING THE BOARD SEQUENCE FORMAT]
The `Localized Human Tactic Sequence` is formatted as:
`[Step Number]. [Player Color][Coordinate] ([Board Spatial Location])_[Identified Go Tactic Name]`
- Example: `1. B[Q16 (top-right)]_4-4 Point`
  - `1.`: The step number (played by Black as the 1st move).
  - `B`: Player Color (Black).
  - `Q16`: Board coordinate.
  - `top-right`: The exact spatial location of this coordinate on the board (e.g., top-left, top-right, bottom-left, bottom-right, left, right, top, bottom, center). Use this exact spatial location in your descriptions to avoid direction errors (e.g., do NOT call Q16 a "bottom" or "left" point).
  - `4-4 Point`: The standard Go tactic identified for this play (e.g., star point).

### [STRICT CONTENT DIRECTIVES]
1. **Commentary Reference Format**: When referring to actual moves in your review log, you MUST combine player, step number, and coordinate in the format "Black 1 (Q16)" or "White 2 (D4)" (based on the numbered sequence). For example: "Black 1 (Q16) establishes the star point, and White 2 (D4) responds at the bottom-left corner." Do NOT write them simply as "Q16" or "1. B[Q16]".
2. **Strict Tactic Binding**: When describing any coordinate, you MUST strictly bind it to its identified tactic name from the sequence. For example, if S4 is marked as Regular Move, do NOT describe it as a "Small Knight" or "Enclosure".
3. **Actual vs. Hypothetical Divergence**:
   - **Actual Moves**: Describe what actually happened in the game using active facts (e.g., "Black 1 (Q16) was played").
   - **Hypothetical Variations**: When discussing alternative correct moves or joseki paths from the RAG tree that were NOT played in the game, you MUST use hypothetical phrasing (e.g., "In standard joseki, White could have played R14..."). Do NOT describe them as actual plays.
4. **Structured Review Layout**: Organize your review into three distinct sections:
   - **Layout Framework & Strategic Theme**: Analyze the overall opening framework and the strategic ideas behind the main layout.
   - **Core Joseki Variations & Tactical Decision Points**: Focus on local joseki choices and critical variations (trick plays, follow-ups).
   - **AI Evaluation & Global Judgment**: Integrate KataGo's winrate/score metrics to discuss the balance of the board.

### [OUTPUT FORMAT RESTRICTIONS]
- If you write an internal thinking process, you MUST wrap it inside the <think> and </think> tags (keep it under 50 words).
- Start your response directly with the <reasoning> tag.
- You MUST wrap your final lecture report inside the <review_commentary> tag.
"""

LECTURER_PROMPT_TEMPLATE = """Here is the finalized audited dossier for your lecture:
- Total Moves: {total_steps}
- AI Evaluation Matrix: {katago_analysis}
- Localized Human Tactic Sequence: {joseki_sequence_human}

[Deduplicated RAG Output - Pure Markdown Tree Context]
{harness_state}

Please compile the opening review log inside the requested tags in professional English."""

# --- 【加固型】Translator 提示詞 ---
TRANSLATOR_SYSTEM_PROMPT = """你是一位專職的圍棋講評翻譯官與台灣本地化轉譯專家。
你的任務是將講師產出的高階全英文專家覆盤日誌，完全轉譯為語氣親切、流暢自然、且完全符合台灣圍棋界學術術語習慣的繁體中文教學覆盤報告。

最高格式限制：
- 如果你有任何內心思考，必須完全包裹在 <think> 與 </think> 標籤中，且字數必須控制在 30 字以內。
1. 嚴禁輸出任何 XML 標籤（如 <reasoning> 或 <review_commentary>）。
2. 直接輸出最終排版精美的繁體中文覆盤報告正文，開頭第一個字就必須是正文。
3. 術語強對齊約束（Strict Vocabulary Grounding）：
   在轉譯正文中的任何座標與棋形術語時，你必須嚴格比對並採用「[對照用真實落子與棋形中文序列]」中由系統預先轉換好的中文術語。
   嚴禁自發性地直譯英文日誌中的數字座標（例如將英文 4-4 或 3-4 直譯為四四或三三），一切翻譯標準均以對照序列中的中文對應詞為準，以確保術語在物理事實上的絕對精準。
4. 手順序號翻譯規則（Move Referencing Alignment）：
   請將英文日誌中的 "Black 1 (Q16)" 翻譯為「黑1（Q16）」，"White 2 (D4)" 翻譯為「白2（D4）」，依此類推，以符合台灣圍棋著述中以「黑1」、「白2」指代特定落子的習慣。"""
def parse_weiqi_coord(coord_str: str) -> tuple:
    col_char = coord_str[0].upper()
    row_num = int(coord_str[1:])
    COLS = "ABCDEFGHJKLMNOPQRST"
    x = COLS.find(col_char)
    y = row_num - 1
    return x, y

def get_spatial_sector(x: int, y: int) -> dict:
    left = x <= 5
    right = x >= 13
    bottom = y <= 5
    top = y >= 13
    
    if left and top:
        return {"en": "top-left", "zh": "左上"}
    elif right and top:
        return {"en": "top-right", "zh": "右上"}
    elif left and bottom:
        return {"en": "bottom-left", "zh": "左下"}
    elif right and bottom:
        return {"en": "bottom-right", "zh": "右下"}
    elif left:
        return {"en": "left", "zh": "左側"}
    elif right:
        return {"en": "right", "zh": "右側"}
    elif top:
        return {"en": "top", "zh": "上方"}
    elif bottom:
        return {"en": "bottom", "zh": "下方"}
    else:
        return {"en": "center", "zh": "中腹"}

def run_agentic_multi_loop(feature_vars: Dict[str, Any], katago_analysis: str, track_a_str: str, track_b_str: str, researcher: ResearcherAgent, max_turns: int) -> Dict[str, str]:
    turn_history = []
    numbered_sequence_human = ""
    joseki_sequence_human_zh = ""
    
    # 1. Go Sensei 推論與驗證階段
    for turn in range(max_turns):
        harness_state_text_llm = researcher.render_state(0, 8192, for_terminal=False)
        harness_state_text_term = researcher.render_state(0, 8192, for_terminal=True)
        
        print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
        print(f"{CYAN}{BOLD}[Harness 狀態實時監視面板] ── 題譜: {feature_vars['question_id']} | Turn: {turn+1}/{max_turns}{RESET}")
        print(harness_state_text_term.strip())
        print(f"{CYAN}{BOLD}========================================================================{RESET}\n", flush=True)

        user_content = PM_PROMPT_TEMPLATE.format(
            active_zone=feature_vars["active_zone"],
            track_b_string=track_b_str, harness_state=harness_state_text_llm
        )
        
        try:
            print(f"{YELLOW}Go Sensei 正在進行純英文戰術評估與決策...{RESET}", flush=True)
            res = requests.post(API_URL, json={
                "model": MODEL_NAME, 
                "messages": [{"role": "system", "content": PM_SYSTEM_PROMPT}] + turn_history + [{"role": "user", "content": user_content}],
                "temperature": 0.0,
                "max_tokens": 2048
            }, timeout=120).json()
            pm_output = res["choices"][0]["message"]["content"]
            
            # 【終極修正 ── 局部回放短期記憶】
            # 在 3 輪對話內部，我們必須保留完整的 pm_output 讓模型擁有「草稿紙記憶」以實現正常的決策狀態移轉。
            # 該區域變數 turn_history 在本題結束後會自動釋放，完美落實了跨題顯存與 Token 的強隔離流水線精神。
            turn_history.append({"role": "assistant", "content": pm_output})
            
            action_idx = pm_output.rfind('<action>')
            if action_idx != -1:
                raw_action = pm_output[action_idx + len('<action>'):].strip()
                close_idx = raw_action.find('</action>')
                if close_idx != -1:
                    raw_action = raw_action[:close_idx].strip()
            else:
                raw_action = pm_output.strip()
            action_data = robust_json_loads(raw_action)
            
            tool_name = action_data.get("tool")
            params = action_data.get("params", {})
            
            if tool_name == "conclude_search": break
                
            print(f"{YELLOW}Harness 觸發執行決定性工具: [{tool_name}]{RESET}", end="", flush=True)
            if tool_name == "hierarchical_trie_retrieval":
                obs = researcher.execute_hierarchical_trie_retrieval(track_a_str, track_b_str)
            elif tool_name == "verify":
                obs = researcher.execute_verify_tool(params.get("claim", ""))
            else:
                obs = f"Unsupported tool error: '{tool_name}'"
                
            print(f" ──► {GREEN}[工具執行完畢]{RESET}", flush=True)
            turn_history.append({"role": "user", "content": f"【Tool Observation】\n{obs}"})
        except Exception as e:
            print(f"決策迴圈異常中編: {e}")
            if 'pm_output' in locals():
                print(f"--- Raw LLM Output ---\n{pm_output}\n---------------------")
            if 'raw_action' in locals():
                print(f"--- Raw Action String ---\n{raw_action}\n---------------------")
            break

    # 2. Grandmaster Lecturer 講評編寫階段
    print(f"🎓 {YELLOW}Grandmaster Lecturer 正在編寫高階英文評盤日誌...{RESET}", flush=True)
    english_review = "未成功生成英大師講評內容。"
    try:
        real_knowledge_context = ""
        if "concentrated_tree_context" in researcher.Dt:
            real_knowledge_context = researcher.Dt["concentrated_tree_context"]["text"]
        raw_seq = feature_vars.get("joseki_sequence_human", "")
        items = [x.strip() for x in raw_seq.split(",") if x.strip()]
        numbered_items = []
        for idx, item in enumerate(items, start=1):
            if "_" in item:
                prefix, tactic = item.rsplit("_", 1)
                color_char = "B" if prefix.startswith("Black") else "W"
                coord = prefix[prefix.find("[")+1:prefix.find("]")]
                cx, cy = parse_weiqi_coord(coord)
                sector = get_spatial_sector(cx, cy)
                numbered_items.append(f"{idx}. {color_char}[{coord} ({sector['en']})]_{tactic}")
            else:
                numbered_items.append(f"{idx}. {item}")
        numbered_sequence_human = ", ".join(numbered_items)
            
        res_lec = requests.post(API_URL, json={
            "model": MODEL_NAME, 
            "messages": [{"role": "system", "content": LECTURER_SYSTEM_PROMPT}, {
                "role": "user", "content": LECTURER_PROMPT_TEMPLATE.format(
                    total_steps=feature_vars["total_steps"], katago_analysis=katago_analysis,
                    joseki_sequence_human=numbered_sequence_human, harness_state=real_knowledge_context
                )
            }], 
            "temperature": 0.4,
            "presence_penalty": 0.05,
            "frequency_penalty": 0.05,
            "max_tokens": 8192
        }, timeout=180).json()
        raw_english_review = res_lec["choices"][0]["message"]["content"]
        
        comm_idx = raw_english_review.rfind('<review_commentary>')
        if comm_idx != -1:
            english_review = raw_english_review[comm_idx + len('<review_commentary>'):].strip()
            close_idx = english_review.find('</review_commentary>')
            if close_idx != -1:
                english_review = english_review[:close_idx].strip()
        else:
            english_review = raw_english_review.strip()
        
    except Exception as e: 
        return {"lecturer_output": f"講師模組崩潰: {e}", "translator_output": "未執行（因講師層崩潰）"}

    # 3. Translation Agent 台灣繁中翻譯階段
    print(f"{GREEN}Translation Agent 啟動！正在轉譯為台灣圍棋講評文案...{RESET}", flush=True)
    try:
        # Dynamically load TRANSLATION_MAP from source-dataset/sabaki_names_list.py
        t_map = {}
        translation_map_path = os.path.join(os.path.dirname(__file__), "source-dataset", "sabaki_names_list.py")
        if os.path.exists(translation_map_path):
            with open(translation_map_path, "r", encoding="utf-8") as f:
                content = f.read()
                loc = {}
                try:
                    exec(content, globals(), loc)
                    t_map = loc.get("TRANSLATION_MAP", {})
                except Exception:
                    pass
        if not t_map:
            from go_tactic_analysis import TRANSLATION_MAP as fallback_map
            t_map = fallback_map

        eng_sequence = feature_vars.get("joseki_sequence_human", "")
        items = [x.strip() for x in eng_sequence.split(",") if x.strip()]
        zh_items = []
        for idx, item in enumerate(items, start=1):
            if "_" in item:
                prefix, eng_tactic = item.rsplit("_", 1)
                color_zh = "黑" if prefix.startswith("Black") else "白"
                coord = prefix[prefix.find("[")+1:prefix.find("]")]
                zh_tactic = t_map.get(eng_tactic, eng_tactic)
                cx, cy = parse_weiqi_coord(coord)
                sector = get_spatial_sector(cx, cy)
                zh_items.append(f"{idx}. {color_zh}[{coord}（{sector['zh']}）]_{zh_tactic}")
            else:
                zh_items.append(f"{idx}. {item}")
        joseki_sequence_human_zh = ", ".join(zh_items)

        user_prompt = f"""請將以下英文覆盤日誌轉譯為繁體中文教學覆盤報告，務必對照並使用以下中文序列中的棋形術語（如「三三」、「肩衝/尖衝」、「長」、「小飛」、「碰」、「拐」等），且不可搞錯黑白子的下子順序：

[對照用真實落子與棋形中文序列]
{joseki_sequence_human_zh}

[待翻譯英文日誌正文]
{english_review}"""

        res_trans = requests.post(API_URL, json={
            "model": MODEL_NAME, 
            "messages": [{"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            "temperature": 0.3,
            "max_tokens": 4096
        }, timeout=180).json()
        chinese_report = res_trans["choices"][0]["message"]["content"]
        
        chinese_report = strip_thinking_process(chinese_report)
            
        return {
            "lecturer_output": english_review, 
            "translator_output": chinese_report,
            "joseki_sequence_human": numbered_sequence_human,
            "joseki_sequence_human_zh": joseki_sequence_human_zh
        }
    except Exception as e: 
        return {
            "lecturer_output": english_review, 
            "translator_output": f"轉譯層異常: {e}",
            "joseki_sequence_human": numbered_sequence_human,
            "joseki_sequence_human_zh": joseki_sequence_human_zh
        }

def run_experiment_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(SGF_PROMPT_DIR): return
    all_files = [f for f in os.listdir(SGF_PROMPT_DIR) if os.path.isfile(os.path.join(SGF_PROMPT_DIR, f)) and not f.startswith('.')]
    if not all_files: return

    print(f"\n📂 成功偵測到 {len(all_files)} 個測試棋譜。啟動時空結構化決定性 RAG 流水線...")
    fe_extractor = GoFeatureExtractor()
    researcher_agent = ResearcherAgent(db_path=DB_PATH)
    experiment_results = []

    for idx, qipu_filename in enumerate(all_files):
        try:
            features = fe_extractor.extract_features_from_file(qipu_filename)
            if "error" in features: continue
            
            track_b_str = features["track_b_string"]
            track_a_str = features["track_a_string"]
            payload = features["kata_and_tactic_payload"]
            
            feature_vars = {
                "question_id": qipu_filename,
                "total_steps": payload["total_steps"],
                "winrate_perspective": payload["winrate_perspective"],
                "score_lead": payload["score_lead"],
                "active_zone": "右上角",
                "joseki_sequence_human": payload["joseki_sequence_human"]
            }

            katago_analysis = f"Winrate: {feature_vars['winrate_perspective']} | Score Lead: {feature_vars['score_lead']} pts"
            researcher_agent.reset_state()
            
            loop_outputs = run_agentic_multi_loop(
                feature_vars=feature_vars, katago_analysis=katago_analysis,
                track_a_str=track_a_str, track_b_str=track_b_str,
                researcher=researcher_agent, max_turns=3
            )
            
            experiment_results.append({
                "question_id": qipu_filename,
                "harness_context_snapshot": researcher_agent.render_state(0, 8192, for_terminal=False),
                "trajectory_log": researcher_agent.Ht,
                "joseki_sequence_human": loop_outputs.get("joseki_sequence_human", ""),
                "lecturer_english_output": loop_outputs["lecturer_output"],  
                "joseki_sequence_human_zh": loop_outputs.get("joseki_sequence_human_zh", ""),
                "final_localized_output": loop_outputs["translator_output"]  
            })
            print(f"  成功棋譜 '{qipu_filename}' 自動化測試完美完賽並成功導出日誌。")
            
            # ======= 顯存與記憶體大掃除 =======
            del loop_outputs  
            gc.collect()              
            torch.cuda.empty_cache()  
            
        except Exception as e:
            print(f"❌ 棋譜 '{qipu_filename}' 發生異常: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        json.dump(experiment_results, outfile, ensure_ascii=False, indent=2)
    print(f"\n全量實驗管線完美收官。黃金數據已導出至：{OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment_pipeline()