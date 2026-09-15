# token_eval.py
# 目的：比較 NaiveLLM 與實驗系統（Multi-Agentic RAG）在處理 bjjq_1 棋盤解說時的 Token 消耗量
# 輸出：token_eval.json

import os
os.environ["PYTORCH_JIT_DISABLE"] = "1"
os.environ["TORCH_CUDA_FUSER_DISABLE"] = "1"

import json
import time
import requests

# ============================================================
# 基礎設定
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(CURRENT_DIR, "eval-results")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "token_eval.json")
API_URL      = os.getenv("API_URL", "http://localhost:8000/v1/chat/completions")
TARGET_FILE  = "bjjq_1"

# bjjq_1 的原始 SGF 內容（直接嵌入，供 NaiveLLM 使用）
BJJQ1_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[Sabaki:0.52.2]SZ[19]DT[2026-07-11]GN[bjjq_1];B[pd];W[cp];B[pp];W[dc];B[np];W[de];B[jp];W[rp];B[jc];W[qn];B[fp];W[dq];B[ep];W[eq];B[fq];W[cn];B[ql];W[qq];B[qo];W[ro];B[pn];W[rn];B[pm])"

# ============================================================
# 工具函式：動態取得模型名稱
# ============================================================
def get_active_model_name(api_url: str) -> str:
    try:
        base_url = api_url.rsplit('/', 2)[0] + "/models"
        res = requests.get(base_url, timeout=5).json()
        if "data" in res and len(res["data"]) > 0:
            return res["data"][0]["id"]
    except Exception:
        pass
    return os.getenv("MODEL_NAME", "qwen3.2-27b")

# ============================================================
# 實驗一：NaiveLLM（直接輸入 SGF，無 RAG，無工具）
# ============================================================

NAIVE_SYSTEM_PROMPT = """你是一位專業圍棋講師。你的任務是根據使用者提供的 SGF 棋譜，生成一份完整且專業的開局覆盤報告。

報告須包含以下三個部分：
1. **布局框架與戰略主題**：分析整體布局框架及主要戰略意圖。
2. **定式變化與關鍵決策點**：分析各局部定式選擇與關鍵手段（騙著、後續應對等）。
3. **整體評估**：綜合評估棋局走向與黑白雙方的優劣。

請直接輸出繁體中文覆盤報告，不需要任何額外說明。"""

NAIVE_USER_PROMPT = f"""請根據以下 SGF 棋譜，生成完整的開局覆盤報告：

[SGF 棋譜]
{BJJQ1_SGF}

請分析每一手棋的意圖，說明定式選擇，並評估整體局勢。"""


def run_naive_llm(model_name: str) -> dict:
    """
    NaiveLLM 實驗：直接餵入 SGF 字串，讓 LLM 生成覆盤。
    不使用任何 RAG、不使用任何工具。
    回傳 token 使用統計與生成內容。
    """
    print("\n" + "="*60)
    print("[實驗一] NaiveLLM 開始執行...")
    print("="*60)

    messages = [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user",   "content": NAIVE_USER_PROMPT}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.6,     # Qwen3 thinking 模式建議 0.6
        "top_p": 0.95,          # Qwen3 thinking 模式建議 0.95
        "max_tokens": 16384,    # 對齊 llama.cpp 伺服器上限，不額外截斷
        "extra_body": {
            "enable_thinking": True  # 啟動 Qwen3 深度推理（博弈思考鏈）
        }
    }

    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=600).json()
        elapsed   = time.time() - start_time

        usage       = response.get("usage", {})
        raw_content = response["choices"][0]["message"]["content"]

        # 取出 thinking_content（若 API 回傳於獨立欄位）
        thinking_content = response["choices"][0]["message"].get("reasoning_content", "")

        # 若 thinking 包含在 content 的 <think>...</think> 標籤內，則分離出來
        import re as _re
        think_match = _re.search(r'<think>(.*?)</think>', raw_content, _re.DOTALL)
        if think_match and not thinking_content:
            thinking_content = think_match.group(1).strip()
        # 去除 think 標籤後取得純正文
        output_text = _re.sub(r'<think>.*?</think>', '', raw_content, flags=_re.DOTALL).strip()

        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens      = usage.get("total_tokens", 0)
        thinking_tokens   = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        # vLLM 有時把 thinking tokens 放在 completion_tokens_details
        thinking_tokens_detail = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        print(f"  [OK] NaiveLLM 完成。耗時：{elapsed:.1f}s")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"    └─ Thinking tokens (reasoning): {thinking_tokens_detail}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Thinking preview: {thinking_content[:100]}..." if thinking_content else "  (無 thinking content)")

        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens_detail,
            "total_tokens": total_tokens,
            "thinking_preview": (thinking_content[:500] + "...") if len(thinking_content) > 500 else thinking_content,
            "output_preview": (output_text[:500] + "...") if len(output_text) > 500 else output_text
        }
    except Exception as e:
        print(f"  [ERR] NaiveLLM 發生異常: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


# ============================================================
# 實驗二：實驗系統（Multi-Agentic RAG 完整管線）
# ============================================================

class TokenAccumulator:
    """攔截 API 呼叫並累計 token 使用量的 Wrapper"""
    def __init__(self, real_post_fn):
        self._real_post = real_post_fn
        self.total_prompt_tokens     = 0
        self.total_completion_tokens = 0
        self.total_tokens            = 0
        self.call_log                = []

    def patched_post(self, url, json=None, timeout=None, **kwargs):
        response_obj = self._real_post(url, json=json, timeout=timeout, **kwargs)

        try:
            data  = response_obj.json()
            usage = data.get("usage", {})
            pt    = usage.get("prompt_tokens", 0)
            ct    = usage.get("completion_tokens", 0)
            tt    = usage.get("total_tokens", 0)

            self.total_prompt_tokens     += pt
            self.total_completion_tokens += ct
            self.total_tokens            += tt

            call_label = "unknown"
            if json and "messages" in json:
                sys_content = json["messages"][0].get("content", "") if json["messages"] else ""
                if "Go Sensei" in sys_content or "Go Teaching System" in sys_content:
                    call_label = "go_sensei"
                elif "Grandmaster" in sys_content or "Lecturer" in sys_content:
                    call_label = "lecturer"
                elif "翻譯官" in sys_content or "繁體中文" in sys_content:
                    call_label = "translator"

            self.call_log.append({
                "role": call_label,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt
            })
            print(f"    [Token Monitor] 角色={call_label} | prompt={pt} | completion={ct} | total={tt}")

        except Exception as parse_err:
            print(f"    [Token Monitor] 警告：無法解析 usage 欄位: {parse_err}")

        return response_obj

    def summary(self) -> dict:
        return {
            "total_prompt_tokens":     self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens":            self.total_tokens,
            "per_call_log":            self.call_log
        }


def run_experimental_system(model_name: str) -> dict:
    """
    實驗系統：執行完整的 Multi-Agentic RAG 管線（僅針對 bjjq_1）。
    使用 TokenAccumulator 攔截所有 LLM API 呼叫並累計 token。
    """
    print("\n" + "="*60)
    print("[實驗二] Multi-Agentic RAG 系統開始執行...")
    print("="*60)

    import importlib.util, sys, types

    module_path = os.path.join(CURRENT_DIR, "sgf_rag_multi-agentic-llm.py")
    spec        = importlib.util.spec_from_file_location("sgf_rag_module", module_path)
    sgf_module  = importlib.util.module_from_spec(spec)

    accumulator = TokenAccumulator(requests.post)

    fake_requests = types.ModuleType("requests")
    fake_requests.post = accumulator.patched_post
    fake_requests.get  = requests.get

    original_requests = sys.modules.get("requests")
    sys.modules["requests"] = fake_requests

    try:
        spec.loader.exec_module(sgf_module)
    finally:
        if original_requests is not None:
            sys.modules["requests"] = original_requests
        else:
            del sys.modules["requests"]

    start_time = time.time()
    try:
        fe_extractor     = sgf_module.GoFeatureExtractor()
        researcher_agent = sgf_module.ResearcherAgent(db_path=sgf_module.DB_PATH)

        features = fe_extractor.extract_features_from_file(TARGET_FILE)
        if "error" in features:
            raise RuntimeError(f"特徵提取失敗: {features['error']}")

        track_b_str = features["track_b_string"]
        track_a_str = features["track_a_string"]
        payload     = features["kata_and_tactic_payload"]

        feature_vars = {
            "question_id":           TARGET_FILE,
            "total_steps":           payload["total_steps"],
            "winrate_perspective":   payload["winrate_perspective"],
            "score_lead":            payload["score_lead"],
            "active_zone":           "右上角",
            "joseki_sequence_human": payload["joseki_sequence_human"]
        }

        katago_analysis = f"Winrate: {feature_vars['winrate_perspective']} | Score Lead: {feature_vars['score_lead']} pts"
        researcher_agent.reset_state()

        loop_outputs = sgf_module.run_agentic_multi_loop(
            feature_vars=feature_vars,
            katago_analysis=katago_analysis,
            track_a_str=track_a_str,
            track_b_str=track_b_str,
            researcher=researcher_agent,
            max_turns=3
        )

        elapsed       = time.time() - start_time
        token_summary = accumulator.summary()

        print(f"\n  [OK] 實驗系統完成。耗時：{elapsed:.1f}s")
        print(f"  Total prompt tokens: {token_summary['total_prompt_tokens']}")
        print(f"  Total completion tokens: {token_summary['total_completion_tokens']}")
        print(f"  Total tokens (all LLM calls): {token_summary['total_tokens']}")

        translator_out = loop_outputs.get("translator_output", "")
        return {
            "status":             "success",
            "elapsed_seconds":    round(elapsed, 2),
            "prompt_tokens":      token_summary["total_prompt_tokens"],
            "completion_tokens":  token_summary["total_completion_tokens"],
            "total_tokens":       token_summary["total_tokens"],
            "per_agent_call_log": token_summary["per_call_log"],
            "output_preview":     (translator_out[:300] + "...") if len(translator_out) > 300 else translator_out
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERR] 實驗系統發生異常: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status":             "error",
            "error":              str(e),
            "elapsed_seconds":    round(elapsed, 2),
            "prompt_tokens":      accumulator.total_prompt_tokens,
            "completion_tokens":  accumulator.total_completion_tokens,
            "total_tokens":       accumulator.total_tokens,
            "per_agent_call_log": accumulator.call_log
        }


# ============================================================
# 主程式
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "#"*60)
    print("  Token 消耗量評估實驗 — bjjq_1")
    print("  比較：NaiveLLM  vs  Multi-Agentic RAG 系統")
    print("#"*60)

    model_name = get_active_model_name(API_URL)
    print(f"\n[系統] 偵測到模型名稱: {model_name}\n")

    naive_result = run_naive_llm(model_name)
    rag_result   = run_experimental_system(model_name)

    output = {
        "experiment_meta": {
            "target_game": TARGET_FILE,
            "model_name":  model_name,
            "api_url":     API_URL,
            "sgf_content": BJJQ1_SGF
        },
        "naive_llm": {
            "description": "直接輸入 SGF，無 RAG，無工具，單次 LLM 呼叫",
            **naive_result
        },
        "experimental_system": {
            "description": "完整 Multi-Agentic RAG 管線（Go Sensei × N輪 + Lecturer + Translator）",
            **rag_result
        },
        "comparison_summary": {
            "naive_total_tokens":      naive_result.get("total_tokens", 0),
            "rag_total_tokens":        rag_result.get("total_tokens", 0),
            "token_difference":        rag_result.get("total_tokens", 0) - naive_result.get("total_tokens", 0),
            "rag_overhead_multiplier": round(
                rag_result.get("total_tokens", 0) / max(naive_result.get("total_tokens", 1), 1), 2
            )
        }
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("Token 消耗量對比摘要")
    print("="*60)
    print(f"  NaiveLLM 總 Token:            {output['comparison_summary']['naive_total_tokens']:>8,}")
    print(f"  Multi-Agentic RAG 總 Token:   {output['comparison_summary']['rag_total_tokens']:>8,}")
    print(f"  差值（RAG - Naive）:           {output['comparison_summary']['token_difference']:>8,}")
    print(f"  RAG 相對倍數:                  {output['comparison_summary']['rag_overhead_multiplier']:>8.2f}x")
    print(f"\n[OK] 結果已寫入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
