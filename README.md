<div align="center">

# 🎓 狀態驅動的圍棋解說：結合 Harness 狀態機與檢索增強多智能體大語言模型之研究
### State-Driven Go Commentary: Harnessing State Machines and Retrieval-Augmented Multi-Agent LLMs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/KataGo-v1.15-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" />
  <img src="https://img.shields.io/badge/vLLM-0.19.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.1.1-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" />
</p>

---

</div>

## 📖 1. 專案簡介 (Project Abstract)

本專案為碩士論文 **《狀態驅動的圍棋解說：結合 Harness 狀態機與檢索增強多智能體大語言模型之研究》** 的官方開源程式碼庫。

圍棋開局定式解說需要極高的空間座標精準度與權威定式知識。本研究提出結合 **Trie 字典樹確定性 RAG**、**Harness 外部工作記憶狀態機** 與 **NLI 戰術推理驗證** 之雙階多 Agent 協作架構（Multi-Agentic LLM Pipeline），能有效解決傳統大語言模型在圍棋領域產生的「座標幻覺」與「語意不對齊」問題，自動生成具備台灣圍棋專業術語之教學覆盤報告。

---

## 📂 2. 資料夾結構 (Directory Structure)

```
State-Driven Go Commentary/
├── README.md                      # [專案說明] 系統架構、模組功能、資料集與執行指引
├── requirements.txt               # [依賴套件] 套件清單與 Conda / venv / uv 安裝指南
│
├── src/                           # 核心程式碼模組 (Source Code)
│   ├── sgf_rag_multi-agentic-llm.py # ★ 主實驗進入點 (Trie RAG + Harness Multi-Agent)
│   ├── go_feature_extraction.py    # SGF 棋譜特徵提取器 (Track A/B & KataGo Payload)
│   ├── go_tactic_analysis.py       # 4-Fold 座標對稱變換與術語對照模組
│   ├── token_eval.py               # Token 消耗與推論成本統計工具
│   └── outputs_processing.py       # LLM 輸出清理與 JSON 轉譯工具
│
├── data/                          # 數據集目錄 (Datasets)
│   ├── source-qipuNsgf/            # 測試用 SGF 圍棋棋譜資料集 (包含多元開局形勢)
│   └── source-dataset/             # 圍棋術語與 Sabaki 術語對照數據 (sabaki_names_list.py)
│
├── go_knowledge_base/             # 圍棋定式向量知識庫目錄 (向量資料庫構建掛載點)
│
└── eval-results/                  # 論文實驗評估結果
    ├── sgf_rag_multi-agentic-llm.json # ★ 主要原始實驗紀錄 (含 Harness 快取與軌跡)
    ├── output_processed.json          # 結構化後處理與標籤清洗數據
    ├── token_eval.json                # Token 消耗與推論成本統計數據
    └── llm_as_a_judge.json            # LLM 裁判自動化多維度評分結果
```

---

## 💻 3. 核心程式碼模組說明 (Codebase Summary)

<table width="100%">
  <thead>
    <tr>
      <th width="32%">檔案路徑</th>
      <th width="68%">核心功能與職責說明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>src/sgf_rag_multi-agentic-llm.py</code></td>
      <td><b>主實驗流水線進入點</b>：控制 Harness 狀態機 (包含 <code>Ct</code> 精選集, <code>Dt</code> 文檔快取, <code>Vt</code> 驗證快取, <code>Ht</code> 軌跡)，整合 Trie 字典樹檢索、mDeBERTa-v3 NLI 戰術驗證、Grandmaster Lecturer 講評生成與 Translation 台灣繁中術語轉譯。</td>
    </tr>
    <tr>
      <td><code>src/go_feature_extraction.py</code></td>
      <td><b>SGF 棋譜特徵提取器</b>：解析 SGF 棋譜檔案，自動分離 Track A (帶座標標籤軌跡) 與 Track B (純落子順序)，並呼叫 KataGo 評估引擎取得勝率與目數差數據。</td>
    </tr>
    <tr>
      <td><code>src/go_tactic_analysis.py</code></td>
      <td><b>Sabaki 術語對齊與動態規則判斷</b>：對齊 Sabaki 函式庫以獲取正確的定式與棋形術語，並補足 Sabaki 函式庫所缺乏的基礎動態圍棋規則與術語判斷（如提子 Take、叫吃 Atari、自殺 Suicide、填子 Fill、粘 Connect、天元 Tengen、星位 Hoshi 等）；提供 8 象限對稱變換基礎函數 <code>get_symmetric_coords</code>。</td>
    </tr>
    <tr>
      <td><code>src/token_eval.py</code></td>
      <td><b>Token 與成本分析工具</b>：統計多 Agent 協作過程中各階段的 Prompt Tokens、Completion Tokens 與推論開銷。</td>
    </tr>
    <tr>
      <td><code>src/outputs_processing.py</code></td>
      <td><b>輸出後處理工具</b>：清洗大語言模型輸出的思考標籤 (<code>&lt;think&gt;</code>) 與 XML 標籤，進行結構化 JSON 轉譯。</td>
    </tr>
  </tbody>
</table>

---

## 📊 4. 資料集與 KataGo 評估引擎介紹 (Datasets & Engine)

### 4.1 資料集介紹 (`data/`)
* **`data/source-qipuNsgf/`**：收錄標準 SGF 圍棋棋譜測試集，涵蓋星位、小目、三三等多元開局戰術佈局。
* **`data/source-dataset/`**：包含 `sabaki_names_list.py` 定式與圍棋戰術名稱詞庫，確保術語翻譯精準對齊。
* **`go_knowledge_base/`**：預留給 ChromaDB 定式知識庫 (`kogo_joseki_dict_v2`) 的資料庫掛載目錄。使用者可將 Kogo's Joseki Dictionary 向量數據放置於此。

### 4.2 KataGo 棋盤形勢評估引擎設定
在特徵提取模組 (`src/go_feature_extraction.py`) 中，系統對接 **KataGo C++ 分析引擎** 以獲取即時客觀的勝率與目數差：
* **KataGo 執行檔**：`katago` (分析模式 `analysis`)
* **神經網絡模型**：`kata_40b.bin.gz` (40-block 高階權重網絡)
* **規則設定 (Rules)**：`chinese` (中國圍棋規則，貼 7.5 目)
* **搜尋預算 (MCTS Visits)**：`maxVisits: 50`（設定為 50 次 MCTS 模擬搜尋，在極短時間內獲取穩定可靠的形勢勝率與目數差分析）。

---

## 🧪 5. 實驗數據檔案說明 (Evaluation Results)

本開源庫於 `eval-results/` 目錄下收錄最新完整的黃金實驗數據與評估結果：

<table width="100%">
  <thead>
    <tr>
      <th width="32%">檔案名稱</th>
      <th width="68%">檔案內容與實驗用途說明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>eval-results/sgf_rag_multi-agentic-llm.json</code></td>
      <td><b>主要原始實驗紀錄</b>：包含完整 <b>Trie 字典樹檢索 + RAG + Harness 狀態機 + NLI 驗證 + 雙階 Multi-Agentic LLM</b> 之全量測試導出資料（含 Harness 快取快照、對策歷史軌跡 <code>Ht</code>、英文日誌與繁中解說）。</td>
    </tr>
    <tr>
      <td><code>eval-results/output_processed.json</code></td>
      <td><b>結構化後處理數據</b>：經過腳本清洗標籤（移除內心思考 <code>&lt;think&gt;</code> 與多餘 XML 標籤）過後的精簡解說數據檔，利於快速閱讀與報告比較。</td>
    </tr>
    <tr>
      <td><code>eval-results/token_eval.json</code></td>
      <td><b>Token 消耗與成本統計</b>：記載全量測試棋譜在推論過程中各階段的 Prompt / Completion Tokens 數量、Context 視窗佔用比例與推論成本統計。</td>
    </tr>
    <tr>
      <td><code>eval-results/llm_as_a_judge.json</code></td>
      <td><b>LLM 裁判自動化評估數據</b>：使用大語言模型作為裁判（LLM-as-a-Judge）針對圍棋術語精準度、定式變化覆蓋率與繁中教學流暢度之多維度評分結果。</td>
    </tr>
  </tbody>
</table>

> 💡 **備註**：早期未採用 Trie 字典樹與精準 Move 檢索時的歷史實驗數據（如舊版 `rag_multi-agentic-llm_experiment.json`）已全數歸檔於備份庫中，本開源庫僅收錄最精準之最終黃金數據。

---

## 🚀 6. 快速開始與專案執行 (Quickstart)

### 6.1 安裝專案依賴套件
```bash
# 推薦使用 Conda 建立 Python 3.12 環境
conda create -n go_commentary python=3.12 -y
conda activate go_commentary

# 安裝所需依賴套件
pip install -r requirements.txt
```

### 6.2 執行主實驗流水線
```bash
# 切換至 src/ 目錄並執行主程式
cd src
python sgf_rag_multi-agentic-llm.py
```
