# go_feature_extraction.py
import os
from sgfmill import sgf

# 匯入定式/棋形分析器
from go_tactic_analysis import GoTacticAnalyzer

class PureGoBoardEngine:
    """純 Python 實作的輕量級圍棋規則引擎，用於實時維護盤面狀態（支援提子與氣數計算）"""
    def __init__(self):
        self.board = [[None for _ in range(19)] for _ in range(19)]
        self.last_captured = 0  # 記錄最後一手棋提吃了幾顆子

    def play_move(self, x, y, color):
        self.board[y][x] = color
        self.last_captured = 0
        opponent_color = 'W' if color == 'B' else 'B'
        
        # 檢查鄰近敵方棋子是否被提吃
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if self.board[ny][nx] == opponent_color:
                    group, liberties = self.get_group_and_liberties(nx, ny)
                    if len(liberties) == 0:
                        for gx, gy in group:
                            self.board[gy][gx] = None
                        self.last_captured += len(group)

    def get_group_and_liberties(self, start_x, start_y):
        """計算 (start_x, start_y) 所在棋塊的座標群與其所有氣（Liberties）的集合"""
        color = self.board[start_y][start_x]
        if color is None:
            return set(), set()
        
        queue = [(start_x, start_y)]
        group = {(start_x, start_y)}
        liberties = set()
        
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 19 and 0 <= ny < 19:
                    if self.board[ny][nx] is None:
                        liberties.add((nx, ny))
                    elif self.board[ny][nx] == color and (nx, ny) not in group:
                        group.add((nx, ny))
                        queue.append((nx, ny))
        return group, liberties

class GoFeatureExtractor:
    def __init__(self, katago_exec="/home/ailab/Desktop/KataGo/cpp/katago", 
                 katago_model="/home/ailab/Desktop/kata_40b.bin.gz", 
                 katago_config="/home/ailab/Desktop/KataGo/cpp/configs/gtp_example.cfg"):
        self.katago_exec = katago_exec
        self.katago_model = katago_model
        self.katago_config = katago_config
        # 預設對齊專案目錄：MY_MASTER_THESIS/source-qipuNsgf/sgf_prompt
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "source-qipuNsgf", "sgf_prompt"))

    def to_weiqi_coord_string(self, x: int, y: int) -> str:
        """將數字座標轉回標準圍棋大寫字母座標 (跳過 I)"""
        COLS = "ABCDEFGHJKLMNOPQRST"
        return f"{COLS[x]}{y + 1}"

    def extract_features_to_dict(self, sgf_string) -> dict:
        """
        解析 SGF 並提取特徵，直接返回包含三大 Pillar 的 Python 字典。
        """
        try:
            sgf_game = sgf.Sgf_game.from_string(sgf_string)
        except Exception:
            return {"error": "SGF 解析失敗"}
            
        cleaned_moves = []
        for node in sgf_game.get_main_sequence():
            color, move = node.get_move()
            if color is None or move is None: continue
            row, col = move
            cleaned_moves.append({
                "color": color.upper(),
                "x": col,
                "y": row,
                "gtp": f"{'abcdefghjklmnopqrst'[col]}{row + 1}"
            })
            
        if not cleaned_moves: 
            return {"error": "棋譜內無有效落子手順"}
            
        total_steps = len(cleaned_moves)

        track_a_tokens = []
        track_b_tokens = []
        incremental_engine = PureGoBoardEngine()
        tactic_human_tokens = []

        # 平行軌跡生成（不進行任何鏡像翻轉）
        for i, current_move in enumerate(cleaned_moves):
            cx, cy = current_move["x"], current_move["y"]
            color_code = current_move["color"]
            
            # Pillar 1：標準化真實座標（用於 Prompt 展示）
            weiqi_b_coord = self.to_weiqi_coord_string(cx, cy)
            track_b_tokens.append(f"{color_code}[{weiqi_b_coord}]")

            # Pillar 2：標準化與 8-fold 還原座標（直通不翻轉，用於庫匹配）
            ax, ay = cx, cy
            weiqi_a_coord = self.to_weiqi_coord_string(ax, ay)
            track_a_tokens.append(f"{color_code}[{weiqi_a_coord}]")

            # 呼叫定式棋形分析器 (此時 go_tactic_analysis 已改回直接吐英文術語)
            incremental_engine.play_move(cx, cy, color_code)
            current_history_slice = cleaned_moves[max(0, i-10):i+1]
            tactic_name = GoTacticAnalyzer.analyze_last_move(
                last_move=current_move, history_moves=current_history_slice,
                pure_engine=incremental_engine, total_game_steps=i + 1
            )
            color_en = "Black" if color_code == "B" else "White"
            tactic_human_tokens.append(f"{color_en}[{weiqi_b_coord}]_{tactic_name}")

        # 啟動 KataGo 算力核心
        import subprocess
        import json
        winrate = 0.5
        score_lead = 0.0
        
        katago_moves = [[m["color"], m["gtp"]] for m in cleaned_moves]
        query = {"id": "exp", "moves": katago_moves, "rules": "chinese", "boardXSize": 19, "boardYSize": 19, "maxVisits": 50}
        cmd = [self.katago_exec, "analysis", "-model", self.katago_model, "-config", self.katago_config]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_output, _ = proc.communicate(input=json.dumps(query) + "\n", timeout=20)
            if stdout_output:
                katago_data = json.loads(stdout_output.strip())
                if katago_data.get("moveInfos"):
                    winrate = katago_data["moveInfos"][0].get("winrate", 0.5)
                    score_lead = katago_data["moveInfos"][0].get("scoreLead", 0.0)
        except Exception: 
            pass

        # 精準返回三大 Pillar 區塊 (全量移除非必要的表情符號)
        return {
            "track_b_string": "_".join(track_b_tokens),
            "track_a_string": "_".join(track_a_tokens),
            "kata_and_tactic_payload": {
                "total_steps": total_steps,
                "game_phase": "布局",
                "winrate_perspective": f"{winrate * 100:.2f}%",
                "score_lead": f"{score_lead:.2f}",
                "joseki_sequence_human": ", ".join(tactic_human_tokens)
            }
        }

    def extract_features_from_file(self, file_name_or_path) -> dict:
        """讀取檔案並直接回傳三大 Pillar 字典"""
        full_path = file_name_or_path if file_name_or_path.startswith("/") or file_name_or_path.startswith(".") else os.path.join(self.base_dir, file_name_or_path)
        if not os.path.exists(full_path): 
            return {"error": f"找不到指定棋譜檔案: {full_path}"}
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return self.extract_features_to_dict(f.read().strip())