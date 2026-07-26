# go_tactic_analysis.py
import os
import json

def get_symmetric_coords(x, y, sym_index):
    """對 (x, y) 進行 8 種對稱變換 (0-7)，涵蓋 4 個角落與旋轉鏡像"""
    if sym_index == 0: return x, y
    elif sym_index == 1: return 18-x, y
    elif sym_index == 2: return x, 18-y
    elif sym_index == 3: return 18-x, 18-y
    elif sym_index == 4: return y, x
    elif sym_index == 5: return 18-y, x
    elif sym_index == 6: return y, 18-x
    else: return 18-y, 18-x

def get_board_sign(board, px, py, player_color):
    """獲取棋盤座標 (px, py) 的狀態相對於 player_color 的符號 (1: 我方, -1: 敵方, 0: 空格)"""
    if px < 0 or px >= 19 or py < 0 or py >= 19:
        return None
    stone = board[py][px]
    if stone is None:
        return 0
    return 1 if stone == player_color else -1

class SabakiPatternMatcher:
    def __init__(self, json_path):
        self.patterns = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.patterns = json.load(f)
            except Exception as e:
                print(f"Error loading Sabaki library: {e}")
        else:
            print(f"Warning: Sabaki library path not found: {json_path}")

    def match_move(self, board, last_move_x, last_move_y, color):
        """
        利用 8 象限對稱變換，將最後一手與庫中的定式/佈局/棋形進行剛性模板比對
        """
        if not self.patterns:
            return None

        for pattern in self.patterns:
            name = pattern.get("name")
            anchors = pattern.get("anchors", [])
            vertices = pattern.get("vertices", [])
            is_corner = pattern.get("type") == "corner"

            for anchor in anchors:
                ax, ay = anchor[0]
                anchor_sign = anchor[1]
                
                if anchor_sign != 1:
                    continue

                for sym_idx in range(8):
                    tax, tay = get_symmetric_coords(ax, ay, sym_idx)

                    if is_corner:
                        if tax != last_move_x or tay != last_move_y:
                            continue
                        dx, dy = 0, 0
                    else:
                        dx = last_move_x - tax
                        dy = last_move_y - tay

                    matched = True
                    for vertex in vertices:
                        vx, vy = vertex[0]
                        expected_sign = vertex[1]

                        tvx, tvy = get_symmetric_coords(vx, vy, sym_idx)
                        bx = tvx + dx
                        by = tvy + dy

                        if bx < 0 or bx >= 19 or by < 0 or by >= 19:
                            matched = False
                            break

                        actual_sign = get_board_sign(board, bx, by, color)
                        if actual_sign != expected_sign:
                            matched = False
                            break

                    if matched:
                        return name
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SABAKI_JSON_PATH = os.path.join(BASE_DIR, "source-dataset", "sabaki_library.json")
if not os.path.exists(SABAKI_JSON_PATH):
    SABAKI_JSON_PATH = os.path.join(BASE_DIR, "source-doc", "sabaki_library.json")

PATTERN_MATCHER = SabakiPatternMatcher(SABAKI_JSON_PATH)

# 保留對照表結構供後續實驗擴充使用，當前分析核心不再調用
TRANSLATION_MAP = {
    "Low Chinese Opening": "低中國流",
    "High Chinese Opening": "高中國流",
    "Orthodox Opening": "星無憂角",
    "Enclosure Opening": "守角",
    "Kobayashi Opening": "小林流",
    "Small Chinese Opening": "迷你中國流",
    "Micro Chinese Opening": "微型中國流",
    "Sanrensei Opening": "三連星",
    "Nirensei Opening": "二連星",
    "Shūsaku Opening": "秀策流",
    "Low Approach": "低掛",
    "High Approach": "高掛",
    "Low Enclosure": "低位守角",
    "High Enclosure": "高位守角",
    "Mouth Shape": "口形",
    "Table Shape": "桌形",
    "Tippy Table": "歪桌形",
    "Bamboo Joint": "雙",
    "Trapezium": "梯形",
    "Diamond": "菱形",
    "Tiger’s Mouth": "虎口",
    "Empty Triangle": "愚形三角",
    "Turn": "拐",
    "Stretch": "長",
    "Diagonal": "尖",
    "Wedge": "挖",
    "Hane": "扳",
    "Cut": "斷",
    "Square": "方形",
    "Throwing Star": "飛鏢形",
    "Parallelogram": "平行四邊形",
    "Dog’s Head": "狗頭形",
    "Horse’s Head": "馬頭形",
    "Attachment": "碰",
    "One-Point Jump": "一間跳",
    "Big Bulge": "小飛方陣少一子",
    "Small Knight": "小飛",
    "Two-Point Jump": "二間跳",
    "Large Knight": "大飛",
    "3-3 Point Invasion": "點三三",
    "Shoulder Hit": "尖衝",
    "Diagonal Jump": "象步",
    "3-3 Point": "三三",
    "3-4 Point": "小目",
    "4-4 Point": "星位",
    "3-5 Point": "目外",
    "4-5 Point": "高目",
    "6-3 Point": "大目外",
    "6-4 Point": "超高目",
    "5-5 Point": "五五",
    "Regular Move": "常規落子",
    "Pass": "虛手",
    "Take": "提子",
    "Atari": "叫吃",
    "Suicide": "自殺",
    "Fill": "填子",
    "Connect": "粘",
    "Tengen": "天元",
    "Hoshi": "星位"
}

class GoTacticAnalyzer:
    """整合 Sabaki Library 棋形匹配與動態規則判定的圍棋定式與戰術分析器"""
    
    @staticmethod
    def analyze_last_move(last_move, history_moves, pure_engine, total_game_steps, focus_group=None):
        """
        分析最後一手落子屬於何種 Joseki / Fuseki / Shape
        """
        if not last_move:
            return "Pass"

        x, y = last_move["x"], last_move["y"]
        color = last_move["color"]
        opponent_color = 'W' if color == 'B' else 'B'
        
        # 1. 判定 Pass (防呆)
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            return "Pass"

        # 2. 判定 Take (提子)
        if getattr(pure_engine, "last_captured", 0) > 0:
            return "Take"

        # 3. 判定 Atari (叫吃)
        # 檢查四周敵方棋子，若在此落子後，有敵方棋塊的氣數變為剛好 1 氣
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if pure_engine.board[ny][nx] == opponent_color:
                    _, liberties = pure_engine.get_group_and_liberties(nx, ny)
                    if len(liberties) == 1:
                        return "Atari"

        # 4. 判定 Suicide (自殺)
        # 檢查己方棋子在此落子後的氣數是否為 0
        _, my_liberties = pure_engine.get_group_and_liberties(x, y)
        if len(my_liberties) == 0:
            return "Suicide"

        # 5. 判定 Fill 與 Connect
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                neighbors.append((nx, ny))
                
        friendlies = [n for n in neighbors if pure_engine.board[n[1]][n[0]] == color]
        
        if len(friendlies) == len(neighbors):
            return "Fill"
        elif len(friendlies) >= 2:
            return "Connect"

        # 6. 100% 依賴 Sabaki JSON 庫進行 8 象限圖案匹配
        sabaki_tactic = PATTERN_MATCHER.match_move(pure_engine.board, x, y, color)
        
        if sabaki_tactic:
            # 排除非開局時的單子（如中盤戰鬥隨手點在三三或星位上）的干擾
            if sabaki_tactic in ["3-3 Point", "3-4 Point", "4-4 Point", "3-5 Point", "4-5 Point", "5-5 Point", "6-3 Point", "6-4 Point"] and total_game_steps > 60:
                return "Regular Move"
            return sabaki_tactic

        # 7. 判定 Tengen 與 Hoshi
        hoshis = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 15), (15, 3), (15, 9), (15, 15)}
        if x == 9 and y == 9:
            return "Tengen"
        elif (x, y) in hoshis:
            return "Hoshi"

        return "Regular Move"