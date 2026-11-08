import math

import chess
import pygame

from config import BOARD_PX, DARK, HILITE, LIGHT, PANEL_W
from renderer import _panel, get_texture

_scaled = {} 
_fonts = {}   


def _texture(piece, size):
    tex = get_texture(piece) 
    if tex is None:
        return None
    key = (piece.piece_type, piece.color, size)
    if key not in _scaled:
        _scaled[key] = pygame.transform.smoothscale(tex, (size, size))
    return _scaled[key]


def draw_small_board(screen, board, last_move, origin, size, fonts=None, coords=False):
    s = size // 8
    ox, oy = origin

    def pos(sq):
        return (ox + chess.square_file(sq) * s, oy + (7 - chess.square_rank(sq)) * s)

    for r in range(8):
        for f in range(8):
            pygame.draw.rect(screen, LIGHT if (r + f) % 2 else DARK, (ox + f * s, oy + (7 - r) * s, s, s))

    def tint(sq, color):
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        surf.fill(color)
        screen.blit(surf, pos(sq))

    if last_move:
        tint(last_move.from_square, (*HILITE, 140))
        tint(last_move.to_square, (*HILITE, 140))
    if board.is_check():
        tint(board.king(board.turn), (220, 50, 50, 170))

    for sq, piece in board.piece_map().items():
        px, py = pos(sq)
        tex = _texture(piece, s)
        if tex:
            screen.blit(tex, (px, py))
            continue
        cx, cy = px + s // 2, py + s // 2        # dự phòng: hình tròn + chữ
        white = piece.color == chess.WHITE
        rad = max(2, s // 2 - s // 9)
        pygame.draw.circle(screen, (250, 250, 250) if white else (25, 25, 25), (cx, cy), rad)
        pygame.draw.circle(screen, (20, 20, 20) if white else (230, 230, 230), (cx, cy), rad, 2)
        if s not in _fonts:
            _fonts[s] = pygame.font.SysFont("arial", max(8, int(s * 0.47)), bold=True)
        lbl = _fonts[s].render(piece.symbol().upper(), True, (20, 20, 20) if white else (240, 240, 240))
        screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

    if coords and fonts:
        for i in range(8):
            screen.blit(fonts["small"].render("abcdefgh"[i], True, (60, 60, 60)), (ox + i * s + s - 12, oy + size - 15))
            screen.blit(fonts["small"].render(str(8 - i), True, (60, 60, 60)), (ox + 3, oy + i * s + 2))


def draw_games_grid(screen, fonts, slots):
    k = len(slots)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    cell = BOARD_PX // max(cols, rows)
    size = max(8, (cell - 6) // 8 * 8)
    pad = (cell - size) // 2
    x_off = (BOARD_PX - cols * cell) // 2
    y_off = (BOARD_PX - rows * cell) // 2
    for i, slot in enumerate(slots):
        r, c = divmod(i, cols)
        origin = (x_off + c * cell + pad, y_off + r * cell + pad)
        draw_small_board(screen, slot.board, slot.last_move, origin, size, fonts, coords=size >= 300)
        pygame.draw.rect(screen, (15, 15, 18), (*origin, size, size), 2)


def draw_concurrent_panel(screen, fonts, game, paused, fast, delay):
    x0, st, line = _panel(screen, fonts)
    line("Tự học cờ vua (RL)", "title", dy=36)
    mode = "ĐÃ TẠM DỪNG" if paused else ("HUẤN LUYỆN NHANH" if fast else f"XEM ({delay} ms/bước)")
    line(f"Chế độ: {mode}", color=(255, 200, 90))
    line(f"Số ván song song: {game.k}   Thiết bị: {game.device_name}")
    avg_live = sum(len(s.board.move_stack) for s in game.slots) / max(1, game.k)
    line(f"Nửa nước TB đang chơi: {avg_live:.0f}")
    line(f"Tổng số ván đã đấu: {game.games}")
    line(f"Kết quả gần nhất: {game.last_result}", dy=30)

    n = max(1, len(game.recent))
    line("Tổng kết", color=(160, 200, 255))
    line(f"  Trắng {game.wins['White']}   Đen {game.wins['Black']}   Hòa {game.wins['Draw']}", dy=28)
    line(f"{len(game.recent)} ván gần đây", color=(160, 200, 255))
    line(f"  Trắng {game.recent.count('White')/n:5.0%}  Đen {game.recent.count('Black')/n:5.0%}  "
         f"Hòa {game.recent.count('Draw')/n:5.0%}")
    avg = sum(game.recent_len) / n if game.recent_len else 0
    line(f"  Độ dài TB: {avg:.0f} nửa nước")
    line(f"  Mất mát (Loss) Trắng {game.white.last_loss:+.3f}  Đen {game.black.last_loss:+.3f}", dy=30)

    gx, gy, gw, gh = x0 + 16, st["y"] + 6, PANEL_W - 32, 130
    pygame.draw.rect(screen, (22, 22, 26), (gx, gy, gw, gh))
    pygame.draw.rect(screen, (90, 90, 100), (gx, gy, gw, gh), 1)
    hist = game.history[-gw:]
    if len(hist) > 1:
        for k, col in ((0, (240, 240, 240)), (1, (120, 170, 255)), (2, (255, 170, 80))):
            pts = [(gx + i * gw / (len(hist) - 1), gy + gh - 4 - v[k] * (gh - 8))
                   for i, v in enumerate(hist)]
            pygame.draw.lines(screen, col, False, pts, 2)
    screen.blit(fonts["small"].render("tỷ lệ thắng trượt:  trắng / đen / hòa", True, (170, 170, 180)),
                (gx, gy + gh + 4))
    st["y"] = gy + gh + 30
    for t in ("SPACE tạm dừng   F chế độ nhanh", "UP/DOWN tốc độ   S lưu   L tải",
              "R đặt lại thống kê   ESC menu"):
        line(t, "small", (170, 170, 180), 18)