import os
import chess
import pygame
from pathlib import Path
from config import (BG, BOARD_PX, DARK, H, HILITE, LIGHT, PANEL_BG, PANEL_W, SQ, TEXT)

# Từ điển ánh xạ ký hiệu quân cờ sang tên file tương ứng
PIECE_FILES = {
    ('p', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_pawn_white.png",
    ('p', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_pawn_black.png",
    ('r', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_rook_white.png",
    ('r', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_rook_black.png",
    ('n', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_knight_white.png",
    ('n', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_knight_black.png",
    ('b', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_bishop_white.png",
    ('b', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_bishop_black.png",
    ('q', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_queen_white.png",
    ('q', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_queen_black.png",
    ('k', chess.WHITE): Path.cwd() / "assets" / "chess_pieces" / "chess_king_white.png",
    ('k', chess.BLACK): Path.cwd() / "assets" / "chess_pieces" / "chess_king_black.png",
}

_PIECE_IMAGES = {}

def get_texture(piece):
    load_piece_images()

    key = (piece.symbol().lower(), piece.color)
    return _PIECE_IMAGES.get(key)

def load_piece_images():
    global _PIECE_IMAGES
    if _PIECE_IMAGES:
        return
    for (symbol, color), filename in PIECE_FILES.items():
        if os.path.exists(filename):
            try:
                img = pygame.image.load(filename).convert_alpha()
                piece_size = SQ - 6
                _PIECE_IMAGES[(symbol, color)] = pygame.transform.smoothscale(img, (piece_size, piece_size))
            except Exception as e:
                print(f"Không thể tải ảnh {filename}: {e}")


def make_fonts():
    return {
        "big": pygame.font.Font(Path.cwd() / "assets" / "JetBrainsMonoNerdFontMono-Bold.ttf", 35),
        "title": pygame.font.Font(Path.cwd() / "assets" / "JetBrainsMonoNerdFontMono-Bold.ttf", 20),
        "main": pygame.font.Font(Path.cwd() / "assets" / "JetBrainsMonoNerdFont-Medium.ttf", 14),
        "small": pygame.font.Font(Path.cwd() / "assets" / "JetBrainsMonoNerdFont-Medium.ttf", 10),
        "piece": pygame.font.Font(Path.cwd() / "assets" / "JetBrainsMonoNerdFontMono-Bold.ttf", 26),
    }


def sq_to_px(sq, flipped=False):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if flipped:
        f, r = 7 - f, 7 - r
    return f * SQ, (7 - r) * SQ


def px_to_sq(pos, flipped=False):
    x, y = pos
    if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
        return None
    f, r = x // SQ, 7 - y // SQ
    if flipped:
        f, r = 7 - f, 7 - r
    return chess.square(f, r)


def draw_board(screen, fonts, board, last_move=None, flipped=False, selected=None, targets=()):
    load_piece_images()

    for r in range(8):
        for f in range(8):
            pygame.draw.rect(screen, LIGHT if (r + f) % 2 else DARK, (f * SQ, (7 - r) * SQ, SQ, SQ))

    def tint(sq, color):
        s = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
        s.fill(color)
        screen.blit(s, sq_to_px(sq, flipped))

    if last_move:
        tint(last_move.from_square, (*HILITE, 140))
        tint(last_move.to_square, (*HILITE, 140))
    if board.is_check():
        tint(board.king(board.turn), (220, 50, 50, 170))
    if selected is not None:
        tint(selected, (80, 160, 255, 150))

    for sq, piece in board.piece_map().items():
        px, py = sq_to_px(sq, flipped)
        key = (piece.symbol().lower(), piece.color)
        
        if key in _PIECE_IMAGES:
            img = _PIECE_IMAGES[key]
            img_rect = img.get_rect(center=(px + SQ // 2, py + SQ // 2))
            screen.blit(img, img_rect)
        else:
            cx, cy = px + SQ // 2, py + SQ // 2
            white = piece.color == chess.WHITE
            pygame.draw.circle(screen, (250, 250, 250) if white else (25, 25, 25), (cx, cy), SQ // 2 - 8)
            pygame.draw.circle(screen, (20, 20, 20) if white else (230, 230, 230), (cx, cy), SQ // 2 - 8, 2)
            lbl = fonts["piece"].render(piece.symbol().upper(), True,
                                        (20, 20, 20) if white else (240, 240, 240))
            screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

    for sq in targets: 
        px, py = sq_to_px(sq, flipped)
        c = (px + SQ // 2, py + SQ // 2)
        if board.piece_at(sq) or sq == board.ep_square:
            pygame.draw.circle(screen, (40, 120, 255), c, SQ // 2 - 4, 4)
        else:
            pygame.draw.circle(screen, (40, 120, 255), c, 9)

    for i in range(8):
        file_ch = "abcdefgh"[7 - i if flipped else i]
        rank_ch = str(i + 1 if flipped else 8 - i)
        screen.blit(fonts["small"].render(file_ch, True, (60, 60, 60)), (i * SQ + SQ - 12, BOARD_PX - 15))
        screen.blit(fonts["small"].render(rank_ch, True, (60, 60, 60)), (3, i * SQ + 2))


def _panel(screen, fonts):
    x0 = BOARD_PX
    pygame.draw.rect(screen, PANEL_BG, (x0, 0, PANEL_W, H))
    state = {"y": 14}

    def line(txt, font="main", color=TEXT, dy=24):
        screen.blit(fonts[font].render(txt, True, color), (x0 + 16, state["y"]))
        state["y"] += dy

    return x0, state, line


def draw_play_panel(screen, fonts, match, hints):
    x0, st, line = _panel(screen, fonts)
    line("Cờ vua", "title", dy=36)
    line(f"Trắng: {match.white.name}", color=(240, 240, 240))
    line(f"Đen: {match.black.name}", color=(170, 190, 255), dy=32)
    turn = "Trắng" if match.board.turn == chess.WHITE else "Đen"
    if match.result:
        line("Kết thúc ván đấu", color=(255, 200, 90))
    else:
        line(f"Lượt: {turn}" + ("   - CHIẾU TƯỚNG!" if match.board.is_check() else ""), color=(255, 200, 90))

    st["y"] += 10
    line("Nước đi", color=(160, 200, 255), dy=26)
    rows = [(i // 2 + 1, match.san[i], match.san[i + 1] if i + 1 < len(match.san) else "")
            for i in range(0, len(match.san), 2)][-14:]
    
    for n, w, b in rows:
        line(f"{n:>3}.   {w:<8}{b}", "main", dy=22)
    y = H - 18 * len(hints) - 12
    for t in hints:
        screen.blit(fonts["small"].render(t, True, (170, 170, 180)), (x0 + 16, y))
        y += 18


def draw_train_panel(screen, fonts, game, paused, fast, delay):
    x0, st, line = _panel(screen, fonts)
    line("Tự học cờ vua (RL)", "title", dy=36)
    turn = "Trắng" if game.board.turn == chess.WHITE else "Đen"
    mode = "ĐÃ TẠM DỪNG" if paused else ("HUẤN LUYỆN NHANH" if fast else f"XEM ({delay} ms/nước)")
    line(f"Chế độ: {mode}", color=(255, 200, 90))
    line(f"Lượt: {turn}   Nửa nước: {len(game.board.move_stack)}")
    line(f"Số ván đã đấu: {game.games}")
    line(f"Kết quả gần nhất: {game.last_result}", dy=30)

    n = max(1, len(game.recent))
    line("Tổng kết", color=(160, 200, 255))
    line(f"   Trắng {game.wins['White']}   Đen {game.wins['Black']}   Hòa {game.wins['Draw']}", dy=28)
    line(f"{len(game.recent)} ván gần đây", color=(160, 200, 255))
    line(f"   Trắng {game.recent.count('White')/n:5.0%}   Đen {game.recent.count('Black')/n:5.0%}   "
         f"Hòa {game.recent.count('Draw')/n:5.0%}")
    avg = sum(game.recent_len) / n if game.recent_len else 0
    line(f"   Độ dài TB: {avg:.0f} nước")
    line(f"   Mất mát (Loss) Trắng {game.white.last_loss:+.3f}   Đen {game.black.last_loss:+.3f}", dy=30)

    gx, gy, gw, gh = x0 + 16, st["y"] + 6, PANEL_W - 32, 150
    pygame.draw.rect(screen, (22, 22, 26), (gx, gy, gw, gh))
    pygame.draw.rect(screen, (90, 90, 100), (gx, gy, gw, gh), 1)
    hist = game.history[-gw:]
    if len(hist) > 1:
        for k, col in ((0, (240, 240, 240)), (1, (120, 170, 255)), (2, (255, 170, 80))):
            pts = [(gx + i * gw / (len(hist) - 1), gy + gh - 4 - v[k] * (gh - 8))
                   for i, v in enumerate(hist)]
            pygame.draw.lines(screen, col, False, pts, 2)
    screen.blit(fonts["small"].render("tỷ lệ thắng trượt:   trắng / đen / hòa", True, (170, 170, 180)),
                (gx, gy + gh + 4))
    st["y"] = gy + gh + 30
    for t in ("SPACE tạm dừng   F chế độ nhanh", "UP/DOWN tốc độ   S lưu   L tải",
              "R đặt lại thống kê   ESC menu"):
        line(t, "small", (170, 170, 180), 18)