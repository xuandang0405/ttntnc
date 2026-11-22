import os

import chess
import pygame

from concurrent_mode.concurrent_renderer import draw_games_grid
from config import BG, BOARD_PX, H, MAX_AI_PLIES, PANEL_BG, TEXT, W
from concurrent_mode.concurrent_config import NUM_GAMES
from match import Match
from players import make_player
from plot_logging.game_record import GameResultRecorder
from scene import Scene

GAMES_PER_UPDATE = NUM_GAMES

class ConcurrentAITestScene(Scene):
    def __init__(self, fonts, white_spec, black_spec, greedy=False, lookahead=False):
        super().__init__(fonts)
        self.white = make_player(white_spec, chess.WHITE, greedy, lookahead)
        self.black = make_player(black_spec, chess.BLACK, greedy, lookahead)
        self.total_games = NUM_GAMES
        self.matches = [Match(self.white, self.black, MAX_AI_PLIES) for _ in range(self.total_games)]
        self.recorder = GameResultRecorder("ai_vs_ai", self.white.name, self.black.name)
        self.cursor = 0
        self.paused = False
        self.status = ""

    @property
    def completed_games(self):
        return len(self.recorder.games)

    def handle_event(self, e):
        if e.type != pygame.KEYDOWN:
            return
        if e.key == pygame.K_ESCAPE:
            from menu_scene import MenuScene
            self.next = MenuScene(self.fonts)
        elif e.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif e.key == pygame.K_s:
            self.save_results()

    def update(self):
        if self.paused or self.completed_games >= self.total_games:
            return

        completed = False
        updated = 0
        while updated < GAMES_PER_UPDATE and self.completed_games < self.total_games:
            by_player = {}
            scanned = 0
            scheduled = 0
            update_limit = min(GAMES_PER_UPDATE - updated, self.total_games)
            while scanned < self.total_games and scheduled < update_limit:
                i = self.cursor
                self.cursor = (self.cursor + 1) % self.total_games
                scanned += 1
                match = self.matches[i]
                if match.result:
                    continue
                player = match.current_player()
                by_player.setdefault(player, []).append((match, match.board))
                scheduled += 1

            if not scheduled:
                break

            for player, games in by_player.items():
                boards = [board for _, board in games]
                choose_moves = getattr(player, "choose_moves", None)
                moves = choose_moves(boards) if choose_moves else [
                    player.choose_move(board) for board in boards
                ]
                if len(moves) != len(games):
                    raise RuntimeError(
                        f"{player.name} returned {len(moves)} moves for {len(games)} games")
                for (match, _), move in zip(games, moves):
                    match.push(move)
                    updated += 1
                    if match.result:
                        self.recorder.record_result(match.result)
                        completed = True
            if scheduled < update_limit and self.total_games > 1:
                break
        if completed:
            self.save_results()

    def save_results(self):
        try:
            self.recorder.save()
        except OSError as ex:
            self.status = f"Lỗi lưu kết quả: {ex}"
            print(f"Không thể lưu kết quả ván đấu: {ex}")
            return False
        self.status = f"Đã lưu: {os.path.basename(self.recorder.path)}"
        return True

    def draw(self, screen):
        screen.fill(BG)
        draw_games_grid(screen, self.fonts, self.matches)
        panel = pygame.Rect(BOARD_PX, 0, W - BOARD_PX, H)
        pygame.draw.rect(screen, PANEL_BG, panel)
        x, y = BOARD_PX + 18, 30

        def line(text, font="main", color=TEXT, spacing=30):
            nonlocal y
            rendered = self.fonts[font].render(text, True, color)
            screen.blit(rendered, (x, y))
            y += spacing

        line("Kiểm thử AI đấu AI", "title", spacing=42)
        line(f"Trắng: {self.white.name}", "small", spacing=24)
        line(f"Đen: {self.black.name}", "small", spacing=38)
        line(f"Hoàn thành: {self.completed_games}/{self.total_games}")
        line(f"Đang đấu: {self.total_games - self.completed_games}")
        line(f"Trắng thắng: {self.recorder.stats['white']['wins']}")
        line(f"Đen thắng: {self.recorder.stats['black']['wins']}")
        line(f"Hòa: {self.recorder.stats['white']['draws']}")
        line(f"Elo Trắng: {self.recorder.ratings['white']:.0f}")
        line(f"Elo Đen: {self.recorder.ratings['black']:.0f}")
        if self.paused:
            line("ĐÃ TẠM DỪNG", color=(255, 200, 90))
        if self.status:
            line(self.status[:40], "small", spacing=24)
        line("SPACE: tạm dừng", "small", (170, 170, 180), 18)
        line("S: lưu kết quả", "small", (170, 170, 180), 18)
        line("ESC: về menu", "small", (170, 170, 180), 18)

    def on_exit(self):
        if self.recorder.games:
            self.save_results()
