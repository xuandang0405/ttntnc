"""Playing a game: human vs human, human vs model, or model vs model."""
from concurrent.futures import ThreadPoolExecutor
import os

import chess
import pygame

from config import AI_MOVE_DELAY, BG, MAX_AI_PLIES, BOARD_PX
from match import Match
from players import ModelPlayer, make_player
from normal_mode.renderer import draw_board, draw_play_panel, px_to_sq
from plot_logging.game_record import GameResultRecorder
from scene import Scene
from ui import GameModePopup, GameOverPopup, Button

GREEN, RED, GREY = (90, 200, 120), (230, 90, 90), (190, 190, 200)
PROMOTION_CHOICES = [(chess.QUEEN, "Queen"), (chess.ROOK, "Rook"), (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight")]

class PlayScene(Scene):
    def __init__(self, fonts, white_spec, black_spec, greedy, lookahead=False):
        super().__init__(fonts)
        self.specs, self.greedy, self.lookahead = (white_spec, black_spec), greedy, lookahead
        self.ai_delay = AI_MOVE_DELAY
        self.recording = False
        self.result_recorder = None
        self.record_status = ""
        self.recording_popup = None
        self.menu = None
        self.move_executor = None
        self.move_future = None
        self.ai_error = ""
        self.new_match()

    def new_match(self):
        white = make_player(self.specs[0], chess.WHITE, self.greedy, self.lookahead)
        black = make_player(self.specs[1], chess.BLACK, self.greedy, self.lookahead)
        humans = [p.is_human for p in (white, black)]
        self.match = Match(white, black, max_plies=None if any(humans) else MAX_AI_PLIES)
        self.recording_allowed = sum(humans) < 2
        self.flipped = humans == [False, True]       # human plays Black -> Black at the bottom
        self.selected, self.targets = None, ()
        self.promotion = None       
        self.popup = None
        self.recorded_current_game = False
        self.last_ai_move = pygame.time.get_ticks()

    def handle_promotion(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:       # cancel: move not made
            self.promotion = None
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for piece_type, button in self.promo_buttons:
                if button.hit(e.pos):
                    move = self.promotion[piece_type]
                    self.promotion, self.selected, self.targets = None, None, ()
                    self.play(move)
                    return

    # input
    def handle_event(self, e):
        if self.recording_popup:
            action = self.recording_popup.handle_event(e)
            if action == "single":
                self.recording_popup = None
                self.handle_recording_key(pygame.K_r)
            elif action == "multiple":
                self.recording_popup = None
                from concurrent_mode.concurrent_test_scene import ConcurrentAITestScene
                self.next = ConcurrentAITestScene(
                    self.fonts, self.specs[0], self.specs[1], self.greedy, self.lookahead)
                self.next.menu = self.menu
            elif action == "cancel":
                self.recording_popup = None
            return
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_r, pygame.K_x, pygame.K_s):
            if e.key == pygame.K_r and self.recording_allowed and not self.match.humans():
                self.recording_popup = GameModePopup()
            else:
                self.handle_recording_key(e.key)
            return
        if self.popup:
            action = self.popup.handle_event(e)
            if action == "rematch":
                self.new_match()
            elif action == "menu":
                self.go_menu()
            return
        if self.promotion:                   # choosing the promotion piece blocks everything else
            self.handle_promotion(e)
            return
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.go_menu()
            elif e.key == pygame.K_UP:
                self.ai_delay = max(0, self.ai_delay - 100)
            elif e.key == pygame.K_DOWN:
                self.ai_delay = min(2000, self.ai_delay + 100)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.match.current_player().is_human:
                self.click(px_to_sq(e.pos, self.flipped))

    def click(self, sq):
        board = self.match.board
        if sq is None:
            return
        if self.selected is not None:
            options = [mv for mv in board.legal_moves if mv.from_square == self.selected and mv.to_square == sq]
            if len(options) == 1:
                self.selected, self.targets = None, ()
                self.play(options[0])
                return
            
            if options:
                self.promotion = {mv.promotion: mv for mv in options}
                self.promo_buttons = [(pt, Button((48 + i * 128, BOARD_PX // 2 - 28, 120, 56), name))
                                      for i, (pt, name) in enumerate(PROMOTION_CHOICES)]
                return
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            self.selected = sq
            self.targets = [m.to_square for m in board.legal_moves if m.from_square == sq]
        else:
            self.selected, self.targets = None, ()

    def play(self, move):
        self.match.push(move)
        self.last_ai_move = pygame.time.get_ticks()
        if self.match.result:
            if self.recording and self.result_recorder:
                self.record_current_result()
                self.start_next_recorded_game()
            else:
                self.popup = self.make_popup()

    def start_next_recorded_game(self):
        white = make_player(self.specs[0], chess.WHITE, self.greedy, self.lookahead)
        black = make_player(self.specs[1], chess.BLACK, self.greedy, self.lookahead)
        humans = [p.is_human for p in (white, black)]
        self.match = Match(white, black, max_plies=None if any(humans) else MAX_AI_PLIES)
        self.selected, self.targets = None, ()
        self.promotion = None
        self.popup = None
        self.recording_popup = None
        self.recording_popup = None
        self.recorded_current_game = False
        self.last_ai_move = pygame.time.get_ticks()

    def handle_recording_key(self, key):
        if not self.recording_allowed:
            return
        if key == pygame.K_r:
            if self.result_recorder is None:
                humans = self.match.humans()
                mode = "human_vs_ai" if humans else "ai_vs_ai"
                self.result_recorder = GameResultRecorder(
                    mode, self.match.white.name, self.match.black.name)
            self.recording = True
            self.record_status = "Đang ghi kết quả (R ghi, X dừng, S lưu)"
            if self.match.result:
                self.record_current_result()
                self.start_next_recorded_game()
        elif key == pygame.K_x:
            if self.recording:
                self.recording = False
                if self.save_results():
                    self.record_status = "Đã dừng ghi; kết quả đã lưu vẫn được giữ lại"
        elif key == pygame.K_s:
            if self.result_recorder:
                self.save_results()
            else:
                self.record_status = "Chưa có dữ liệu ghi để lưu"

    def record_current_result(self):
        if self.recorded_current_game or not self.result_recorder or not self.match.result:
            return
        self.result_recorder.record_result(self.match.result)
        self.recorded_current_game = True
        self.save_results()

    def save_results(self):
        try:
            self.result_recorder.save()
        except OSError as ex:
            self.record_status = f"Lỗi lưu kết quả: {ex}"
            print(f"Không thể lưu kết quả ván đấu: {ex}")
            return False
        else:
            self.record_status = f"Đã lưu: {os.path.basename(self.result_recorder.path)}"
            return True

    def go_menu(self):
        from menu_scene import MenuScene   # local import avoids a circular dependency
        if self.menu is None:
            self.next = MenuScene(self.fonts)
        else:
            self.menu.next = None
            self.next = self.menu

    # result popup
    def make_popup(self):
        res, humans = self.match.result, self.match.humans()
        if res.winner is None:
            return GameOverPopup("Draw", res.reason, GREY)
        name = "White" if res.winner == chess.WHITE else "Black"
        if len(humans) == 1:
            won = humans[0] == res.winner
            return GameOverPopup("Bạn đã thắng!" if won else "Bạn đã thua", f"{res.reason} - {name} thắng",
                                 GREEN if won else RED)
        return GameOverPopup(f"{name} đã thắng!", res.reason, GREEN)

    # loop
    def update(self):
        if self.recording and self.result_recorder:
            try:
                if self.result_recorder.save_if_due():
                    self.record_status = f"Đã tự lưu: {os.path.basename(self.result_recorder.path)}"
            except OSError as ex:
                self.record_status = f"Lỗi lưu kết quả: {ex}"
        if self.next is not None or self.popup or self.ai_error:
            return
        if self.move_future is not None:
            if not self.move_future.done():
                return
            try:
                move = self.move_future.result()
            except Exception as exc:
                self.ai_error = f"Lỗi nước đi AI: {exc}"
            else:
                self.play(move)
            finally:
                self.move_future = None
            return
        player = self.match.current_player()
        if player.is_human:
            return
        if pygame.time.get_ticks() - self.last_ai_move >= self.ai_delay:
            if isinstance(player, ModelPlayer):
                if self.move_executor is None:
                    self.move_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chess-play")
                self.move_future = self.move_executor.submit(
                    player.choose_move, self.match.board.copy(stack=True))
            else:
                self.play(player.choose_move(self.match.board))

    def draw(self, screen):
        screen.fill(BG)
        draw_board(screen, self.fonts, self.match.board, self.match.last_move,
                   self.flipped, self.selected, self.targets)
        hints = ["ESC: về menu"]
        if self.recording_allowed:
            default_record_hint = ("R chọn chế độ đấu AI" if not self.match.humans()
                                   else "R ghi kết quả   X dừng   S lưu")
            hints.append(self.record_status or default_record_hint)
        if self.match.humans():
            hints.insert(0, "Chọn quân cờ, sau đó chọn ô màu xanh")
        if self.result_recorder:
            hints.insert(0, f"Elo Trắng {self.result_recorder.ratings['white']:.0f} / "
                             f"Đen {self.result_recorder.ratings['black']:.0f}")
        if len(self.match.humans()) < 2:
            hints.insert(0, f"UP/DOWN: AI delay ({self.ai_delay} ms)")
        if self.ai_error:
            hints.insert(0, self.ai_error)
        elif self.move_future is not None:
            hints.insert(0, "AI đang tính nước đi…")
        draw_play_panel(screen, self.fonts, self.match, hints)
        if self.promotion:
            dim = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 150))
            screen.blit(dim, (0, 0))
            label = self.fonts["main"].render("Thăng tiến lên:  (Esc để thoát)", True, (240, 240, 240))
            screen.blit(label, label.get_rect(center=(BOARD_PX // 2, BOARD_PX // 2 - 50)))
            for _, button in self.promo_buttons:
                button.draw(screen, self.fonts["main"])
        if self.popup:
            self.popup.draw(screen, self.fonts)
        if self.recording_popup:
            self.recording_popup.draw(screen, self.fonts)

    def on_exit(self):
        if self.move_executor is not None:
            self.move_executor.shutdown(wait=False, cancel_futures=True)
        if self.result_recorder and self.result_recorder.games:
            self.save_results()
