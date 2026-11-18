"""Playing a game: human vs human, human vs model, or model vs model."""
import chess
import pygame

from config import AI_MOVE_DELAY, BG, MAX_AI_PLIES, BOARD_PX
from match import Match
from players import make_player
from normal_mode.renderer import draw_board, draw_play_panel, px_to_sq
from scene import Scene
from ui import GameOverPopup, Button

GREEN, RED, GREY = (90, 200, 120), (230, 90, 90), (190, 190, 200)
PROMOTION_CHOICES = [(chess.QUEEN, "Queen"), (chess.ROOK, "Rook"), (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight")]

class PlayScene(Scene):
    def __init__(self, fonts, white_spec, black_spec, greedy, lookahead=False):
        super().__init__(fonts)
        self.specs, self.greedy, self.lookahead = (white_spec, black_spec), greedy, lookahead
        self.ai_delay = AI_MOVE_DELAY
        self.new_match()

    def new_match(self):
        white = make_player(self.specs[0], chess.WHITE, self.greedy, self.lookahead)
        black = make_player(self.specs[1], chess.BLACK, self.greedy, self.lookahead)
        humans = [p.is_human for p in (white, black)]
        self.match = Match(white, black, max_plies=None if any(humans) else MAX_AI_PLIES)
        self.flipped = humans == [False, True]       # human plays Black -> Black at the bottom
        self.selected, self.targets = None, ()
        self.promotion = None       
        self.popup = None
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
            self.popup = self.make_popup()

    def go_menu(self):
        from menu_scene import MenuScene   # local import avoids a circular dependency
        self.next = MenuScene(self.fonts)

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
        if self.popup:
            return
        player = self.match.current_player()
        if player.is_human:
            return
        if pygame.time.get_ticks() - self.last_ai_move >= self.ai_delay:
            self.play(player.choose_move(self.match.board))

    def draw(self, screen):
        screen.fill(BG)
        draw_board(screen, self.fonts, self.match.board, self.match.last_move,
                   self.flipped, self.selected, self.targets)
        hints = ["ESC: về menu"]
        if self.match.humans():
            hints.insert(0, "Chọn quân cờ, sau đó chọn ô màu xanh")
        if len(self.match.humans()) < 2:
            hints.insert(0, f"UP/DOWN: AI delay ({self.ai_delay} ms)")
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