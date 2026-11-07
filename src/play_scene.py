import chess
import pygame

from config import AI_MOVE_DELAY, BG, MAX_AI_PLIES
from match import Match
from players import make_player
from renderer import draw_board, draw_play_panel, px_to_sq
from scene import Scene
from ui import GameOverPopup

GREEN, RED, GREY = (90, 200, 120), (230, 90, 90), (190, 190, 200)


class PlayScene(Scene):
    def __init__(self, fonts, white_spec, black_spec, greedy):
        super().__init__(fonts)
        self.specs, self.greedy = (white_spec, black_spec), greedy
        self.ai_delay = AI_MOVE_DELAY
        self.new_match()

    def new_match(self):
        white = make_player(self.specs[0], chess.WHITE, self.greedy)
        black = make_player(self.specs[1], chess.BLACK, self.greedy)
        humans = [p.is_human for p in (white, black)]
        self.match = Match(white, black, max_plies=None if any(humans) else MAX_AI_PLIES)
        self.flipped = humans == [False, True]       # human plays Black -> Black at the bottom
        self.selected, self.targets = None, ()
        self.popup = None
        self.last_ai_move = pygame.time.get_ticks()

    # ---- input ----
    def handle_event(self, e):
        if self.popup:
            action = self.popup.handle_event(e)
            if action == "rematch":
                self.new_match()
            elif action == "menu":
                self.go_menu()
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
            for mv in board.legal_moves:   # promotions auto-queen
                if mv.from_square == self.selected and mv.to_square == sq and mv.promotion in (None, chess.QUEEN):
                    self.selected, self.targets = None, ()
                    self.play(mv)
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
            return GameOverPopup("Bạn đã thắng!" if won else "Bạn đã thua!", f"{res.reason} - {name} wins",
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
        hints = ["ESC: back to menu"]
        if self.match.humans():
            hints.insert(0, "Click a piece, then a blue square")
        if len(self.match.humans()) < 2:
            hints.insert(0, f"UP/DOWN: AI delay ({self.ai_delay} ms)")
        draw_play_panel(screen, self.fonts, self.match, hints)
        if self.popup:
            self.popup.draw(screen, self.fonts)