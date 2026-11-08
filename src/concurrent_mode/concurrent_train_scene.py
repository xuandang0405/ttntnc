"""Self-play training view with k games running (and rendered) in parallel."""
import time

import pygame

from concurrent_mode.concurrent_game import ConcurrentSelfPlay
from concurrent_mode.concurrent_renderer import draw_concurrent_panel, draw_games_grid
from config import BG
from scene import Scene


class ConcurrentTrainScene(Scene):
    def __init__(self, fonts):
        super().__init__(fonts)
        self.game = ConcurrentSelfPlay()
        if self.game.load_models():
            print("Loaded saved models.")
        self.paused, self.fast, self.delay = False, False, 300
        self.last_step, self.last_save = 0, time.time()

    def handle_event(self, e):
        if e.type != pygame.KEYDOWN:
            return
        g = self.game
        if e.key == pygame.K_ESCAPE:
            from menu_scene import MenuScene
            self.next = MenuScene(self.fonts)
        elif e.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif e.key == pygame.K_f:
            self.fast = not self.fast
        elif e.key == pygame.K_UP:
            self.delay = max(0, self.delay - 50)
        elif e.key == pygame.K_DOWN:
            self.delay = min(2000, self.delay + 50)
        elif e.key == pygame.K_s:
            g.save_models(); print("Saved.")
        elif e.key == pygame.K_l:
            if g.load_models():
                print("Loaded.")
        elif e.key == pygame.K_r:
            g.stats_reset()

    def update(self):
        if not self.paused:
            if self.fast:   # as many steps as fit in ~25 ms so the window stays responsive
                t0 = time.time()
                while time.time() - t0 < 0.025:
                    self.game.step()
            else:
                now = pygame.time.get_ticks()
                if now - self.last_step >= self.delay:
                    self.game.step()
                    self.last_step = now
        if time.time() - self.last_save > 120:
            self.game.save_models()
            self.last_save = time.time()

    def draw(self, screen):
        screen.fill(BG)
        draw_games_grid(screen, self.fonts, self.game.slots)
        draw_concurrent_panel(screen, self.fonts, self.game, self.paused, self.fast, self.delay)

    def on_exit(self):
        self.game.save_models()