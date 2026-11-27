import os

import pygame
import torch
 
from config import H, W
from menu_scene import MenuScene
from model import device
from normal_mode.renderer import make_fonts
from scene import QUIT

def main():
    scene = None
    try:
        if device.type == "cpu" and not (os.getenv("OMP_NUM_THREADS") or os.getenv("MKL_NUM_THREADS")):
            torch.set_num_threads(min(4, torch.get_num_threads()))
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Cờ vua TTNTNC")
        fonts = make_fonts()
        clock = pygame.time.Clock()
        scene = MenuScene(fonts)

        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    break
                else:
                    scene.handle_event(e)

            if not running:
                break
            scene.update()
            scene.draw(screen)
            pygame.display.flip()

            if scene.next is not None:
                previous, destination = scene, scene.next
                scene = None if destination is QUIT else destination
                running = scene is not None
                previous.on_exit()
            clock.tick(60)
    finally:
        if scene is not None:
            scene.on_exit()
        try:
            from stockfish.stockfish_coach import close_coach
        except ImportError:
            pass
        try:
            close_coach()
        except Exception:
            pass
        pygame.quit()
 
 
if __name__ == "__main__":
    main()
