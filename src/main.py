import pygame
 
from config import H, W
from menu_scene import MenuScene
from normal_mode.renderer import make_fonts
from scene import QUIT

def main():
    scene = None
    try:
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
                else:
                    scene.handle_event(e)

            scene.update()
            scene.draw(screen)
            pygame.display.flip()

            if scene.next is not None:
                scene.on_exit()
                if scene.next is QUIT:
                    running = False
                else:
                    scene = scene.next
            clock.tick(60)
    finally:
        try:
            from stockfish.stockfish_coach import close_coach
        except ImportError:
            from stockfish_coach import close_coach
        try:
            close_coach()
        except Exception:
            pass
        pygame.quit()
 
 
if __name__ == "__main__":
    main()