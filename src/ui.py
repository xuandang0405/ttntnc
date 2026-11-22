import pygame
from config import H, TEXT, W
from concurrent_mode.concurrent_config import NUM_GAMES

class Button:
    def __init__(self, rect, text, base=(70, 90, 140), hover=(95, 120, 180), active=(60, 150, 100)):
        self.rect, self.text = pygame.Rect(rect), text
        self.base, self.hover, self.active = base, hover, active

    def draw(self, screen, font, active=False):
        over = self.rect.collidepoint(pygame.mouse.get_pos())
        col = self.active if active else (self.hover if over else self.base)
        pygame.draw.rect(screen, col, self.rect, border_radius=8)
        t = font.render(self.text, True, TEXT)
        screen.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class GameOverPopup:
  
    def __init__(self, title, subtitle, accent):
        self.title, self.subtitle, self.accent = title, subtitle, accent
        self.box = pygame.Rect(0, 0, 460, 260)
        self.box.center = (W // 2, H // 2)
        b = self.box
        self.rematch = Button((b.x + 30, b.bottom - 74, 190, 48), "Đấu lại")
        self.menu = Button((b.right - 220, b.bottom - 74, 190, 48), "Về menu")
        self.dim = pygame.Surface((W, H), pygame.SRCALPHA)
        self.dim.fill((0, 0, 0, 170))

    def draw(self, screen, fonts):
        screen.blit(self.dim, (0, 0))
        pygame.draw.rect(screen, (44, 44, 52), self.box, border_radius=14)
        pygame.draw.rect(screen, self.accent, self.box, 4, border_radius=14)
        t = fonts["big"].render(self.title, True, self.accent)
        screen.blit(t, t.get_rect(center=(self.box.centerx, self.box.y + 70)))
        s = fonts["main"].render(self.subtitle, True, TEXT)
        screen.blit(s, s.get_rect(center=(self.box.centerx, self.box.y + 125)))
        self.rematch.draw(screen, fonts["main"])
        self.menu.draw(screen, fonts["main"])

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.rematch.hit(e.pos):
                return "rematch"
            if self.menu.hit(e.pos):
                return "menu"
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                return "rematch"
            if e.key == pygame.K_ESCAPE:
                return "menu"
        return None


class GameModePopup:
    def __init__(self):
        self.box = pygame.Rect(0, 0, 520, 270)
        self.box.center = (W // 2, H // 2)
        b = self.box
        self.single = Button((b.x + 30, b.y + 150, 220, 52), "Một ván")
        self.multiple = Button((b.right - 250, b.y + 150, 220, 52), f"{NUM_GAMES} ván song song")
        self.dim = pygame.Surface((W, H), pygame.SRCALPHA)
        self.dim.fill((0, 0, 0, 190))

    def draw(self, screen, fonts):
        screen.blit(self.dim, (0, 0))
        pygame.draw.rect(screen, (44, 44, 52), self.box, border_radius=14)
        pygame.draw.rect(screen, (100, 160, 220), self.box, 3, border_radius=14)
        title = fonts["big"].render("Chế độ đấu AI", True, TEXT)
        screen.blit(title, title.get_rect(center=(self.box.centerx, self.box.y + 60)))
        subtitle = fonts["main"].render("Chọn một ván hoặc chạy nhiều ván đồng thời", True, TEXT)
        screen.blit(subtitle, subtitle.get_rect(center=(self.box.centerx, self.box.y + 108)))
        self.single.draw(screen, fonts["main"])
        self.multiple.draw(screen, fonts["main"])

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.single.hit(e.pos):
                return "single"
            if self.multiple.hit(e.pos):
                return "multiple"
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_1, pygame.K_KP1):
                return "single"
            if e.key in (pygame.K_2, pygame.K_KP2):
                return "multiple"
            if e.key == pygame.K_ESCAPE:
                return "cancel"
        return None