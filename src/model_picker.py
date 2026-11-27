import pygame

from config import H, TEXT, W
from ui import Button


PAGE_SIZE = 6


class ModelPicker:
    def __init__(self, row, presets, selected, white):
        self.row = row
        self.groups = [
            [preset for preset in presets if preset.campaign and preset.checkpoint(white).is_file()],
            [preset for preset in presets if not preset.campaign and preset.checkpoint(white).is_file()],
        ]
        self.selected = selected
        self.group = int(not selected.campaign) if selected else int(not self.groups[0])
        self.page = (self.groups[self.group].index(selected) // PAGE_SIZE
                     if selected in self.groups[self.group] else 0)
        self.box = pygame.Rect(0, 0, 616, 488)
        self.box.center = (W // 2, H // 2)
        x, y = self.box.topleft
        self.tabs = [
            Button((x + 30, y + 59, 270, 36), "Chọn nhanh"),
            Button((x + 316, y + 59, 270, 36), "Thí nghiệm"),
        ]
        self.previous = Button((x + 30, y + 340, 95, 32), "‹  Trước")
        self.following = Button((x + 491, y + 340, 95, 32), "Sau  ›")
        self.random = Button((x + 30, y + 381, 270, 38), "Người chơi ngẫu nhiên")
        self.browse = Button((x + 316, y + 381, 270, 38), "Chọn tệp .pt / .pth")
        self.cancel = Button((x + 224, y + 436, 168, 34), "Quay lại (Esc)")
        self.dim = pygame.Surface((W, H), pygame.SRCALPHA)
        self.dim.fill((0, 0, 0, 168))

    @property
    def page_count(self):
        return max(1, (len(self.groups[self.group]) + PAGE_SIZE - 1) // PAGE_SIZE)

    def rows(self):
        start = self.page * PAGE_SIZE
        return [
            (preset, Button((self.box.x + 30, self.box.y + 107 + 38 * index, 556, 33),
                            preset.title, base=(55, 65, 83), hover=(76, 100, 130)))
            for index, preset in enumerate(self.groups[self.group][start:start + PAGE_SIZE])
        ]

    def change_page(self, step):
        self.page = min(self.page_count - 1, max(0, self.page + step))

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "cancel"
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.change_page(1 if event.key == pygame.K_RIGHT else -1)
        elif event.type == pygame.MOUSEWHEEL:
            self.change_page(-event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, button in enumerate(self.tabs):
                if button.hit(event.pos):
                    self.group, self.page = index, 0
                    return None
            for preset, button in self.rows():
                if button.hit(event.pos):
                    return preset
            if self.previous.hit(event.pos):
                self.change_page(-1)
            elif self.following.hit(event.pos):
                self.change_page(1)
            elif self.random.hit(event.pos):
                return "random"
            elif self.browse.hit(event.pos):
                return "browse"
            elif self.cancel.hit(event.pos):
                return "cancel"
        return None

    def draw(self, screen, fonts):
        screen.blit(self.dim, (0, 0))
        pygame.draw.rect(screen, (41, 43, 51), self.box, border_radius=12)
        pygame.draw.rect(screen, (103, 130, 165), self.box, 2, border_radius=12)
        title = fonts["title"].render(f"Chọn mô hình cho {self.row.lower()}", True, TEXT)
        screen.blit(title, title.get_rect(center=(self.box.centerx, self.box.y + 28)))
        for index, button in enumerate(self.tabs):
            button.draw(screen, fonts["main"], active=index == self.group)
        for preset, button in self.rows():
            button.draw(screen, fonts["main"], active=preset == self.selected)
        if not self.groups[self.group]:
            message = fonts["main"].render("Chưa có checkpoint cho màu quân này.", True, TEXT)
            screen.blit(message, message.get_rect(center=self.box.center))
        if self.page_count > 1:
            if self.page > 0:
                self.previous.draw(screen, fonts["main"])
            if self.page < self.page_count - 1:
                self.following.draw(screen, fonts["main"])
            message = fonts["main"].render(f"Trang {self.page + 1} / {self.page_count}", True, TEXT)
            screen.blit(message, message.get_rect(center=(self.box.centerx, self.previous.rect.centery)))
        for button in (self.random, self.browse, self.cancel):
            button.draw(screen, fonts["main"])
