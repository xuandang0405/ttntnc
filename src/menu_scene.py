import pygame

from agent import models_available
from config import BG, H, TEXT, W
from play_scene import PlayScene
from players import PLAYER_CHOICES
from scene import QUIT, Scene
from train_scene import TrainScene
from ui import Button

MODES = ["Người vs Người", "Người vs AI", "AI vs AI", "Huấn luyện AI"]
CHOICES = {
    "Bạn chơi": ["Trắng", "Đen"],
    "Đối thủ": PLAYER_CHOICES,
    "Người chơi Trắng": PLAYER_CHOICES,
    "Người chơi Đen": PLAYER_CHOICES,
    "Kiểu AI": ["Lấy mẫu (đa dạng)", "Tham lam (mạnh nhất)"],
}

ROWS = {0: [], 1: ["Bạn chơi", "Đối thủ", "Kiểu AI"], 2: ["Người chơi Trắng", "Người chơi Đen", "Kiểu AI"], 3: []}
DESCRIPTIONS = {
    0: "Hai người chơi chung trên màn hình này. Nhấp vào một quân cờ, sau đó nhấp vào ô được đánh dấu.",
    1: "Chơi với mô hình đã huấn luyện (hoặc đối thủ di chuyển ngẫu nhiên).",
    2: "Xem hai mô hình (hoặc đối thủ di chuyển ngẫu nhiên) thi đấu với nhau.",
    3: "Để hai mô hình Trắng và Đen tự học bằng cách đấu với nhau.",
}


class MenuScene(Scene):
    def __init__(self, fonts):
        super().__init__(fonts)
        self.mode = 1
        self.sel = {k: 0 for k in CHOICES}
        self.sel["Người chơi Đen"] = 1
        self.start_btn = Button((W // 2 - 140, H - 120, 280, 56), "BẮT ĐẦU", (50, 130, 90), (70, 160, 110))
        bw = (W - 80) // len(MODES)
        self.mode_btns = [Button((40 + i * bw, 110, bw - 10, 50), m) for i, m in enumerate(MODES)]

    def row_buttons(self):
        out = []
        for i, name in enumerate(ROWS[self.mode]):
            val = CHOICES[name][self.sel[name]]
            out.append((name, Button((W // 2 - 280, 200 + i * 66, 560, 52), f"{name}:    {val}")))
        return out

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                self.start()
            elif e.key == pygame.K_ESCAPE:
                self.next = QUIT
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            for i, b in enumerate(self.mode_btns):
                if b.hit(e.pos):
                    self.mode = i
                    return
            for name, b in self.row_buttons():
                if b.hit(e.pos):
                    step = 1 if e.button == 1 else -1
                    self.sel[name] = (self.sel[name] + step) % len(CHOICES[name])
                    return
            if e.button == 1 and self.start_btn.hit(e.pos):
                self.start()

    def start(self):
        if self.mode == 3:
            self.next = TrainScene(self.fonts)
            return
        greedy = self.sel["Kiểu AI"] == 1
        pick = lambda k: CHOICES[k][self.sel[k]]
        if self.mode == 0:
            white, black = "Human", "Human"
        elif self.mode == 1:
            white, black = ("Human", pick("Đối thủ")) if pick("Bạn chơi") == "White" \
                else (pick("Đối thủ"), "Human")
        else:
            white, black = pick("Người chơi Trắng"), pick("Người chơi Đen")
        self.next = PlayScene(self.fonts, white, black, greedy)

    def draw(self, screen):
        screen.fill(BG)
        t = self.fonts["big"].render("Cờ vua Học Tăng cường", True, TEXT)
        screen.blit(t, t.get_rect(center=(W // 2, 58)))
        for i, b in enumerate(self.mode_btns):
            b.draw(screen, self.fonts["main"], active=(i == self.mode))
        d = self.fonts["main"].render(DESCRIPTIONS[self.mode], True, (190, 190, 200))
        screen.blit(d, d.get_rect(center=(W // 2, 180)))
        for _, b in self.row_buttons():
            b.draw(screen, self.fonts["main"])
        if ROWS[self.mode]:
            h = self.fonts["small"].render("nhấp chuột trái: tùy chọn tiếp theo    chuột phải: quay lại", True, (140, 140, 150))
            screen.blit(h, h.get_rect(center=(W // 2, 200 + len(ROWS[self.mode]) * 66)))
        self.start_btn.draw(screen, self.fonts["title"])
        av = models_available()

        msg = ("Mô hình đã lưu: " + "   ".join(f"{n} {'đã có' + (' (màu Trắng)' if n=='white' else ' (màu Đen)' if n=='black' else '') if ok else 'thiếu'}" for n, ok in av.items())
                + ("" if all(av.values()) else "   (hãy dùng 'Huấn luyện AI' trước; mô hình chưa có sẽ đi ngẫu nhiên)"))
        m = self.fonts["small"].render(msg, True, (140, 140, 150))
        screen.blit(m, m.get_rect(center=(W // 2, H - 30)))