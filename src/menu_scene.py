import os
 
import pygame
 
from algorithms import ALGORITHM_NAMES, read_checkpoint_info
from concurrent_mode.concurrent_config import NUM_GAMES
from concurrent_mode.concurrent_train_scene import ConcurrentTrainScene
from config import BG, H, TEXT, W
from file_dialog import pick_model_file
from play_scene import PlayScene
from scene import QUIT, Scene
from train_scene import TrainScene
from ui import Button
 
MODES = ["Người vs Người", "Người vs AI", "AI vs AI", "Huấn luyện: 1 ván", f"Huấn luyện: {NUM_GAMES} ván"]
CHOICES = {
    "Bạn chơi": ["Trắng", "Đen"],
    "Thuật toán Trắng": ALGORITHM_NAMES,
    "Thuật toán Đen": ALGORITHM_NAMES,
    "Kiểu AI Encoding": ["Sampling", "Greedy"],
}
TRAIN_ROWS = [("cycle", "Thuật toán Trắng"), ("file", "Tệp mô hình Trắng"),
              ("cycle", "Thuật toán Đen"), ("file", "Tệp mô hình Đen")]
ROWS = {
    0: [],
    1: [("cycle", "Bạn chơi"), ("file", "Đối thủ"), ("cycle", "Kiểu AI Encoding")],
    2: [("file", "Người chơi Trắng"), ("file", "Người chơi Đen"), ("cycle", "Kiểu AI Encoding")],
    3: TRAIN_ROWS,
    4: TRAIN_ROWS,
}

FILE_OF = {"Thuật toán Trắng": "Tệp mô hình Trắng", "Thuật toán Đen": "Tệp mô hình Đen"}
DESCRIPTIONS = {
    0: "Hai người chơi chung một màn hình. Nhấp vào một quân cờ, sau đó nhấp vào ô được làm sáng.",
    1: "Chơi với mô hình đã lưu (chọn tệp mô hình) hoặc người chơi ngẫu nhiên.",
    2: "Xem hai mô hình đã lưu (hoặc người chơi ngẫu nhiên) đấu với nhau.",
    3: "Huấn luyện các tác nhân Trắng và Đen bằng cách tự đấu, từng ván một.",
    4: f"Huấn luyện trên {NUM_GAMES} ván tự đấu chạy song song (sử dụng GPU nếu có).",
}
HINTS = {
    1: "hàng tệp: nhấp chuột trái = chọn tệp mô hình, nhấp chuột phải = người chơi ngẫu nhiên",
    2: "hàng tệp: nhấp chuột trái = chọn tệp mô hình, nhấp chuột phải = người chơi ngẫu nhiên",
    3: "hàng tệp: nhấp chuột trái = tải mô hình đã lưu để tiếp tục huấn luyện, nhấp chuột phải = bắt đầu lại từ đầu",
    4: "hàng tệp: nhấp chuột trái = tải mô hình đã lưu để tiếp tục huấn luyện, nhấp chuột phải = bắt đầu lại từ đầu",
}

class MenuScene(Scene):
    def __init__(self, fonts):
        super().__init__(fonts)
        self.mode = 1
        self.sel = {k: 0 for k in CHOICES}
        self.paths = {}      # tên hàng tệp -> đường dẫn mô hình đã chọn
        self.labels = {}     # đường dẫn mô hình -> nhãn thuật toán
        self.error = ""
        self.start_btn = Button((W // 2 - 140, H - 100, 280, 52), "BẮT ĐẦU", (50, 130, 90), (70, 160, 110))
        bw = (W - 80) // len(MODES)
        self.mode_btns = [Button((40 + i * bw, 110, bw - 10, 50), m) for i, m in enumerate(MODES)]
 
    # hàng
    def row_buttons(self):
        out = []
        for i, (kind, name) in enumerate(ROWS[self.mode]):
            out.append((kind, name, Button((W // 2 - 300, 196 + i * 56, 600, 46), f"{name}:   {self.row_value(kind, name)}")))
        return out
 
    def row_value(self, kind, name):
        if kind == "cycle":
            file_row = FILE_OF.get(name)
            if file_row and self.paths.get(file_row):
                return f"{self.labels[self.paths[file_row]]}  (từ tệp)"
            return CHOICES[name][self.sel[name]]
        path = self.paths.get(name)
        if path:
            return f"{os.path.basename(path)}  [{self.labels[path]}]"
        return "(mô hình mới - huấn luyện từ đầu)" if self.mode >= 3 else "Người chơi ngẫu nhiên"
 
    def pick(self, name):
        path, err = pick_model_file(f"Chọn tệp mô hình cho: {name}")
        if err:
            self.error = err
            return
        if not path:
            return
        try:
            info = read_checkpoint_info(path)
        except Exception as ex:
            self.error = f"Không phải tệp mô hình hợp lệ: {ex}"
            return
        self.paths[name], self.labels[path], self.error = path, info["label"], ""
 
    # đầu vào
    def handle_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                self.start()
            elif e.key == pygame.K_ESCAPE:
                self.next = QUIT
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            for i, b in enumerate(self.mode_btns):
                if b.hit(e.pos):
                    self.mode, self.error = i, ""
                    return
            for kind, name, b in self.row_buttons():
                if not b.hit(e.pos):
                    continue
                if kind == "cycle":
                    if not self.paths.get(FILE_OF.get(name)):       # tệp đã tải sẽ quyết định thuật toán
                        step = 1 if e.button == 1 else -1
                        self.sel[name] = (self.sel[name] + step) % len(CHOICES[name])
                elif e.button == 1:
                    self.pick(name)
                else:
                    self.paths.pop(name, None)
                return
            if e.button == 1 and self.start_btn.hit(e.pos):
                self.start()
 
    def cfg(self, side):
        return {"algorithm": CHOICES[f"Thuật toán {side}"][self.sel[f"Thuật toán {side}"]],
                "path": self.paths.get(f"Tệp mô hình {side}")}
 
    def start(self):
        pick = lambda k: CHOICES[k][self.sel[k]]
        spec = lambda name: f"file:{self.paths[name]}" if self.paths.get(name) else "Người chơi ngẫu nhiên"
        try:
            if self.mode == 3:
                self.next = TrainScene(self.fonts)
            elif self.mode == 4:
                self.next = ConcurrentTrainScene(self.fonts)
            else:
                greedy = self.sel["Kiểu AI Encoding"] == 1
                if self.mode == 0:
                    white, black = "Human", "Human"
                elif self.mode == 1:
                    white, black = ("Human", spec("Đối thủ")) if pick("Bạn chơi") == "Trắng" \
                        else (spec("Đối thủ"), "Human")
                else:
                    white, black = spec("Người chơi Trắng"), spec("Người chơi Đen")
                self.next = PlayScene(self.fonts, white, black, greedy)
        except Exception as ex:             # ví dụ: tệp mô hình không thể tải
            self.next = None
            self.error = f"Không thể bắt đầu: {ex}"
 
    # vẽ giao diện
    def draw(self, screen):
        screen.fill(BG)
        t = self.fonts["big"].render("Cờ Vua Học Tăng Cường", True, TEXT)
        screen.blit(t, t.get_rect(center=(W // 2, 58)))
        for i, b in enumerate(self.mode_btns):
            b.draw(screen, self.fonts["main"], active=(i == self.mode))
        d = self.fonts["main"].render(DESCRIPTIONS[self.mode], True, (190, 190, 200))
        screen.blit(d, d.get_rect(center=(W // 2, 180)))
        rows = self.row_buttons()
        for _, _, b in rows:
            b.draw(screen, self.fonts["main"])
        if rows:
            h = self.fonts["small"].render("nhấp chuột trái: tùy chọn tiếp theo / chọn tệp    nhấp chuột phải: trước đó / xóa",
                                           True, (140, 140, 150))
            screen.blit(h, h.get_rect(center=(W // 2, 196 + len(rows) * 56 + 8)))
            if self.mode in HINTS:
                h2 = self.fonts["small"].render(HINTS[self.mode], True, (140, 140, 150))
                screen.blit(h2, h2.get_rect(center=(W // 2, 196 + len(rows) * 56 + 28)))
        self.start_btn.draw(screen, self.fonts["title"])
        msg, color = (self.error, (240, 120, 120)) if self.error else \
            ("Chọn mô hình thông qua hộp thoại.", (140, 140, 150))
        m = self.fonts["small"].render(msg[:120], True, color)
        screen.blit(m, m.get_rect(center=(W // 2, H - 24)))