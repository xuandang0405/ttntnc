import os
 
import pygame
 
from algorithms.algorithms import ALGORITHM_NAMES, read_checkpoint_info
from concurrent_mode.concurrent_config import NUM_GAMES
from concurrent_mode.concurrent_train_scene import ConcurrentTrainScene
from config import BG, H, TEXT, W
from file_dialog import pick_model_file
from model_picker import ModelPicker
from model_presets import find_model_presets
from play_scene import PlayScene
from scene import QUIT, Scene
from normal_mode.train_scene import TrainScene
from ui import Button
 
MODES = ["Người vs Người", "Người vs AI", "AI vs AI", "Huấn luyện: 1 ván", f"Huấn luyện: {NUM_GAMES} ván"]
CHOICES = {
    "Bạn chơi": ["Trắng", "Đen"],
    "Thuật toán Trắng": ALGORITHM_NAMES,
    "Thuật toán Đen": ALGORITHM_NAMES,
    "Kiểu AI Encoding": ["Sampling", "Greedy"],
    "Đoán nước đi": ["Tắt", "Bật"],
}
TRAIN_ROWS = [("cycle", "Thuật toán Trắng"), ("file", "Tệp mô hình Trắng"),
              ("cycle", "Thuật toán Đen"), ("file", "Tệp mô hình Đen")]
ROWS = {
    0: [],
    1: [("cycle", "Bạn chơi"), ("model", "Đối thủ"), ("cycle", "Kiểu AI Encoding"), ("cycle", "Đoán nước đi")],
    2: [("model", "Người chơi Trắng"), ("model", "Người chơi Đen"), ("cycle", "Kiểu AI Encoding"), ("cycle", "Đoán nước đi")],
    3: TRAIN_ROWS,
    4: TRAIN_ROWS,
}

FILE_OF = {"Thuật toán Trắng": "Tệp mô hình Trắng", "Thuật toán Đen": "Tệp mô hình Đen"}
DESCRIPTIONS = {
    0: "Hai người chơi chung một màn hình. Nhấp vào một quân cờ, sau đó nhấp vào ô được làm sáng.",
    1: "Chọn màu quân của bạn và mô hình đối thủ.",
    2: "Chọn mô hình riêng cho Trắng và Đen.",
    3: "Huấn luyện các tác nhân Trắng và Đen bằng cách tự đấu, từng ván một.",
    4: f"Huấn luyện trên {NUM_GAMES} ván tự đấu chạy song song (sử dụng GPU nếu có).",
}
HINTS = {
    1: "Bấm hàng đối thủ để mở danh sách mô hình. Chuột phải chọn lại người chơi ngẫu nhiên.",
    2: "Bấm từng hàng để chọn mô hình. Chuột phải chọn lại người chơi ngẫu nhiên.",
    3: "Bấm hàng tệp để tiếp tục huấn luyện. Chuột phải để chọn mô hình mới.",
    4: "Bấm hàng tệp để tiếp tục huấn luyện. Chuột phải để chọn mô hình mới.",
}

class MenuScene(Scene):
    def __init__(self, fonts):
        super().__init__(fonts)
        self.mode = 1
        self.sel = {k: 0 for k in CHOICES}
        self.paths = {}      # tên hàng tệp -> đường dẫn mô hình đã chọn
        self.labels = {}     # đường dẫn mô hình -> nhãn thuật toán
        self.presets = {}    # hàng thi đấu -> mô hình đã huấn luyện
        self.model_picker = None
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
        if kind == "model" and name in self.presets:
            return self.presets[name].title
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
            if self.mode >= 3 and info["legacy"]:
                raise ValueError("Checkpoint 16 kênh dùng trong chế độ chơi. Huấn luyện cần checkpoint 56 kênh.")
        except Exception as ex:
            self.error = f"Không phải tệp mô hình hợp lệ: {ex}"
            return
        self.paths[name], self.labels[path], self.error = path, info["label"], ""
        self.presets.pop(name, None)

    def model_is_white(self, name):
        if name == "Đối thủ":
            return self.sel["Bạn chơi"] == 1
        return name == "Người chơi Trắng"

    def model_spec(self, name):
        if name in self.paths:
            return f"file:{self.paths[name]}"
        preset = self.presets.get(name)
        if preset is None:
            return "Random mover"
        checkpoint = preset.checkpoint(self.model_is_white(name))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Thiếu {preset.title} ({checkpoint.name}).")
        return f"file:{checkpoint}"
 
    # đầu vào
    def handle_event(self, e):
        if self.model_picker:
            selection = self.model_picker.handle_event(e)
            if selection is None:
                return
            name = self.model_picker.row
            self.model_picker = None
            if selection == "browse":
                self.pick(name)
            elif selection != "cancel":
                self.paths.pop(name, None)
                if selection == "random":
                    self.presets.pop(name, None)
                else:
                    self.presets[name] = selection
                self.error = ""
            return
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
                elif kind == "model" and e.button == 1:
                    self.model_picker = ModelPicker(
                        name, find_model_presets(), self.presets.get(name), self.model_is_white(name))
                elif e.button == 1:
                    self.pick(name)
                else:
                    self.paths.pop(name, None)
                    self.presets.pop(name, None)
                    self.error = ""
                return
            if e.button == 1 and self.start_btn.hit(e.pos):
                self.start()
 
    def cfg(self, side):
        return {"algorithm": CHOICES[f"Thuật toán {side}"][self.sel[f"Thuật toán {side}"]],
                "path": self.paths.get(f"Tệp mô hình {side}")}
 
    def start(self):
        try:
            if self.mode == 3:
                self.next = TrainScene(self.fonts, self.cfg("Trắng"), self.cfg("Đen"))
            elif self.mode == 4:
                self.next = ConcurrentTrainScene(self.fonts, self.cfg("Trắng"), self.cfg("Đen"))
            else:
                greedy = self.sel["Kiểu AI Encoding"] == 1
                lookahead = self.sel["Đoán nước đi"] == 1

                if self.mode == 0:
                    white, black = "Human", "Human"
                elif self.mode == 1:
                    opponent = self.model_spec("Đối thủ")
                    white, black = (opponent, "Human") if self.model_is_white("Đối thủ") else ("Human", opponent)
                else:
                    white = self.model_spec("Người chơi Trắng")
                    black = self.model_spec("Người chơi Đen")
                self.next = PlayScene(self.fonts, white, black, greedy, lookahead)
                self.next.menu = self
        except Exception as ex:
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
        if self.mode in HINTS:
            hint = self.fonts["small"].render(HINTS[self.mode], True, (160, 160, 170))
            screen.blit(hint, hint.get_rect(center=(W // 2, 196 + len(rows) * 56 + 13)))
        self.start_btn.draw(screen, self.fonts["title"])
        if self.error:
            message = self.fonts["small"].render(self.error[:140], True, (240, 120, 120))
            screen.blit(message, message.get_rect(center=(W // 2, H - 24)))
        if self.model_picker:
            self.model_picker.draw(screen, self.fonts)
