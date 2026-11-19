# Cờ Vua Học Tăng Cường

Ứng dụng Cờ Vua bằng Pygame kết hợp luật cờ vua, huấn luyện tự đấu và nhiều thuật toán Reinforcement Learning:

- **A2C** - Advantage Actor-Critic
- **PPO** - Proximal Policy Optimization
- **DQN** - Double Deep Q-Network
- **AlphaZero** - mạng policy/value kết hợp Monte Carlo Tree Search

Mã nguồn chính nằm trong thư mục `src/`. Mạng nơ-ron dùng phiên bản tác tử duy nhất cho cả hai phe, có đầu dự đoán:

- policy cho các nước đi
- value cho trạng thái bàn cờ
- opponent policy
- opening/middlegame/endgame phase

---

## Tính năng

### Chơi cờ

- Người vs Người
- Người vs AI
- AI vs AI
- Tải mô hình `.pt` bằng hộp thoại chọn tệp
- Chơi bằng người ngẫu nhiên khi không chọn mô hình
- Hai kiểu chọn nước đi:
  - **Sampling**: lấy nước đi theo xác suất
  - **Greedy**: luôn chọn nước đi được đánh giá cao nhất
- **Opponent-aware lookahead**: dự đoán một số nước đáp trả có khả năng cao của đối thủ

### Huấn luyện

- Huấn luyện một ván dùng chế độ tự đấu tuần tự
- Huấn luyện nhiều ván song song
- 128 bàn cờ song song mặc định
- Hỗ trợ mở đầu ngẫu nhiên, trung cuộc và tàn cuộc theo curriculum
- Tiếp tục huấn luyện từ checkpoint cũ hoặc huấn luyện mô hình mới
- Tự lưu checkpoint và chỉ số huấn luyện định kỳ
- Kiểm tra kỹ năng mate-in-one, tránh chiếu hết, bắt quân tự do và chuyển đổi tàn cuộc
- Ghi log và vẽ biểu đồ reward, loss, entropy, win/draw rate, throughput và draw reason
- AlphaZero có thể chạy MCTS bằng native Rust để chuyển luật cờ, mã hóa trạng thái và cây tìm kiếm khỏi Python

---

## Kiến trúc hiện tại

Phiên bản mạng và checkpoint hiện tại là **v3**.

| Thành phần | Giá trị |
|---|---|
| Lịch sử trạng thái | 8 bàn cờ |
| Mặt nơ-ron bàn cờ | 12 mặt quân cờ cho mỗi lịch sử |
| Mặt metadata | 8 mặt |
| Kích thước đầu vào | `104 x 8 x 8` |
| Không gian hành động | `20,480` |
| Residual blocks | 6 |
| Channels | 64 |
| Kiến trúc | Policy/Value Residual CNN |
| Ván tối đa mặc định | 600 nửa nước |

Không gian hành động gồm:

- `64 x 64 = 4,096` cặp ô đi - ô đến
- 3 × `4,096` mặt hành động phong hậu, xe, tượng
- 1 × `4,096` mặt hành động phong mã

Toàn bộ trạng thái trong MCTS được sao chép kèm lịch sử nước đi để giữ repetition, halfmove clock và fullmove number. Dữ liệu replay được lưu dưới dạng `float16` và chuyển sang `float32` khi huấn luyện.

> **Không tương thích checkpoint cũ:** checkpoint v1/v2 không thể nạp trực tiếp vào kiến trúc v3 vì số channel đầu vào và kích thước policy đều đã thay đổi. Hãy bắt đầu một phiên huấn luyện mới thay vì tải checkpoint cũ như checkpoint tương thích.

---

## Cấu trúc thư mục

```text
src/
├── main.py                     # Khởi chạy Pygame
├── config.py                   # Cửa sổ, màu sắc, reward và giới hạn ván
├── menu_scene.py               # Chọn chế độ, thuật toán và checkpoint
├── play_scene.py               # Luồng chơi người/người hoặc người/AI
├── match.py                    # Quản lý một trận đấu
├── players.py                  # Human, random và model players
├── checkpoints.py              # Tạo experiment, lưu metrics và checkpoint
├── encoding.py                 # Tensor trạng thái 104 x 8 x 8 và action mapping
├── model.py                    # Residual policy/value network
├── metrics.py                  # Thống kê ván, reward và kết quả huấn luyện
├── curriculum.py               # Opening, middlegame và endgame positions
├── strategy.py                 # Dự đoán đối thủ và opponent-aware lookahead
├── strategy_config.py          # Tham số strategy và curriculum
├── skill_eval.py               # Bộ kiểm tra kỹ năng cờ
│
├── algorithms/
│   ├── algorithms.py           # Factory và nạp checkpoint theo thuật toán
│   ├── algo_base.py            # Hành vi chung của tác tử
│   ├── algo_a2c.py
│   ├── algo_ppo.py
│   ├── algo_dqn.py
│   └── algo_alphazero.py
│
├── algorithms-rs/              # Rust MCTS backend optional cho AlphaZero
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── README.md
│   └── src/
│
├── normal_mode/
│   ├── game.py                 # Tự đấu và huấn luyện tuần tự
│   ├── train_scene.py          # Giao diện huấn luyện một ván
│   └── renderer.py             # Bàn cờ và panel Pygame
│
├── concurrent_mode/
│   ├── concurrent_config.py    # Số ván, thiết bị và chu kỳ cập nhật
│   ├── concurrent_game.py      # Tự đấu và huấn luyện nhiều ván
│   ├── concurrent_train_scene.py
│   └── concurrent_renderer.py
│
└── plot_logging/
    ├── training_log.py          # Ghi training_log.json
    └── plot_training.py         # Vẽ biểu đồ huấn luyện

assets/
└── chess_pieces/                # Hình quân cờ PNG

checkpoints/
└── exp_YYYYMMDD_HHMMSS/
    ├── white_<algo>.pt
    ├── black_<algo>.pt
    ├── metrics.json
    └── training_log.json
```

`src2/` là bản sao/thư mục tích hợp thử nghiệm và không phải đường chạy chính của README. Hãy dùng `src/` cho phiên bản hiện tại.

---

## Yêu cầu

- Python 3.10 trở lên
- Pygame
- PyTorch
- NumPy
- python-chess
- Matplotlib, pandas, seaborn để vẽ biểu đồ

Tạo môi trường và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install pygame torch numpy matplotlib pandas seaborn python-chess
```

Nếu máy có NVIDIA GPU, cài PyTorch với build CUDA phù hợp để tăng tốc huấn luyện. GPU là tùy chọn; ứng dụng có thể chạy bằng CPU.

AlphaZero có thể dùng MCTS native trong Rust. Hai backend Rust được hỗ trợ:

| Biến môi trường | Backend |
|---|---|
| `CHESS_AI_RUST_SEARCH=1` | `BatchSearch` native: Rust xử lý luật cờ, mã hóa thế cờ, cây tìm kiếm, chọn nhánh và backup |
| `CHESS_AI_RUST_MCTS=1` | Backend cây Rust cũ: chỉ một số bước PUCT/expand/backup chạy trong Rust |

Hai biến này mặc định là `1` trong cấu hình hiện tại. Để dùng MCTS thuần Python,
đặt cả hai biến thành `0`:

```powershell
$env:CHESS_AI_RUST_SEARCH = "0"
$env:CHESS_AI_RUST_MCTS = "0"
python src\main.py
```

Để cài extension Rust, chạy từ thư mục crate:

```bash
cd src/algorithms-rs
pip install maturin
python -m pip install .
```

Sau đó khởi chạy với native search mặc định:

```powershell
python src\main.py
```

Nếu đặt `CHESS_AI_RUST_SEARCH=1` hoặc `CHESS_AI_RUST_MCTS=1` nhưng extension
chưa được cài, ứng dụng sẽ báo lỗi rõ ràng thay vì tự động chuyển sang Python.
Backend native lấy seed nhiễu và seed RNG của NumPy; thứ tự nước đi có thể khác
nhưng visit counts và action parity được kiểm tra theo tập test của Rust.

---

## Chạy ứng dụng

Luôn chạy từ thư mục gốc của repository để Pygame có thể tìm đúng thư mục `assets/`:

```bash
python src/main.py
```

Hoặc:

```bash
cd src
python main.py
```

> Lưu ý: thư mục làm việc hợp lệ cho ứng dụng là repository root. Khi chạy lệnh dưới đây, vị trí hiện tại phải là thư mục gốc của dự án.

---

## Điều khiển trong menu

1. Chọn một trong 5 chế độ ở thanh nút phía trên.
2. Cấu hình các hàng tùy chọn bằng chuột:
   - **Nhấp trái**: chuyển sang lựa chọn tiếp theo hoặc mở hộp thoại mô hình.
   - **Nhấp phải**: quay lại lựa chọn trước hoặc xóa mô hình đã chọn.
3. Nhấn **BẮT ĐẦU** hoặc `Enter`.
4. Nhấn `Esc` để thoát.

Trong chế độ chơi và huấn luyện:

- Nhấp quân cờ, sau đó nhấp ô được làm sáng để đi quân.
- Trong màn hình huấn luyện:
  - `Space`: tạm dừng/tiếp tục
  - `F`: bật/tắt chế độ huấn luyện nhanh
  - `Up` / `Down`: chỉnh độ trễ
  - `S`: lưu checkpoint
  - `L`: tải checkpoint
  - `R`: đặt lại thống kê
  - `M`: bật/tắt panel metrics
  - `Esc`: quay lại menu

---

## Tải và tiếp tục huấn luyện

Mở menu `Huấn luyện: 1 ván` hoặc `Huấn luyện: 128 ván`, sau đó chọn tệp ở hàng **Tệp mô hình Trắng** hoặc **Tệp mô hình Đen**.

- Nhấp trái vào hàng tệp: chọn checkpoint `.pt`
- Nhấp phải vào hàng tệp: bỏ checkpoint và huấn luyện mô hình mới
- Thuật toán tương ứng được đọc tự động từ metadata checkpoint

Checkpoint mới được lưu trong `checkpoints/exp_<thời gian>/` và tự lưu lại sau khoảng 120 giây. Khi thoát màn hình huấn luyện bằng `Esc`, ứng dụng cũng lưu checkpoint trước khi chuyển scene.

Nếu chọn checkpoint cho một phe, phe còn lại có thể được khởi tạo với thuật toán đã chọn hoặc huấn luyện từ đầu tùy cấu hình tại menu.

---

## Xem kết quả huấn luyện

Các chỉ số được lưu trong `metrics.json` và `training_log.json`.

Vẽ biểu đồ từ checkpoint mới nhất:

```bash
python src/plot_logging/plot_training.py
```

Vẽ biểu đồ từ một log cụ thể:

```bash
python src/plot_logging/plot_training.py checkpoints/exp_YYYYMMDD_HHMMSS/training_log.json
```

Biểu đồ chứa win/draw rate, actor/critic loss, entropy, value explained variance, reward, repetition, độ dài ván, throughput, CPU/GPU utilization, nguyên nhân hòa và thời gian từng giai đoạn của thuật toán. `training_log.json` lưu thời gian trung bình và p95 (ms) trong tối đa 256 lần gọi gần nhất cho từng giai đoạn: chọn nước đi, các bước MCTS, chuẩn bị episode/replay/batch, forward/loss/backward, optimizer và strategy. Đây là thời gian wall-clock; phép tính GPU không bị đồng bộ cưỡng bức nên không làm chậm huấn luyện.

---

## Kiểm tra mã nguồn

Test import các module theo cấu trúc chạy của ứng dụng, nên chạy từ thư mục `src`:

```bash
cd src
python -m compileall -q .
python -m unittest discover -s tests -v
```

Nếu đã build extension Rust, cũng chạy test parity cho native search:

```bash
cd src/algorithms-rs
cargo test

cd ../
python -m unittest tests.test_rust_search
```
