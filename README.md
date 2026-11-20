# Cờ Vua Học Tăng Cường

Ứng dụng cờ vua Pygame kết hợp giao diện chơi cờ, huấn luyện tự đối kháng và nhiều thuật toán reinforcement learning:

- **A2C** - Advantage Actor-Critic
- **PPO** - Proximal Policy Optimization
- **DQN** - Deep Q-Network
- **AlphaZero** - policy/value network kết hợp Monte Carlo Tree Search

Mã nguồn hiện tại nằm trong `src/`. Các agent dùng cùng mạng policy/value cho cả hai phe và có các đầu dự đoán:

- policy cho các nước đi hợp lệ
- value cho trạng thái bàn cờ
- opponent policy
- opening/middlegame/endgame phase

## Showcase

| Chơi cờ | Huấn luyện |
|---|---|
| ![Giao diện chơi cờ](assets/showcase/image1.png) | ![Giao diện huấn luyện](assets/showcase/image2.png) |

## Tính năng

### Chơi cờ

- Người vs Người
- Người vs AI
- AI vs AI
- Tải checkpoint `.pt`
- Chơi với người chơi ngẫu nhiên khi không chọn mô hình
- Chọn nước đi theo xác suất (**Sampling**) hoặc theo điểm cao nhất (**Greedy**)
- **Opponent-aware lookahead** để dự đoán các nước đáp trả có khả năng cao

### Huấn luyện

- Huấn luyện từng ván một
- Huấn luyện nhiều ván song song
- 128 bàn cờ mặc định trong chế độ song song
- Curriculum cho opening, middlegame và endgame
- Tiếp tục huấn luyện từ checkpoint hoặc bắt đầu mô hình mới
- Tự lưu checkpoint và chỉ số huấn luyện
- Kiểm tra mate-in-one, tránh chiếu hết, bắt quân tự do và chuyển đổi tàn cuộc
- Ghi log và vẽ reward, loss, entropy, win/draw rate và throughput
- AlphaZero có thể dùng native Rust MCTS thay cho MCTS Python

### Teacher và Coach với Stockfish

- Tạo dữ liệu tiền huấn luyện từ nước đi và đánh giá của Stockfish
- Tiền huấn luyện mạng AlphaZero hoặc DQN từ dữ liệu đã tạo
- Dùng đánh giá Stockfish làm reward-shaping trong quá trình huấn luyện

## Kiến trúc mạng hiện tại

Các giá trị mặc định được lấy trực tiếp từ `src/encoding.py` và `src/model.py`.

| Thành phần | Giá trị |
|---|---|
| Lịch sử trạng thái | 3 thế cờ |
| 12 mặt quân cờ cho mỗi lịch sử | 36 mặt |
| Mặt metadata | 8 mặt |
| Kích thước tensor đầu vào | `44 x 8 x 8` |
| Không gian hành động | 16.384 |
| Residual blocks mặc định | 4 |
| Channels mặc định | 32 |
| Kiến trúc | Policy/Value Residual CNN |
| Ván tối đa mặc định | 300 nửa nước |

Không gian hành động gồm:

- `64 x 64 = 4.096` cặp ô đi - ô đến
- 3 x `4.096` mặt phong hậu, xe, tượng
- 1 x `4.096` mặt phong mã

Mỗi phe được lưu dưới dạng checkpoint riêng với tên `white_<thuật toán>.pt` hoặc `black_<thuật toán>.pt`.

> Checkpoint chỉ tương thích khi metadata checkpoint và kiến trúc mạng khớp với mã nguồn hiện tại. Nếu checkpoint cũ không thể nạp, hãy bắt đầu một lần huấn luyện mới.

## Cấu trúc thư mục

```text
src/
├── main.py                     # Điểm khởi chạy Pygame
├── menu_scene.py               # Chọn chế độ và cấu hình mô hình
├── play_scene.py               # Chế độ chơi
├── match.py                    # Quản lý trận đấu
├── players.py                  # Human, random và model players
├── encoding.py                 # Tensor bàn cờ 44 x 8 x 8 và action mapping
├── model.py                    # Residual policy/value network
├── strategy.py                 # Opponent model và lookahead
├── curriculum.py               # Opening/middlegame/endgame
├── skill_eval.py               # Kiểm tra kỹ năng
├── checkpoints.py              # Lưu checkpoint và metrics
├── metrics.py                  # Thống kê huấn luyện
├── algorithms/                 # A2C, PPO, DQN và AlphaZero
├── algorithms-rs/              # Native Rust MCTS
├── normal_mode/                # Huấn luyện một ván
├── concurrent_mode/            # Huấn luyện song song
├── plot_logging/               # Ghi log và vẽ biểu đồ
├── stockfish/                  # Teacher và Coach với Stockfish
└── tests/                      # Unit test

assets/chess_pieces/            # Hình quân cờ PNG
checkpoints/                    # Checkpoint và metrics theo từng experiment
```

## Yêu cầu

- Python 3.10 trở lên
- Pygame
- PyTorch
- NumPy
- python-chess
- Matplotlib, pandas, seaborn
- Stockfish chỉ cần khi sử dụng teacher/coach
- Rust/Cargo và maturin chỉ cần khi build native MCTS

Tạo môi trường ảo và cài các thư viện Python:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install pygame torch numpy matplotlib pandas seaborn python-chess
```

Nếu máy có NVIDIA GPU, cài PyTorch với build CUDA phù hợp để tăng tốc huấn luyện. GPU là tùy chọn.

## Chạy ứng dụng

Luôn chạy từ thư mục gốc của repository:

```powershell
python src\main.py
```

Ứng dụng có thể chạy bằng CPU hoặc GPU; PyTorch sẽ tự chọn CUDA khi thiết bị tương thích khả dụng.

## Điều khiển

### Menu

1. Chọn một trong năm chế độ ở thanh nút phía trên.
2. Nhấp chuột trái vào các hàng để chuyển tùy chọn hoặc chọn checkpoint.
3. Nhấp chuột phải vào hàng mô hình để quay lại lựa chọn trước hoặc bỏ mô hình đã chọn.
4. Nhấn **BẮT ĐẦU** hoặc `Enter`.
5. Nhấn `Esc` để thoát.

Trong các chế độ chơi:

- Nhấp một quân cờ, sau đó nhấp ô được làm sáng để đi quân.

Trong các chế độ huấn luyện:

- `Space`: tạm dừng/tiếp tục
- `F`: bật/tắt chế độ huấn luyện nhanh
- `Up` / `Down`: điều chỉnh độ trễ
- `S`: lưu checkpoint
- `L`: tải checkpoint
- `R`: đặt lại thống kê
- `M`: bật/tắt panel metrics
- `Esc`: lưu checkpoint và quay lại menu

## Huấn luyện và checkpoint

Mở:

- `Huấn luyện: 1 ván`
- `Huấn luyện: 128 ván`

Sau đó chọn checkpoint ở các hàng **Tệp mô hình Trắng** hoặc **Tệp mô hình Đen**:

- Nhấp trái: chọn checkpoint `.pt`
- Nhấp phải: bỏ checkpoint và khởi tạo mô hình mới

Thuật toán tương ứng được đọc từ metadata checkpoint. Checkpoint mới được lưu tại:

```text
checkpoints/exp_YYYYMMDD_HHMMSS/
├── white_<algorithm>.pt
├── black_<algorithm>.pt
├── metrics.json
└── training_log.csv
```

Ứng dụng cũng tự lưu định kỳ và lưu checkpoint trước khi thoát màn hình huấn luyện bằng `Esc`.

## Xem kết quả huấn luyện

Vẽ biểu đồ từ log mới nhất:

```powershell
python src\plot_logging\plot_training.py
```

Hoặc chỉ định một log cụ thể:

```powershell
python src\plot_logging\plot_training.py checkpoints\exp_YYYYMMDD_HHMMSS\training_log.csv
```

`training_log.csv` lưu loss, reward, entropy, win/draw rate, repetition, độ dài ván, throughput, nguyên nhân hòa và timing từng giai đoạn. `metrics.json` lưu các thống kê tổng hợp của experiment.

## Native Rust MCTS cho AlphaZero

Từ thư mục `src/algorithms-rs`:

```powershell
pip install maturin
python -m pip install .
```

Sau đó bật native search:

```powershell
$env:CHESS_AI_RUST_SEARCH = "1"
python src\main.py
```

Backend tùy chọn:

| Biến môi trường | Hành vi |
|---|---|
| `CHESS_AI_RUST_SEARCH=1` | Rust xử lý luật cờ, mã hóa, cây tìm kiếm, chọn nhánh và backup |
| `CHESS_AI_RUST_MCTS=1` | Backend cây Rust cũ; chỉ PUCT/expand/backup chạy trong Rust |

Nếu biến được bật nhưng extension chưa được cài, chương trình sẽ báo lỗi rõ ràng thay vì im lặng chuyển sang Python.

Để chạy hoàn toàn bằng MCTS Python:

```powershell
$env:CHESS_AI_RUST_SEARCH = "0"
$env:CHESS_AI_RUST_MCTS = "0"
python src\main.py
```

Chi tiết backend và test parity nằm trong [src/algorithms-rs/README.md](src/algorithms-rs/README.md).

## Stockfish Teacher và Coach

Tải tệp thực thi native/UCI của Stockfish, sau đó cấu hình:

```powershell
$env:STOCKFISH_PATH = "C:\path\stockfish.exe"
```

Có thể bỏ biến này nếu `stockfish.exe` đã nằm trong `src\stockfish` hoặc `stockfish` đã nằm trong `PATH`. Ứng dụng không tự tải hoặc sao chép binary Stockfish.

Tạo dữ liệu và tiền huấn luyện từ thư mục gốc:

```powershell
python src\stockfish\pretrain_stockfish.py generate --out data\sf.npz --games 2500 --workers 10 --nodes 50000
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm AlphaZero --color white --epochs 6 --lr 1e-3
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm DQN --color black --epochs 6 --lr 1e-3
```

Sau khi tạo dữ liệu, kiểm tra ranh giới tập holdout:

```powershell
python src\stockfish\pretrain_stockfish.py inspect --data data\sf.npz --holdout 0.1 --seed 0
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm AlphaZero --color both --epochs 6 --holdout 0.1 --seed 0
```

Hoặc chạy huấn luyện với Stockfish làm nguồn reward:

```powershell
$env:CHESS_AI_STOCKFISH_COACH = "1"
$env:CHESS_AI_STOCKFISH_COACH_NODES = "2000"
python src\main.py
```

Chi tiết nằm trong [src/stockfish/STOCKFISH.md](src/stockfish/STOCKFISH.md).

## Kiểm tra mã nguồn

Chạy các test Python:

```powershell
cd src
python -m compileall -q .
python -m unittest discover -s tests -v
```

Kiểm tra native Rust nếu đã build extension:

```powershell
cd src\algorithms-rs
cargo test

cd ..
python -m unittest tests.test_rust_search
```
