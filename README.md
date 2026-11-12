# Reinforcement Learning Chess (Cờ Vua Học Tăng Cường)

Dự án xây dựng môi trường trò chơi Cờ Vua hoàn chỉnh bằng **Pygame**, tích hợp các thuật toán **Học Tăng Cường (Reinforcement Learning - RL)** để huấn luyện các tác nhân (AI agents) thông qua cơ chế tự đấu (self-play), đánh giá hiệu năng và trực quan hóa quá trình học.

---

## Tính năng nổi bật

* **Đa dạng thuật toán RL:** Hỗ trợ nhiều kiến trúc và thuật toán học tăng cường phổ biến:
  * **A2C** (`algo_a2c.py`) - Advantage Actor-Critic
  * **PPO** (`algo_ppo.py`) - Proximal Policy Optimization
  * **DQN** (`algo_dqn.py`) - Deep Q-Network
  * **AlphaZero** (`algo_alphazero.py`) - Kết hợp mạng nơ-ron và tìm kiếm cây Monte Carlo (MCTS)
* **5 Chế độ chơi & Huấn luyện linh hoạt:**
  1. **Người vs Người (Human vs Human):** Hai người chơi trên cùng một màn hình.
  2. **Người vs AI (Human vs AI):** Đối đầu trực tiếp với mô hình đã huấn luyện hoặc máy đánh ngẫu nhiên (Random mover).
  3. **AI vs AI:** Quan sát hai mô hình AI tự đấu với nhau.
  4. **Huấn luyện: 1 ván (Train: 1 game):** Tự đấu tuần tự từng ván một để quan sát trực tiếp nước đi và quá trình cập nhật trọng số.
  5. **Huấn luyện song song (Train: N games):** Chạy đồng thời nhiều bàn cờ (`concurrent_mode`) tận dụng tối đa sức mạnh của GPU/CPU để tăng tốc độ thu thập dữ liệu và huấn luyện.
* **Quản lý Checkpoint thông minh:**
  * Lưu và nạp lại trạng thái mô hình bất cứ lúc nào thông qua hộp thoại chọn tệp (`file_dialog.py`).
  * Tự động nhận diện thuật toán tương ứng từ tệp checkpoint đã lưu (`read_checkpoint_info`).
* **Theo dõi & Vẽ biểu đồ huấn luyện (`plot_logging/`):**
  * Tự động ghi log các chỉ số trong quá trình huấn luyện (`training_log.py`).
  * Trực quan hóa đường cong học tập (loss, reward, tỷ lệ thắng/thua/hòa) bằng biểu đồ (`plot_training.py`).
* **Tùy chỉnh phong cách chơi của AI:**
  * **Sampled (Đa dạng):** Chọn nước đi theo phân phối xác suất của policy (phù hợp cho khám phá và tạo sự đa dạng).
  * **Greedy (Tham lam):** Luôn chọn nước đi có xác suất/giá trị cao nhất (tối ưu sức mạnh khi thi đấu).

---

## Cấu trúc thư mục

```text
src/
├── main.py                     # Điểm khởi chạy chính của ứng dụng
├── config.py                   # Cấu hình chung (kích thước cửa sổ, màu sắc, FPS...)
├── game.py                     # Logic cốt lõi của bàn cờ và luật chơi cờ vua
├── encoding.py                 # Mã hóa trạng thái bàn cờ thành tensor đầu vào cho mạng nơ-ron
├── model.py                    # Kiến trúc mạng nơ-ron (Neural Network)
├── metrics.py                  # Tính toán các chỉ số đánh giá và phần thưởng (Reward)
│
├── algorithms.py               # Quản lý danh sách và khởi tạo các thuật toán
├── algo_config.py              # Siêu tham số (hyperparameters) cho từng thuật toán
├── algo_base.py                # Lớp cơ sở (Base class) cho các thuật toán RL
├── algo_a2c.py                 # Cài đặt thuật toán A2C
├── algo_ppo.py                 # Cài đặt thuật toán PPO
├── algo_dqn.py                 # Cài đặt thuật toán DQN
├── algo_alphazero.py           # Cài đặt thuật toán AlphaZero
├── checkpoints.py              # Xử lý lưu/tải trọng số và metadata của mô hình
│
├── players.py                  # Định nghĩa các loại người chơi (Human, Random, AI Model)
├── match.py                    # Quản lý luồng trận đấu giữa hai người chơi
├── scene.py                    # Lớp cơ sở quản lý các màn hình (Scene)
├── menu_scene.py               # Màn hình Menu chính chọn chế độ và cấu hình mô hình
├── play_scene.py               # Màn hình chơi cờ trực quan
├── train_scene.py              # Màn hình huấn luyện đơn luồng (1 ván)
├── renderer.py                 # Vẽ bàn cờ, quân cờ và thông tin trạng thái
├── ui.py                       # Các thành phần giao diện người dùng (Button...)
├── file_dialog.py              # Hộp thoại hệ thống để chọn tệp mô hình
│
├── concurrent_mode/            # Module huấn luyện đa bàn cờ song song
│   ├── concurrent_config.py    # Cấu hình số lượng bàn cờ chạy song song (NUM_GAMES)
│   ├── concurrent_game.py      # Quản lý trạng thái nhiều ván cờ cùng lúc
│   ├── concurrent_train_scene.py # Giao diện và vòng lặp huấn luyện song song
│   └── concurrent_renderer.py  # Hiển thị lưới nhiều bàn cờ thu nhỏ
│
└── plot_logging/               # Module ghi log và vẽ đồ thị
    ├── training_log.py         # Lưu lịch sử huấn luyện ra tệp log
    └── plot_training.py        # Vẽ biểu đồ phân tích quá trình huấn luyện
```

---

## Yêu cầu cài đặt

Đảm bảo máy tính của bạn đã cài đặt **Python 3.10+** và các thư viện cần thiết:

```bash
pip install pygame torch numpy matplotlib chess
```
*(Lưu ý: Nếu có GPU hỗ trợ CUDA, hãy cài đặt phiên bản `torch` tương ứng để tăng tốc độ huấn luyện ở chế độ song song).*

---

## Hướng dẫn sử dụng

### 1. Khởi chạy ứng dụng
Di chuyển vào thư mục `src` và chạy tệp `main.py`:

```bash
cd src
python main.py
```

### 2. Thao tác trên Menu chính
* **Chọn chế độ:** Nhấp chuột trái vào các nút chế độ ở hàng trên cùng (`Người vs Người`, `Người vs AI`, `AI vs AI`, `Huấn luyện: 1 ván`, `Huấn luyện: k ván`).
* **Cấu hình tham số:**
  * **Nhấp chuột trái** vào các hàng tùy chọn để chuyển sang lựa chọn tiếp theo hoặc mở hộp thoại chọn tệp mô hình (`.pt`).
  * **Nhấp chuột phải** vào hàng tùy chọn để quay lại lựa chọn trước đó hoặc xóa tệp mô hình đã chọn (đưa về *Random mover* hoặc *Huấn luyện từ đầu*).
* **Bắt đầu:** Nhấn nút **START** hoặc phím `Enter` để vào màn hình chơi/huấn luyện. Nhấn `ESC` để thoát.

### 3. Xem biểu đồ huấn luyện
Sau khi quá trình huấn luyện ghi lại dữ liệu log, bạn có thể chạy script vẽ biểu đồ để theo dõi sự tiến bộ của mô hình:

```bash
python plot_logging/plot_training.py
```

---

## Lưu ý khi huấn luyện
* Hệ thống **không tự động nạp mô hình cũ** khi khởi động để tránh ghi đè ngoài ý muốn. Nếu muốn huấn luyện tiếp (fine-tune) từ một checkpoint có sẵn, hãy chọn tệp tương ứng ở mục `White model file` / `Black model file` trước khi nhấn `START`.
* Chế độ **Train: N games** được tối ưu hóa cho việc thu thập kinh nghiệm nhanh trên nhiều môi trường đồng thời, rất phù hợp khi huấn luyện các thuật toán như PPO hoặc A2C từ con số 0.