# Broadcast Pre-training Pipeline

## 1. Giới thiệu

**Broadcast Pre-training Pipeline** là quy trình tiền huấn luyện mô hình `PolicyValueNet` trên dữ liệu các ván cờ broadcast từ Lichess, bao gồm các giải đấu chính thức được tổ chức trên bàn cờ thật.

Pipeline hỗ trợ tùy chọn sử dụng Stockfish hoặc dữ liệu đánh giá có sẵn từ Lichess để bổ sung nhãn cho các vị trí cờ. Sau khi hoàn thành tiền huấn luyện, mô hình được xuất thành các checkpoint tương thích với những agent hiện có, cho phép tiếp tục huấn luyện bằng các thuật toán khác nhau.

### 1.1. Mục tiêu

- Khai thác dữ liệu ván cờ broadcast có chất lượng cao.
- Lọc và chuẩn hóa dữ liệu trước khi huấn luyện.
- Bổ sung nhãn Policy và Value bằng Stockfish hoặc Lichess Evaluation Database.
- Huấn luyện chung Policy và Value trong một mô hình `PolicyValueNet`.
- Xuất checkpoint tương thích với các thuật toán AlphaZero, DQN, PPO và A2C.

### 1.2. Kiến trúc tổng quan

Pipeline gồm năm giai đoạn chính:

```text
       DOWNLOAD
           |
           v
        PREPARE
           |
           v
    +------+------+
    |             |
    v             v
 STOCKFISH      EVAL JOIN
  LABEL
    |             |
    +------+------+
           |
           v
         TRAIN
           |
           v
         EXPORT
           |
           v
  ALGORITHM-SPECIFIC
     CHECKPOINTS
```

Trong đó, `STOCKFISH LABEL` và `EVAL JOIN` là hai phương thức bổ sung nhãn có thể lựa chọn thay thế cho nhau.

---

## 2. Yêu cầu hệ thống

### 2.1. Thư viện Python

Cài đặt các thư viện cần thiết:

```bash
pip install python-chess zstandard numpy torch
```

### 2.2. Stockfish

Nếu sử dụng phương thức `stockfish_label`, cần cài đặt Stockfish và xác định đường dẫn đến tệp thực thi.

Ví dụ:

```text
/path/to/stockfish
```

Nếu sử dụng `eval_join`, cần chuẩn bị tệp `lichess_db_eval.jsonl.zst` tương ứng.

---

## 3. Quy trình thực hiện

### Bước 1. Download dữ liệu Broadcast

Tải dữ liệu broadcast từ Lichess theo tháng.

```bash
python -m pretrain.download \
    --months 2026-01 \
    --out data/raw
```

**Thông tin dữ liệu:**

- Dữ liệu tháng 01/2026.
- Dung lượng tham khảo: khoảng 16 MB.
- Số lượng tham khảo: khoảng 25.000 ván cờ.

Có thể tải dữ liệu trong một khoảng thời gian dài hơn:

```bash
python -m pretrain.download \
    --from 2020-01 \
    --to 2026-10 \
    --out data/raw
```

**Đầu ra:**

```text
data/
└── raw/
    └── lichess_db_broadcast_2026-01.pgn.zst
```

### Bước 2. Prepare và chia dữ liệu thành Shards

Lọc, chuẩn hóa và chia dữ liệu PGN thành các shard phục vụ quá trình gán nhãn và huấn luyện.

```bash
python -m pretrain.prepare \
    data/raw/lichess_db_broadcast_2026-01.pgn.zst \
    --out data/shards \
    --min-elo 2000
```

**Các tiêu chí lọc:**

- Cả hai kỳ thủ đều có Elo tối thiểu 2000.
- Loại bỏ các ván chưa hoàn thành.
- Loại bỏ các ván có nước đi không hợp lệ.
- Loại bỏ các biến thể cờ không chuẩn.
- Loại bỏ các ván hòa quá ngắn.
- Loại bỏ các ván bị trùng lặp.

Sau khi lọc, dữ liệu được chia thành các shard để thuận tiện cho việc xử lý song song.

**Đầu ra:**

```text
data/
└── shards/
    ├── ...
    └── ...
```

### Bước 3. Gán nhãn dữ liệu

Có hai phương thức gán nhãn. Chọn một trong hai phương thức tùy theo nguồn tài nguyên và dữ liệu hiện có.

#### Phương thức A: Stockfish Label

Sử dụng Stockfish để phân tích các vị trí cờ và tạo nhãn đánh giá.

```bash
python -m pretrain.stockfish_label \
    data/shards \
    --out data/labeled \
    --engine /path/to/stockfish \
    --nodes 5000 \
    --every 4 \
    --workers 8
```

**Tham số:**

| Tham số | Ý nghĩa |
|---|---|
| `--engine` | Đường dẫn đến Stockfish executable |
| `--nodes` | Ngân sách tìm kiếm cho mỗi vị trí |
| `--every` | Khoảng cách lấy mẫu giữa các vị trí |
| `--workers` | Số worker xử lý song song |

Phương thức này bổ sung thông tin đánh giá của Stockfish, bao gồm giá trị vị trí và nước đi tốt nhất theo kết quả phân tích.

#### Phương thức B: Lichess Evaluation Join

Thay vì tự chạy Stockfish, có thể tra cứu nhãn từ Lichess Evaluation Database.

```bash
python -m pretrain.eval_join \
    data/shards \
    --evals data/lichess_db_eval.jsonl.zst \
    --out data/labeled
```

**Đặc điểm:**

- Sử dụng dữ liệu đánh giá có sẵn từ Lichess.
- Không cần chạy Stockfish để phân tích lại toàn bộ vị trí.
- Tệp dữ liệu tham khảo có khoảng 417 triệu vị trí, với độ sâu tìm kiếm khoảng 30+.

Phương thức này phù hợp khi muốn tiết kiệm tài nguyên tính toán và tận dụng dữ liệu đánh giá có sẵn.

**Đầu ra chung của bước 3:**

```text
data/
└── labeled/
    └── ...
```

### Bước 4. Huấn luyện PolicyValueNet

Sử dụng dữ liệu đã gán nhãn để tiền huấn luyện mô hình `PolicyValueNet`.

```bash
python -m pretrain.train \
    --shards data/labeled \
    --out runs/pre1 \
    --epochs 4 \
    --batch 512 \
    --workers 6 \
    --policy-min-elo 2300 \
    --sf-value-weight 0.7 \
    --w-sf 0.3
```

#### 4.1. Cấu hình huấn luyện

| Tham số | Giá trị | Mô tả |
|---|---:|---|
| `--shards` | `data/labeled` | Thư mục chứa dữ liệu đã gán nhãn |
| `--out` | `runs/pre1` | Thư mục lưu kết quả huấn luyện |
| `--epochs` | 4 | Số epoch huấn luyện |
| `--batch` | 512 | Batch size |
| `--workers` | 6 | Số worker nạp dữ liệu |
| `--policy-min-elo` | 2300 | Ngưỡng Elo tối thiểu cho dữ liệu Policy |
| `--sf-value-weight` | 0.7 | Trọng số thành phần SF Value |
| `--w-sf` | 0.3 | Trọng số giám sát từ Stockfish |

#### 4.2. Mục tiêu huấn luyện

Mô hình được huấn luyện đồng thời trên hai thành phần:

**Policy Head**

Học cách lựa chọn nước đi từ dữ liệu ván cờ của con người và các nhãn nước đi từ Stockfish.

**Value Head**

Học cách đánh giá vị trí thông qua kết quả thực tế của ván đấu kết hợp với đánh giá từ Stockfish.

#### 4.3. Công thức Value Target

Với các vị trí có nhãn đánh giá Stockfish, Value Target được tính theo công thức:

\[
v=(1-w)\cdot result+
w\cdot\tanh\left(\frac{cp}{543}\right)
\]

Trong đó:

- \(v\): Giá trị mục tiêu của vị trí.
- \(result\): Kết quả ván đấu theo góc nhìn của vị trí đang xét.
- \(cp\): Điểm đánh giá centipawn.
- \(w\): Trọng số kết hợp giữa kết quả thực tế và đánh giá Stockfish.
- \(543\): Hằng số tỷ lệ được sử dụng trong phép biến đổi tanh.

Đối với các vị trí chưa có nhãn Stockfish, mô hình sử dụng trực tiếp kết quả ván đấu:

\[
v=result
\]

### Bước 5. Export checkpoint

Sau khi hoàn thành huấn luyện, xuất checkpoint của `PolicyValueNet` sang định dạng tương thích với các agent hiện có.

```bash
python -m pretrain.export \
    --ckpt runs/pre1/best.pt \
    --out models \
    --algorithms alphazero dqn ppo a2c
```

**Tham số:**

| Tham số | Ý nghĩa |
|---|---|
| `--ckpt` | Checkpoint tốt nhất sau pretrain |
| `--out` | Thư mục lưu checkpoint đã xuất |
| `--algorithms` | Danh sách thuật toán cần xuất checkpoint |

**Các thuật toán hỗ trợ:**

- AlphaZero
- DQN
- PPO
- A2C

Kết quả là các checkpoint được xuất riêng theo từng cặp thuật toán và màu quân, phục vụ việc khởi tạo hoặc tiếp tục huấn luyện các agent tương ứng.

---

## 4. Ghi chú thiết kế kỹ thuật

### 4.1. Phân chia Train/Validation theo sự kiện

Dữ liệu được phân chia theo **Event (giải đấu)** thay vì chia ngẫu nhiên từng vị trí cờ.

Mục đích:

- Tránh các ván thuộc cùng một giải đấu xuất hiện đồng thời trong Train và Validation.
- Hạn chế rò rỉ dữ liệu giữa hai tập.
- Đánh giá khả năng tổng quát hóa trên các sự kiện chưa xuất hiện trong quá trình huấn luyện.

### 4.2. Policy Loss trên các nước đi hợp lệ

Policy Loss chỉ được tính trên các logit tương ứng với những nước đi hợp lệ tại vị trí hiện tại (**Legal Logits**).

Không áp dụng softmax trên toàn bộ không gian 20.480 hành động.

Cách xử lý này giúp mô hình tập trung phân phối xác suất trên tập nước đi hợp lệ, tương tự luồng xử lý Policy trong kiến trúc AlphaZero.

### 4.3. Tái tạo trạng thái bàn cờ

Pipeline không lưu trữ trước toàn bộ tensor trạng thái bàn cờ.

Thay vào đó, các ván cờ được phát lại tuần tự trên một đối tượng bàn cờ liên tục. Hàm `encode` được gọi trực tiếp trong quá trình phát lại để tạo biểu diễn trạng thái.

Cách tiếp cận này cho phép duy trì thông tin lịch sử nước đi thực tế, phục vụ các kiến trúc sử dụng lịch sử trạng thái.

### 4.4. Xác định góc nhìn Centipawn (POV)

Góc nhìn của giá trị `cp` trong Lichess Evaluation Database cần được xác định chính xác trước khi sử dụng làm Value Target.

Do tài liệu nguồn không mô tả rõ quy ước POV trong dữ liệu đang sử dụng, công cụ `eval_join` thực hiện kiểm tra tương quan giữa giá trị đánh giá và tương quan lực lượng/chất cờ, sau đó in ra kết luận về góc nhìn được xác định.

Việc thống nhất POV là cần thiết để tránh gán nhãn Value sai chiều cho quân Trắng và quân Đen.

---

## 5. Cấu trúc thư mục đầu ra

Cấu trúc tổng quát của pipeline:

```text
project/
├── pretrain/
│   ├── download.py
│   ├── prepare.py
│   ├── stockfish_label.py
│   ├── eval_join.py
│   ├── train.py
│   └── export.py
│
├── data/
│   ├── raw/
│   ├── shards/
│   └── labeled/
│
├── runs/
│   └── pre1/
│       └── best.pt
│
└── models/
    ├── ...
    └── ...
```

---

## 6. Quy trình sử dụng checkpoint

Checkpoint sau khi export có thể được sử dụng để khởi tạo các agent tương ứng.

Luồng tổng thể:

```text
Lichess Broadcast Games
          |
          v
    Data Preparation
          |
          v
   Stockfish / Lichess
     Evaluation Labels
          |
          v
    PolicyValueNet
      Pre-training
          |
          v
    Checkpoint Export
          |
    +-----+-----+-----+
    |     |     |     |
    v     v     v     v
   AZ    DQN   PPO   A2C
    |     |     |     |
    +-----+-----+-----+
          |
          v
    Further Training
```

Quá trình tiền huấn luyện cung cấp kiến thức ban đầu về lựa chọn nước đi và đánh giá vị trí. Các agent sau đó có thể tiếp tục được huấn luyện bằng thuật toán tương ứng để phát triển năng lực chơi cờ.

---

## 7. Tóm tắt

Broadcast Pre-training Pipeline cung cấp một quy trình thống nhất để chuyển đổi dữ liệu ván cờ broadcast thành dữ liệu huấn luyện cho mô hình `PolicyValueNet`.

Pipeline bao gồm:

1. **Download:** Thu thập dữ liệu broadcast từ Lichess.
2. **Prepare:** Lọc, chuẩn hóa và chia dữ liệu thành các shard.
3. **Label:** Bổ sung nhãn bằng Stockfish hoặc Lichess Evaluation Database.
4. **Train:** Tiền huấn luyện PolicyValueNet với Policy và Value supervision.
5. **Export:** Xuất checkpoint tương thích với AlphaZero, DQN, PPO và A2C.

Thiết kế này cho phép tận dụng dữ liệu thi đấu thực tế kết hợp với thông tin đánh giá từ engine, tạo nền tảng khởi tạo cho các mô hình chơi cờ trước khi bước vào các giai đoạn huấn luyện chuyên sâu hơn.