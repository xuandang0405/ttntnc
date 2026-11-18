# Huấn luyện các mô hình AI Cờ Vua

Bộ mã có hai kiến trúc MLP/CNN dùng chung đầu policy và value, học có giám sát,
tác tử dùng mạng, Alpha-beta dùng value học được và TD(0) tự chơi.
Phần huấn luyện nằm trong `model/learning/`; game hiện tại vẫn dùng các đối thủ cũ.

## 1. Chuẩn bị môi trường

Mở PowerShell tại thư mục dự án. Môi trường AI riêng dùng Python 3.12 và PyTorch
2.10.0. Không cần kích hoạt môi trường nếu gọi đúng tệp Python bên dưới.

```powershell
Set-Location -LiteralPath 'D:\QUAN TRỌNG\TTNTNC\Cờ vua'
& .\CAI_DAT_AI.ps1 -Device cpu
$py = '.\.venv-ai\Scripts\python.exe'
```

Máy hiện tại đã cài bản CPU và chạy kiểm thử. Để chuyển sang NVIDIA GPU:

```powershell
& .\CAI_DAT_AI.ps1 -Device cuda
& $py -X utf8 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

Script chọn wheel CUDA 12.8 từ kho chính thức PyTorch. Chỉ dùng `--device cuda`
khi phép kiểm tra trả `True`. Nếu PowerShell chặn script, có thể gọi:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\CAI_DAT_AI.ps1 -Device cpu
```

Đây là chính sách cho tiến trình chạy script đó, không đổi cấu hình toàn máy.
Nếu máy khác chưa có Python 3.12, cài Python hoặc truyền đường dẫn qua
`-PythonExe 'C:\duong-dan\python.exe'`.

## 2. Sinh dữ liệu nhỏ trước

```powershell
& $py -X utf8 -m model.learning.data --out data/training/pilot.jsonl --games 100 --max-plies 80 --teacher alphabeta --teacher-ms 30 --seed 42
```

Mỗi dòng có FEN, mã ván, thông tin lặp lại, nước đề xuất của teacher, nước thực
hiện và value. Teacher mặc định là Alpha-beta độ sâu tối đa 2 của dự án.
Khám phá mặc định 15% làm đa dạng vị trí; nước khám phá không thay nhãn teacher.
Value là `tanh(cp / 600)` theo bên đến lượt, không phải xác suất thắng.

Nếu có engine Stockfish, chọn nguồn nhãn khác bằng đường dẫn thực tế:

```powershell
& $py -X utf8 -m model.learning.data --out data/training/stockfish.jsonl --games 100 --teacher stockfish --engine-path 'C:\engines\stockfish.exe' --teacher-ms 100 --seed 42
```

Không ghi đè tệp dữ liệu đã tồn tại. Thay tên đầu ra khi sinh đợt mới.
Sidecar `.jsonl.meta.json` lưu tham số, SHA256 và kết thúc từng ván; ván bị dừng
ở trần ply có `result: null`, `termination: truncated`.

## 3. Train MLP và CNN trên cùng dữ liệu

```powershell
& $py -X utf8 -m model.learning.train --data data/training/pilot.jsonl --out experiments/mlp-01 --model mlp --epochs 10 --batch-size 64 --device cpu --seed 42
& $py -X utf8 -m model.learning.train --data data/training/pilot.jsonl --out experiments/cnn-01 --model cnn --epochs 10 --batch-size 64 --device cpu --seed 42
```

Đổi `--device cpu` thành `--device cuda` sau khi cài và kiểm tra CUDA.
Dữ liệu chia theo mã ván khoảng 70/15/15; vị trí trùng giữa các tập được lọc.
`run.json` lưu danh sách ván và số mẫu bị loại. Tập kiểm tra được đo sau khi
chọn checkpoint bằng validation, không dùng để chọn epoch.

Đầu ra của mỗi lần train:

| Tệp | Nội dung |
|---|---|
| `best.pt` | Mô hình có validation loss nhỏ nhất |
| `last.pt` | Mô hình cuối đợt học |
| `metrics.csv` | Loss, policy accuracy theo epoch |
| `run.json` | Tham số, split, SHA256 dữ liệu và phiên bản |
| `test.json` | Kết quả trên tập kiểm tra của checkpoint đã chọn |

Mỗi lần chạy dùng một thư mục đầu ra mới. Loss giảm hoặc accuracy cao hơn chưa
chứng minh mô hình đánh mạnh hơn; cần đấu thử.

## 4. Nạp mô hình để chọn một nước

```powershell
& $py -X utf8 -m model.learning.predict --checkpoint experiments/cnn-01/best.pt --agent policy --device cpu
& $py -X utf8 -m model.learning.predict --checkpoint experiments/cnn-01/best.pt --agent value --device cpu
& $py -X utf8 -m model.learning.predict --checkpoint experiments/cnn-01/best.pt --agent search --depth 2 --time-ms 100 --device cpu
```

Có thể truyền `--fen '...'` để thử một thế cờ cụ thể.

- `policy`: chọn logit lớn nhất trong tập nước hợp lệ.
- `value`: xét các trạng thái sau nước đi, đổi dấu value của phía đối thủ.
- `search`: Alpha-beta, dùng mạng value để đánh giá lá; cắt tỉa và thời gian
  được quản lý trong `LearnedAlphaBetaAgent`.

`value` trong telemetry dùng thang [-1,1], không phải centipawn. Thời hạn là
kiểm tra hợp tác; một lượt suy luận đang chạy có thể làm vượt hạn và sẽ được ghi.

## 5. Fine-tune value bằng TD(0)

```powershell
& $py -X utf8 -m model.learning.selfplay --init experiments/cnn-01/best.pt --out experiments/td-01 --episodes 100 --max-plies 200 --device cpu --seed 42
```

Target của nước chưa kết thúc là `-gamma * V_target(s_next)` vì lượt đã chuyển
sang đối thủ. Nước kết thúc dùng +1/0/-1 theo phía vừa đi. Ván cắt ở trần ply
vẫn bootstrap giá trị, không bị coi là hòa. Target network được đồng bộ theo
`--target-sync` (mặc định 32 bước).

Checkpoint đầu ra là `td-final.pt`; nó chưa được chọn bằng đấu kiểm tra nên
không được gọi là mô hình mạnh nhất. TD chỉ tối ưu value và phần thân; policy
cũ có thể thay đổi do dùng chung thân, vì vậy checkpoint đánh dấu
`policy_trained=False` và được dùng ở chế độ `value` hoặc `search`:

```powershell
& $py -X utf8 -m model.learning.predict --checkpoint experiments/td-01/td-final.pt --agent value --device cpu
```

Đây là TD(0) học V(s), không phải DQN, PPO hay AlphaZero.

## 6. Đấu thử đổi màu

```powershell
& $py -X utf8 -m model.learning.arena --checkpoint experiments/cnn-01/best.pt --agent search --opponent random --games 20 --out experiments/arena-cnn-random --device cpu
& $py -X utf8 -m model.learning.arena --checkpoint experiments/cnn-01/best.pt --agent search --opponent alphabeta --games 20 --out experiments/arena-cnn-ab --device cpu
```

Mỗi cặp ván dùng cùng mở đầu và đổi màu. Tệp `summary.json` tách thắng, hòa,
thua và `truncated`. `completed_score` chỉ tính trên ván kết thúc đúng luật;
nếu tất cả bị cắt thì bằng `null`. Tỷ lệ cắt cao làm kết quả chưa đủ đại diện,
cần tăng giới hạn ply hoặc xem thêm PGN. Nhật ký từng ván ở `games.jsonl` và
`games.pgn`. Chưa có bootstrap/CI hoặc hệ Elo trong bộ chạy này.

## 7. Dùng từ backend

Hai tác tử mới tuân theo giao diện `select_action(env, time_limit_ms)`:

```python
from model.learning.agents import LearnedAlphaBetaAgent

# Khởi tạo một lần khi worker bắt đầu, không nạp file trong mỗi request.
agent = LearnedAlphaBetaAgent(
    'experiments/cnn-01/best.pt', depth=2, device='cpu'
)
uci, telemetry = agent.select_action(env, time_limit_ms=100)
env.step(uci)
```

Điểm thay đối thủ hiện tại nằm trong `GameService.play` ở
`backend/api/games/service.py`. Cần bổ sung lựa chọn checkpoint vào cấu hình ván và
đưa suy luận vào worker trước khi dùng trong luồng web. Bộ mã này cung cấp
giao diện và ví dụ tích hợp; các nút độ khó trong game chưa được đổi sang mô
hình mới. Không đưa trọng số chạy thử ít dữ liệu vào mức Ác mộng để mặc định
rằng nó mạnh hơn Alpha-beta.

## 8. Kiểm thử và kết quả chạy kỹ thuật

```powershell
& $py -X utf8 -m pytest tests -q
```

Đã kiểm tra trên CPU, Python 3.12.14, PyTorch 2.10.0+cpu:

- 31 kiểm thử đạt, gồm 10 kiểm thử mới cho phần học máy.
- Sinh 276 mẫu từ 16 ván nhỏ; sau lọc có 194 train, 33 validation, 45 test.
- Cả MLP và CNN chạy 2 epoch, lưu/nạp checkpoint và chọn nước hợp lệ.
- TD chạy 2 tập × 8 ply, thực hiện 16 cập nhật; hai tập đều ghi `truncated`.
- Đấu kiểm tra 2 ván ở trần 20 ply ghi 2 `truncated`, Score để `null`.

Các tệp chạy thử nằm trong `test-results/ai-models-smoke/`. Đây là bằng chứng
luồng mã chạy được, chưa phải bằng chứng về sức mạnh chơi. CUDA và nhánh
Stockfish chưa được chạy kiểm tra trên máy trong đợt này.

## Các tệp nên đọc theo thứ tự

1. `model/learning/encoding.py`: đặc trưng và mã hóa nước đi, gồm phong cấp.
2. `model/learning/models.py`: MLP/CNN, hai đầu ra, mask và checkpoint.
3. `model/learning/data.py`: sinh nhãn và chia tập theo ván.
4. `model/learning/train.py`: vòng lặp supervised và đánh giá.
5. `model/learning/agents.py`: policy, value và Alpha-beta dùng mạng.
6. `model/learning/selfplay.py`: TD(0) và target network.
7. `model/learning/arena.py`: đấu đổi màu và thống kê.

Tài liệu chính thức: [PyTorch](https://pytorch.org/get-started/locally/),
[lưu và nạp mô hình](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
[python-chess engine](https://python-chess.readthedocs.io/en/latest/engine.html).
