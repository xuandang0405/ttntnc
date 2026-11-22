# Giáo viên và Huấn luyện viên Stockfish (Stockfish Teacher and Coach)

Cần một tệp thực thi Stockfish (tải tại https://stockfishchess.org/download/), đặt biến môi trường 
`$env:STOCKFISH_PATH = "<your path to Stockfish executable>` và copy tệp thực thi vào trong `/src/stockfish`.
Tệp thực thi phải là bản native và chạy được ở chế độ UCI. Ứng dụng không tự tải hoặc sao chép các tệp nhị phân Stockfish.

Chạy các lệnh của teacher từ thư mục gốc của repository:

```powershell
python src\stockfish\pretrain_stockfish.py generate --out data\sf.npz --games 2500 --workers 10 --nodes 50000
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm AlphaZero --color white --epochs 6 --lr 1e-3
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm DQN --color black --epochs 6 --lr 1e-3
```

Cũng có thể chạy sau khi chuyển vào thư mục `src` (`cd src`):

```powershell
cd src
python -m stockfish.pretrain_stockfish generate --out ..\data\sf.npz --games 800 --workers 10 --nodes 25000
```

## Teacher: Tiền huấn luyện (Pretrain) từ nhãn của Stockfish

Các checkpoint (`white_alphazero.pt`, ...) được tải vào ứng dụng giống như mọi mô hình khác.
Tham số `--init file.pt` dùng để tiếp tục huấn luyện từ một checkpoint có sẵn (thực hiện từng màu quân một).
Hãy dùng hàng chục nghìn thế cờ trở lên: trong một thử nghiệm nhỏ, đầu ra giá trị (value head) học khá nhanh, nhưng độ khớp nước đi (move agreement) đòi hỏi nhiều dữ liệu hơn đáng kể. DQN được huấn luyện bằng phương pháp hồi quy giá trị Q trên các nước đi ứng viên của Stockfish, kết hợp với một cận (bound) áp dụng cho tất cả các nước đi hợp lệ còn lại.
Mỗi lần lệnh `train` lưu checkpoint, thư mục chứa các tệp `.pt` cũng nhận một `stats.json` với thống kê từng epoch, cấu hình huấn luyện và thông tin tập dữ liệu.

## Coach: Đánh giá của Stockfish làm thế năng tạo thưởng (Reward-shaping potential)

Thiết lập `$env:CHESS_AI_STOCKFISH_COACH = "1"` trước khi khởi động ứng dụng. Cân bằng vật chất (material balance) sẽ được thay thế bằng điểm đánh giá của Stockfish (tính theo đơn vị tốt/pawns, giới hạn trong khoảng $\pm 10$) tại mỗi thời điểm ra quyết định của tác tử (agent); một ván cờ đã kết thúc có thế năng bằng $0$. Có thể tinh chỉnh độ sâu tìm kiếm bằng biến `CHESS_AI_STOCKFISH_COACH_NODES` (mặc định là `2000`).

## Vòng đời tiến trình (Process lifecycle)

Mỗi pool đều chờ các tác vụ phân tích đang chạy hoàn tất và đóng các tiến trình engine UCI khi gọi `close()` hoặc khi thoát khỏi context manager. Ứng dụng sẽ đóng coach dùng chung trong khối `finally` lúc Pygame thoát, đồng thời mỗi pool cũng đăng ký cơ chế dự phòng `atexit`. Các tiến trình worker của teacher tự đóng pool của chúng một cách tường minh.

Nếu tiến trình vẫn còn xuất hiện trong Task Manager sau một lần kết thúc bất thường, hãy đóng nó thủ công từ Task Manager; các lần thoát bình thường của ứng dụng, script và worker sẽ không để tiến trình chạy ngầm sót lại.

* **Cấu hình:** `stockfish_config.py`
* **Kiểm thử:** `python -m unittest tests.test_stockfish_pipeline` (sẽ tự động bỏ qua nếu không tìm thấy Stockfish).

## Tập kiểm thử không rò rỉ dữ liệu (Leakage-free held-out set)

Các thế cờ trong cùng một ván đấu thường gần như giống hệt nhau, do đó việc chia tách ngẫu nhiên từng thế cờ riêng lẻ sẽ khiến mô hình "kiểm thử" trên những dòng dữ liệu mà thực chất nó đã học qua. Hiện tại, tập dữ liệu kiểm thử (held-out set) được chọn **theo từng ván đấu** (`sklearn.model_selection.GroupShuffleSplit`), và bất kỳ thế cờ nào cũng xuất hiện trong một ván đấu khác (như các khai cuộc dùng chung, hoán vị nước đi) đều sẽ bị loại bỏ khỏi tập huấn luyện.

* Các bộ dữ liệu mới (định dạng 2) lưu trữ một mã định danh `game` cho mỗi dòng.
* Các tệp `.npz` cũ không cần phải tạo lại: ranh giới giữa các ván đấu được khôi phục chính xác từ các mặt phẳng lịch sử (history planes) trong mỗi trạng thái (đã được xác minh khớp với số lượng ván đấu thực tế).
* Lệnh `python stockfish\pretrain_stockfish.py inspect --data data\sf.npz` hiển thị số ván đấu tìm thấy và kiểm tra để đảm bảo không có ván đấu nào nằm ở cả hai tập. Lệnh `train` cũng in ra dòng thông tin phân chia tương tự và chấp nhận các tham số `--holdout 0.1` (tỷ lệ ván đấu dùng để kiểm thử; đặt `0` để tắt) và `--seed`.
* Yêu cầu cài đặt thư viện: `pip install scikit-learn`.
