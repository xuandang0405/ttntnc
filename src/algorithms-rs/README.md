# algorithms-rs

Tìm kiếm AlphaZero native cho ứng dụng cờ vua.

| Backend | Biến môi trường | Phần chạy bằng Rust |
|---|---|---|
| Python MCTS (mặc định) | – | không có |
| Cây Rust cũ (`MctsTree`) | `CHESS_AI_RUST_MCTS=1` | chỉ phần tính toán PUCT / expand / backup |
| **Tìm kiếm native (`BatchSearch`)** | `CHESS_AI_RUST_SEARCH=1` | luật cờ, phát hiện kết thúc/hòa, mã hóa đầu vào, cây tìm kiếm, chọn nhánh, backup, nhiễu Dirichlet |

Backend cũ không nhanh hơn Python: bước backup chỉ chiếm ~0.1% thời gian của MCTS, trong khi chi phí thực sự (`copy`/`push`/`outcome(claim_draw=True)` của python-chess và việc phát lại lịch sử nước đi trong `encode`) vẫn nằm ở Python, cộng thêm một lần gọi qua PyO3 ở mỗi tầng của cây. `BatchSearch` chuyển toàn bộ các phần đó sang Rust và chỉ vượt qua ranh giới Python/Rust đúng một lần cho mỗi vòng mô phỏng.

## Build và test

```bash
pip install maturin
python -m pip install .          # chạy từ src/algorithms-rs; build ở chế độ release
cargo test                        # unit test phía Rust (cây cũ)
python -m unittest tests.test_rust_search   # test đối chiếu độ tương đồng với python-chess / Python MCTS
```

Bật tính năng bằng `CHESS_AI_RUST_SEARCH=1` (PowerShell: `$env:CHESS_AI_RUST_SEARCH = "1"`).
Nếu đã bật nhưng chưa cài đặt, chương trình sẽ báo lỗi rõ ràng ngay lúc khởi động.
Đặt `RAYON_NUM_THREADS` để giới hạn số luồng tìm kiếm (mặc định: toàn bộ số nhân).

## Giao thức (Protocol)

```python
search = algorithms_rs.BatchSearch(root_fens, move_lists, sims, c_puct,
                                   noise_frac, dirichlet_alpha, seed)
while (batch := search.next_batch()) is not None:
    states, legal = batch            # (N,104,8,8) float32, (N,L) int32 đệm bằng -1
    priors, values = evaluate(states, legal)   # softmax trên các nước hợp lệ, (N,L) / (N,)
    search.submit(priors, values)
results = search.results()           # theo từng ván: (chỉ số nước đi, số lượt thăm, nước đi UCI)
```

Mỗi ván được truyền vào dưới dạng FEN gốc kèm danh sách các nước đi UCI đã chơi kể từ đó, để Rust có thể dựng lại lịch sử 8 thế cờ và số lần lặp lại y hệt như `encoding.py`. Các thế cờ gốc đã kết thúc sẽ trả về kết quả rỗng.

`algorithms/algo_alphazero.py::_evaluate_encoded` chỉ thu thập các logit của nước đi hợp lệ trước khi chạy softmax, nhờ đó loại bỏ luôn phép masked softmax rộng 20480 phần tử.

## Hành vi so với bản tìm kiếm Python

Hoàn toàn giống nhau (đã kiểm chứng trên ~49k thế cờ và 240 lượt tìm kiếm, xem `tests/test_rust_search.py`):
các mặt phẳng đầu vào (input planes), chỉ số hành động của nước đi hợp lệ, luật hòa/kết thúc bao gồm hòa do lặp lại ba lần và luật 50 nước có thể yêu cầu, công thức PUCT và quy ước dấu, cũng như số lượt thăm.

Khác biệt có chủ đích:

- Các nút con được sắp xếp theo thứ tự sinh nước của `shakmaty` chứ không phải `python-chess`. Điều này chỉ làm thay đổi nước đi được chọn khi có hai điểm PUCT *bằng nhau tuyệt đối* (ví dụ khi prior đồng đều).
- Nhiễu Dirichlet được sinh trong Rust từ seed lấy từ RNG của NumPy cho mỗi lần gọi.
- Các nút lá kết thúc sẽ cache giá trị thay vì kiểm tra lại điều kiện kết thúc.
- Các khóa thống kê hiệu năng: bước chọn nhánh + mã hóa được gộp vào `mcts_select_traverse`, bước mở rộng + backup được gộp vào `mcts_expand`; `mcts_position_encode` và `mcts_backup` đều trả về `0`.

## Cấu trúc thư mục

- `src/search/encode.rs` – chuyển dịch từ `encoding.py` (các mặt phẳng, chỉ số hành động)
- `src/search/rules.rs` – chuyển dịch từ `Board.outcome(claim_draw=True)`
- `src/search/tree.rs` – cây tìm kiếm theo từng ván và giao thức nút lá
- `src/lib.rs` – các binding PyO3 (`BatchSearch`, `MctsTree`, `encode_position`, `terminal_status`)
- `src/alphazero/` – các thành phần gốc của arena, giữ nguyên