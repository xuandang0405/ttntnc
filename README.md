# Cờ Vua

Game cờ vua chạy cục bộ với **FastAPI + React/TypeScript + Vite**. Node.js dùng để phát triển và build frontend; khi chơi bản build, FastAPI phục vụ cả giao diện và API tại một địa chỉ.

## Mở game

Máy hiện tại đã có thư viện và bản build. Mở **`CHAY_GAME.bat`**, trình duyệt sẽ mở khi máy chủ sẵn sàng:

**http://127.0.0.1:8000** · API tương tác: **http://127.0.0.1:8000/docs**

Giữ cửa sổ máy chủ mở khi chơi. Nhấn `Ctrl+C` trong cửa sổ đó để dừng. Nếu game đã chạy tại cổng 8000, launcher mở lại game hiện có. Có thể đổi cổng:

```powershell
.\.venv\Scripts\python.exe run_game.py --port 8001
```

Khi cài trên máy khác, cài Python và Node.js, thêm vào PATH, rồi chạy **`CAI_DAT.bat`** một lần. Môi trường đã kiểm chứng: Windows, Python 3.14.7, Node.js 24.21.0. Vite 7 cần Node.js 20.19+ hoặc 22.12+; nên dùng Node.js LTS hiện hành. Python 3.12 trở lên được khuyến nghị.

## Đã có

- Bàn cờ 8 × 8, tọa độ, chọn ô/kéo thả, chấm nước đi hợp lệ, tô nước vừa đi và vua đang bị chiếu.
- Sprite quân trắng/đen; animation di chuyển, ra đòn, va chạm và quân bị ăn tan biến. Nền đại sảnh hoàng gia, ba màu bàn cờ, giao diện co giãn theo màn hình.
- Idle phát liên tục, có nhịp thở và vòng sáng. Trong Cài đặt → Animation quân cờ, chọn **Đầy đủ**, **Theo thiết bị** hoặc **Tắt**; chế độ Đầy đủ bật hiệu ứng trong game kể cả khi thiết bị đang yêu cầu giảm chuyển động.
- Menu toàn màn hình, nút kiểu game, chọn **Dễ / Bình thường / Khó / Ác mộng**, hoặc **Hai người** cùng thiết bị. Chọn cầm Trắng/Đen khi đấu máy; máy cầm Trắng tự đi trước.
- Ván mới, hoàn tác cả lượt, tạm dừng/tiếp tục, xoay bàn, toàn màn hình, đầu hàng, kết thúc hòa, bật/tắt âm thanh và chỉnh âm lượng.
- Lịch sử SAN, quân đã mất, thời lượng ván, xuất PGN, nhập/sao chép FEN. Có thế cờ mẫu để thử ăn quân, nhập thành và phong cấp.
- Phong cấp thành Hậu/Xe/Tượng/Mã, nhập thành, bắt tốt qua đường, chiếu bí, hết nước đi và thiếu lực lượng đều do backend kiểm tra.
- Lưu phiên vào SQLite; tải lại trang tiếp tục ván hiện tại. Cài đặt giao diện được lưu trong trình duyệt.

Phím tắt: `F` xoay bàn · `H` gợi ý · `Space` tạm dừng · `Ctrl+Z` hoàn tác · `Esc` bỏ chọn/đóng hộp thoại. Phím mũi tên di chuyển tiêu điểm giữa các ô.

## Phạm vi AI và luật kết thúc

| Mức | Đối thủ |
|---|---|
| Dễ | Random chọn đều nước hợp lệ |
| Bình thường | Chọn nước theo đánh giá vật chất/vị trí, tìm 1 ply |
| Khó | Alpha-beta, tìm 2 ply, ngân sách tối đa 600 ms |
| Ác mộng | Alpha-beta tăng dần đến 4 ply, tối đa 1400 ms |

Độ khó được lưu trong ván và backend xác nhận khi máy đi. Khi hết ngân sách tìm kiếm, dùng kết quả của độ sâu hoàn tất gần nhất. Dễ vẫn dùng `random.Random(seed).choice(legal_actions)` đúng yêu cầu ban đầu. API hỗ trợ seed; UI chính ưu tiên ít chữ và thao tác chơi.

Gợi ý là một nước ngẫu nhiên hợp lệ, không phải nước tối ưu. Lịch sử, PGN và công cụ FEN nằm trong cửa sổ Lịch sử; các thống kê AI được bỏ khỏi màn chơi. Bản đồ ô trong công cụ phụ chỉ đếm các đích có thể đến trong lượt hiện tại.

Theo cấu hình đề tài, mỗi ván tối đa **80 ply = 40 lượt đủ Trắng/Đen**. Nếu chưa kết thúc theo luật cờ, chênh từ 300 centipawn (3 điểm quân) phân thắng thua; nhỏ hơn thì hòa. Điểm: Tốt 100, Mã 320, Tượng 330, Xe 500, Hậu 900; không cộng Vua vào lực lượng. Backend tự nhận hòa khi có thể yêu cầu theo luật lặp lại/50 nước. Đồng hồ là thời lượng toàn ván, chưa có đồng hồ riêng cho mỗi bên.

Giải đấu, đăng nhập, AI học máy và chơi online chưa nằm trong bản này. Nút Hòa cờ kết thúc ván sau xác nhận; chưa triển khai thương lượng hòa với AI.

## Phát triển

Tại thư mục gốc, chạy backend trong một terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 --reload
```

Mở terminal thứ hai:

```powershell
cd frontend
npm run dev
```

Mở **http://127.0.0.1:5173**. Vite chuyển `/api` sang FastAPI. Hai địa chỉ 5173 và 8000 có bộ nhớ phiên trình duyệt riêng, nên có thể mở hai ván độc lập.

Sau khi sửa frontend, chạy `npm run build` trong `frontend` để cập nhật bản chơi tại cổng 8000. Nếu build lần đầu sau khi máy chủ đã chạy, khởi động lại backend để gắn thư mục giao diện.

## Cấu trúc

```text
frontend/src/           React, giao diện, client API và canvas animation
frontend/public/       Bộ sprite và nền game
backend/api/           FastAPI routes, schema, repository và service
backend/core/          Luật bàn cờ, môi trường và hàm đánh giá
backend/agents/        Tác tử Random và Alpha-beta truyền thống
model/learning/        Encoder, MLP/CNN, train, self-play và arena
backend/api/           FastAPI, cấu hình, schema, lưu phiên và xử lý ván
backend/agents/        RandomAgent và SearchAgent cho bốn độ khó
backend/core/          ChessEnv, luật kết thúc, đánh giá lực lượng
tests/                 Kiểm thử luật, API, seed, lưu trữ và nước đặc biệt
data/chess.sqlite3     Dữ liệu ván, tự tạo khi chạy (không đưa vào Git)
docs/ARCHITECTURE.md    Thiết kế hệ thống và hợp đồng API
```

Đọc **[thiết kế hệ thống](docs/ARCHITECTURE.md)** để nối thêm AI hoặc frontend khác. `.env.example` chứa các tùy chọn đường dẫn SQLite, số ply tối đa và danh sách origin. Sao chép thành `.env` nếu cần thay đổi; mặc định dùng được ngay.

Phiên dùng token riêng do backend cấp, chỉ lưu bản băm trong SQLite. Trình duyệt lưu `game_id` và token của **ván hiện tại**; xóa dữ liệu trình duyệt sẽ mất liên kết tiếp tục ván. Đây là bản chạy tại máy cá nhân, chưa có tài khoản hoặc giao diện tìm lại các ván cũ.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

Kiểm thử timeline, đủ frame idle cho 12 quân, cài đặt chuyển động và manifest: chạy `npm test` trong `frontend` (7 kiểm thử).

21 kiểm thử backend và 7 kiểm thử animation đã qua; frontend build TypeScript thành công. Các kiểm thử mới bao gồm lưu độ khó, chọn đúng loại agent, hạn thời gian, bắt Hậu và tìm chiếu bí. Chi tiết giao diện game mới ở [UI-2026-11-02.md](docs/UI-2026-11-02.md); bản QA trước ở `docs/QA.md`.

`package-lock.json` khóa thư viện JavaScript. `requirements-lock.txt` ghi lại chính xác môi trường Python đã kiểm chứng trên máy này; `requirements.txt` là khai báo trực tiếp dùng cho cài đặt.

## Tài nguyên

Bộ animation được đưa vào `frontend/public/animations` từ `img/animations`; manifest quy định frame, điểm neo và thời lượng. Bản nguồn trong `img` được giữ lại. Nền mới: `frontend/public/assets/backgrounds/royal-hall.png`; prompt và thông tin tạo ảnh bằng công cụ image generation tích hợp nằm ở `royal-hall.prompt.json` trong cùng thư mục. Font được đóng gói cục bộ nên giao diện không cần gọi Google Fonts.

Luật cờ dùng thư viện [python-chess](https://python-chess.readthedocs.io/en/latest/), giấy phép [GPL-3.0+](https://github.com/niklasf/python-chess/blob/master/LICENSE.txt). Các dependency giữ giấy phép riêng của chúng. Báo cáo PDF được dùng làm tài liệu tham khảo; không chỉnh sửa báo cáo trong bước triển khai này.
