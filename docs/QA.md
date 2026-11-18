# Kiểm chứng bản đầu — 25/10/2026

Môi trường: Windows, Python 3.14.7, Node.js 24.21.0, React 19, Vite 7. Giao diện kiểm tra trong trình duyệt với bản build do FastAPI phục vụ tại cổng 8000.

## Kiểm thử tự động

| Nhóm | Nội dung | Kết quả |
|---|---|---|
| Môi trường cờ | Thế đầu, tensor 14 × 8 × 8, clone/undo, nước sai luật | Qua |
| Kết thúc | Chiếu bí, hết nước đi, lặp lại, giới hạn ply và điểm quân | Qua |
| Nước đặc biệt | Quân bị ghim, nhập thành, bắt tốt qua đường, phong cấp | Qua |
| Random | Lặp lại theo seed và telemetry xác suất 1/N | Qua |
| API và phiên | Tách ván/token, dữ liệu sai, revision cũ, sai lượt | Qua |
| Vòng chơi | Máy đi trước khi người cầm Đen, pause, đầu hàng, undo cả lượt | Qua |
| Event và lưu trữ | Ô quân bị ăn, Xe nhập thành, loại quân phong cấp, PGN, SQLite qua lần khởi động lại | Qua |

Lệnh `python -m pytest -q`: **12 passed**, có hai cảnh báo deprecation từ thư viện về asyncio và TestClient; không có kiểm thử thất bại. `npm run build`: TypeScript và Vite hoàn tất.

Launcher được chạy riêng tại cổng 8001 với DB kiểm thử: `/ready`, HTML, JS/CSS, manifest sprite và nền đều trả 200. Tiến trình kiểm thử được dừng sau khi xác nhận. Chế độ nhận lại máy chủ đang chạy tại cổng 8000 cũng đã kiểm tra.

## Kiểm tra trực tiếp trên giao diện

| Thao tác | Kết quả quan sát |
|---|---|
| Chọn e2 → e4 ở ván AI | Tô e3/e4 hợp lệ; máy trả đúng một nước; lịch sử tổng cộng 2 ply rồi trả lượt người |
| Hoàn tác | Quay về 0 ply và đủ 32 quân ở thế đầu |
| Tạm dừng/tiếp tục | Có lớp khóa bàn, thông báo tạm dừng và nút trở lại ván |
| FEN mẫu ăn quân | e4 × d5; thấy quân di chuyển, sau hiệu ứng chỉ còn Tốt trắng ở d5; lực lượng +1.0 |
| FEN mẫu phong cấp | Hiện đủ bốn lựa chọn; chọn Mã tạo `a8=N`, Mã ở a8; kết thúc hòa do thiếu lực lượng |
| FEN mẫu nhập thành | `O-O`: Vua g1, Xe f1, e1/h1 trống, lịch sử 1 ply |
| FEN sai rồi sửa | Báo lỗi ngay trong hộp nhập; đổi sang FEN đúng nạp được ván |
| Người cầm Đen, seed 42 | Bàn tự xoay, máy mở đầu `Na3`, đến lượt người; tải lại trang vẫn đúng 1 ply |
| Gợi ý | Thông báo nêu rõ một nước ngẫu nhiên, không phải nước tối ưu |
| Bản đồ và xoay bàn | Nút đổi trạng thái; hướng tọa độ và hai thanh người chơi đổi đúng |
| Cài đặt | Đổi sang bàn Óc chó và tắt âm thanh, tải lại giữ cài đặt; sau thử đã trả về Đá cổ và bật âm thanh |
| Hướng dẫn | Có luật, phím tắt; mở được cả nút trên desktop và biểu tượng trên điện thoại |
| Đầu hàng | Có hộp xác nhận; người cầm Đen xác nhận thì Trắng thắng |
| Xuất PGN | Trình duyệt tải thật tệp `ky-dai-van-co.pgn` |

Ở viewport 390 × 844, vùng nội dung rộng 375 px sau thanh cuộn: đủ 64 ô, `scrollWidth == clientWidth`, thanh điểm nằm trong khung nhìn. Đã sửa việc lịch sử kéo cuộn cả trang, khoảng trống cho nhãn điểm và lối mở hướng dẫn trên điện thoại. Desktop kiểm tra ở 1366 × 900. Không có lỗi hoặc cảnh báo JavaScript trong lần kiểm tra bản build cuối.

Ảnh tại `test-results/screenshots/desktop.jpg` và `mobile.jpg` trong thư mục dự án. Đây là kiểm tra tương tác thực tế, chưa phải bộ kiểm thử E2E tự động. Chưa kiểm tra nghe âm thanh qua loa hoặc toàn màn hình trên mọi trình duyệt; hai chức năng dùng Web Audio và Fullscreen API, có xử lý khi trình duyệt không hỗ trợ.
