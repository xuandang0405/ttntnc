# Các sprite đã sửa ngày 24/10/2026

Đã thay tổng cộng 27 tệp PNG, tương ứng 19 dải animation khác nhau: move của đủ sáu loại quân ở cả hai màu; die/defeat vua/hậu/tượng hai màu và tốt trắng. Hai tên move_xe_đen.png và move_xe_den.png cùng trỏ tới một nội dung. die và defeat của cùng quân được đồng bộ giống hệt nhau.

Lượt rà soát move bổ sung đã kiểm tra đủ 12 dải, làm lại 7 dải còn dùng tư thế tấn công hoặc đổi kích thước: mã/tượng/tốt hai màu và xe trắng. Đã sửa riêng chi tiết dải viền cổ tượng trắng bị đảo chiều giữa các khung. Năm dải move đã đạt từ lần trước được giữ nguyên. Mọi PNG ngoài bảy tệp move này giữ nguyên nội dung sau lượt rà soát bổ sung.

Ảnh được tạo lại bằng công cụ imagegen tích hợp, lấy idel của chính quân đó làm tham chiếu tạo hình. Không dùng lại khung attack làm move hoặc ghép khói của quân khác. Các bản gốc và prompt được lưu để có thể đối chiếu.

## Hành động

- `move`: quân còn nguyên, lướt/nhấp nhô nhẹ cùng vòng sáng nhỏ; lặp bốn khung theo thứ tự trái sang phải. Khi làm frontend, vị trí trên bàn cờ vẫn cần được nội suy riêng từ ô nguồn tới ô đích.
- `die` / `defeat`: nứt → bong mảnh → vỡ → hạt sáng mờ; chỉ phát một lần, sau đó ẩn quân. Khung cuối đã giảm alpha còn khoảng 16% của bản tạo gốc.
- Đối với vua, `defeat` dùng làm hiệu ứng kết thúc khi thua/chiếu hết; luật cờ vua không có nước bắt vua.

## Kích thước và tọa độ

Giữ nguyên kích thước PNG ban đầu, nền RGBA trong suốt. Mỗi dải có bốn khung ngang. Chiều rộng 1254 px không chia hết cho bốn: các khoảng x lần lượt là [0,314), [314,627), [627,940), [940,1254). Khi triển khai, dùng tọa độ nguyên trong `animation-fix-manifest.json` thay vì giả định bốn ô nguyên rộng bằng nhau.

Thời lượng gợi ý trong manifest: move 120 ms/khung, tổng 480 ms; die 140/160/180/200 ms, tổng 680 ms. Đây là thông số chuẩn bị asset, chưa phải phần triển khai FastAPI/React.

## Tệp đối chiếu

- `../animation-preview/move-review/all-move-animations.gif`: bản xem chuyển động mới nhất của đủ 12 quân, bốn khung ở 120 ms/khung, lặp 480 ms.
- `../animation-preview/move-review/all-move-frames.png`: toàn bộ 48 khung move hiện tại trên nền ô caro.
- `../animation-preview/move-review/validation.json`: kết quả kiểm tra lượt bổ sung, có đối chiếu hash để bảo đảm chỉ bảy PNG dự kiến bị thay đổi.
- `../asset-backups/before-move-review-2026-10-24/`: bản sao 13 tệp move trước lượt rà soát bổ sung (gồm tên trùng của xe đen), manifest cũ và hash của 63 PNG.
- `../asset-source/move-fix-2026-10-24/generation-spec.json`: prompt imagegen và ảnh tham chiếu cho bảy dải bổ sung, bao gồm bước sửa viền cổ tượng trắng; atlas nguồn nằm cùng thư mục. `pack-sprites.ps1` tại đây gộp các mục mới vào manifest hiện có; `make-move-previews.ps1` tạo bản xem riêng của toàn bộ move.
- `../animation-preview/fixed-animations.gif`: xem chuyển động; GIF cố ý lặp lại lượt minh họa và có khoảng trống sau khi tan. PNG gốc mới là tài nguyên dùng cho game; GIF chỉ dùng xem thử.
- `../animation-preview/all-fixed-frames.png`: toàn bộ 48 khung đã sửa, xem trên nền ô caro để kiểm tra alpha.
- `../animation-preview/validation.json`: kết quả kiểm tra 20 tệp, kích thước, alpha, mép khung, các tên tệp đồng bộ và GIF.
- `../asset-backups/before-animation-fix-2026-10-24/img/`: 20 tệp trước khi sửa, giữ nguyên cấu trúc thư mục.
- `../asset-source/animation-fix-2026-10-24/generation-spec.json`: toàn bộ prompt, ảnh tham chiếu và ánh xạ đầu ra; các atlas nguồn 2×2 nằm cùng thư mục.

`pack-sprites.ps1` đóng gói atlas nguồn thành dải ngang, không vẽ thêm nội dung. `make-previews.ps1` tạo ảnh kiểm tra và GIF. Hai script này chỉ phục vụ tài nguyên, chưa triển khai game.
