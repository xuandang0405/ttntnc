# Bộ animation cờ vua

Bộ chuẩn gồm 38 PNG trong suốt: 12 `move`, 12 `attack`, 12 `die` và hai hiệu ứng va chạm vàng/tím. Mỗi dải có bốn khung ngang. Manifest v3 có 12 trạng thái **idle**, lặp đủ 4 frame đã căn của cùng nhân vật với nhịp riêng 240/280/240/280 ms. Idle không còn bị giữ ở frame 0. Game thêm nhịp thở, vòng sáng và các hạt nhỏ bằng canvas.

Game FastAPI/React nằm tại thư mục gốc dự án. Bộ đang được game sử dụng ở `frontend/public/animations`. Các ảnh `idel_...` cũ được giữ nguyên; game dùng bộ đã căn này để giữ cùng tạo hình/tỷ lệ và tránh nền đặc ở một số ảnh cũ.

## Cách dùng khi làm frontend

Sao chép toàn bộ nội dung thư mục này vào `public/animations/`. Đọc `manifest.json`; các đường dẫn ảnh tương đối với manifest và dùng tên ASCII: `white/move_knight.png`, `black/attack_rook.png`...

- Khung quân cờ: 320 × 360 px; dải ảnh: 1280 × 360 px. Khung thứ `i` bắt đầu ở x = `i × 320`, y = 0.
- Điểm neo trên bàn cờ: (160,316) trong mỗi khung. Vẽ ảnh tại tọa độ bàn cờ trừ điểm neo, nhân cùng hệ số tỷ lệ cho cả hai.
- `move`: 120 ms/khung, lặp 480 ms. Quỹ đạo trên bàn cờ nội suy bằng `easeInOutCubic`; nhịp lên xuống trong sprite đã được căn lại thành 0/−6/−12/−6 px.
- `idle`: 240/280/240/280 ms; đọc `frameIndices` và phát đủ bốn frame, không ép về frame 0.
- `attack`: 140/140/140/180 ms, phát một lượt rồi trở về `idle`. Quân tấn công vẫn nguyên vẹn.
- `die`: 170/200/230/260 ms, phát một lượt rồi ẩn quân. Đối với vua, chỉ dùng trạng thái này như hiệu ứng thua trận, không phải nước ăn vua.
- `impact`: bốn khung 320 × 320 px, điểm neo (160,192), thời lượng 70/90/120/160 ms. Chọn màu theo bên tấn công.

## Nhịp ăn quân mẫu

| Thời điểm | Hiển thị |
|---|---|
| 0–480 ms | Di chuyển đến gần ô đích |
| 480 ms | Bắt đầu lấy đà/ra đòn |
| 760 ms | Đánh trúng, bật impact và bắt đầu die của quân địch |
| 1080 ms | Thu đòn |
| 1160 ms | Quân tấn công đứng ở ô đích |
| 1620 ms | Quân địch đã tan hết |
| 1760 ms | Hoàn tất hiệu ứng, mở lượt tiếp theo |

Quân mã nhảy theo cung cao 0,45 ô. Nước đi thường nội suy 650–1100 ms theo khoảng cách, có vệt sáng. Tốt đi thẳng và ăn chéo. Khi nhập thành, phát hai chuyển động vua/xe đồng thời; bắt tốt qua đường dùng ô nạn nhân riêng; phong cấp đổi tạo hình ở cuối quỹ đạo. Luật và kết quả nước đi do backend xác nhận trước, animation chỉ biểu diễn kết quả.

Renderer chờ vẽ xong rồi gọi callback mở lượt, không dùng timeout tính từ lúc API trả về để xóa sớm hiệu ứng. Khi tab bị ẩn, giữ tiến độ để không bỏ qua toàn bộ chuỗi. Idle tối đa 30 fps và dừng khi tạm dừng ván. Trong Cài đặt chọn **Đầy đủ**, **Theo thiết bị** hoặc **Tắt**; chế độ Theo thiết bị tôn trọng yêu cầu giảm chuyển động.

Mã phát tại `frontend/src/game/renderer.ts` và `motion.ts`. `scripts/upgrade-animation-metadata.py` đồng bộ hai manifest, không sửa pixel PNG. Chạy `npm test` và `npm run build` trong `frontend` sau thay đổi. Các bản xem dưới đây thuộc lần tạo ảnh cũ, không thay thế việc kiểm tra renderer đang chạy.

## Bản xem và kiểm tra

- `../../animation-preview/capture-complete/`: GIF liên hoàn của cả hai màu, các bảng khung và `validation.json`.
- `../../asset-source/capture-polish-2026-10-24/generation-spec.json`: prompt và tham chiếu của 19 atlas tạo bằng imagegen tích hợp. Các script tại đó chỉ phân tích tọa độ, đóng gói và dựng bản xem.
- `../../asset-backups/before-capture-polish-2026-10-24/`: ảnh trước lượt hoàn thiện này.

Các PNG tên tiếng Việt ở thư mục `img/trắng/` và `img/đen/` vẫn là bản tương thích kích thước cũ. Khi triển khai mới, dùng bộ chuẩn này để có khung nguyên 320 px và điểm neo thống nhất.
