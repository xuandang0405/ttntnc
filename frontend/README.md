# Frontend — giao diện game

React + TypeScript + Vite; Canvas hiển thị bản đồ và sprite animation. Frontend gọi REST API; backend quyết định luật, kết quả và mở khóa map.

| Đường dẫn | Nội dung |
|---|---|
| src/main.tsx | Điều phối menu, chơi, lịch sử, pause và request |
| src/Menu.tsx | Menu bắt đầu, chọn độ khó, chọn map |
| src/Board.tsx | Vẽ bàn cờ, animation, laser, kim cương, EXIT |
| src/api.ts, src/types.ts | API client và kiểu dữ liệu |
| src/*.css | Giao diện và responsive |
| public/assets/ | Logo, nhân vật, background và ảnh 5 map |
| vite.config.ts | Build và proxy tới backend localhost:8000 |

Chạy từ thư mục frontend, khi API ở localhost:8000 đã sẵn sàng:

```powershell
npm.cmd ci
npm.cmd run dev
```

Build: `npm.cmd run build`. Thư mục dist là kết quả sinh tự động, được gitignore. Vite sao chép public/assets vào dist/assets; FastAPI phục vụ bản build khi chạy Docker.

Lệnh Docker chạy từ gốc Souce: `docker compose up --build -d`. Đường dẫn HTTP `/assets/...`, `/api/...` được giữ nguyên sau khi tách thư mục.
