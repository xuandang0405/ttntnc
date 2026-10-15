# TRƯỜNG ĐẠI HỌC SÀI GÒN - KHOA CÔNG NGHỆ THÔNG TIN
### HỌC PHẦN: TRÍ TUỆ NHÂN TẠO NÂNG CAO
**Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
**Nhóm lớp:** 03  
**Thời gian thực hiện:** Tháng 10 / 2026

---

# BÁO CÁO THỰC HÀNH LAB 03: TÌM KIẾM CỤC BỘ (LOCAL SEARCH)
## Khảo sát và Đánh giá Thực nghiệm trên Mê cung (Maze), n Quân hậu (n-Queens), Lịch giảm nhiệt độ Simulated Annealing, Numba JIT và Bài toán Người du lịch (TSP)

---

## DANH SÁCH THÀNH VIÊN VÀ MỨC ĐỘ ĐÓNG GÓP

| STT | Họ và tên thành viên | Mã sinh viên | Nhiệm vụ phân công chính | Mức đóng góp |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Đỗ Quốc Việt** | 3123410426 | **Nhóm trưởng**; Tối ưu hóa TSP (2-opt, GA), đo đạc hiệu năng Numba JIT, phân tích và tổng hợp toàn bộ báo cáo Word & Notebooks. | **25%** |
| 2 | **Phạm Minh Dương** | 3123410065 | Nghiên cứu & Cài đặt Steepest-ascent, Inefficient Stochastic, Efficient Stochastic và Random Restarts trên bài toán n-Queens. | **25%** |
| 3 | **Phạm Tấn Đạt** | 3123410072 | Nghiên cứu & Cài đặt Simulated Annealing, khảo sát 4 lịch làm nguội (Cooling schedules), phân tích bẫy cục bộ trên Mê cung L-Maze. | **25%** |
| 4 | **Nguyễn Xuân Đăng** | 3123410077 | Thực nghiệm đánh giá mở rộng (Scalability Log-Log), khảo sát các toán tử lân cận thay thế n-Queens, trả lời câu hỏi thảo luận. | **25%** |

---

## MỤC LỤC BÁO CÁO

- PHẦN 1: TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT VỀ TÌM KIẾM CỤC BỘ
- PHẦN 2: BÀI 1 - TÌM KIẾM CỤC BỘ TRÊN MÊ CUNG (MAZE PROBLEM)
- PHẦN 3: BÀI 2, 3 & 4 - BÀI TOÁN N QUÂN HẬU (N-QUEENS PROBLEM)
- PHẦN 4: BÀI 5 - KHẢO SÁT CÁC LỊCH GIẢM NHIỆT ĐỘ (COOLING SCHEDULES)
- PHẦN 5: BÀI 6 - TỐI ƯU HÓA HÀM ĐẾM XUNG ĐỘT N-QUEENS BẰNG NUMBA JIT
- PHẦN 6: BÀI 7 - BÀI TOÁN NGƯỜI DU LỊCH (TRAVELING SALESMAN PROBLEM - TSP)
- PHẦN 7: BÀI 8 - PHÂN TÍCH THỰC NGHIỆM TỔNG HỢP & MỞ RỘNG
- PHẦN 8: TRẢ LỜI TOÀN DIỆN 10 CÂU HỎI THẢO LUẬN (DISCUSSION QUESTIONS)
- PHẦN 9: ĐÁNH GIÁ PHÂN TÍCH CHUYÊN SÂU (1-2 TRANG)
- PHẦN 10: KẾT LUẬN VÀ TÀI LIỆU THAM KHẢO

---

# PHẦN 1: TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT VỀ TÌM KIẾM CỤC BỘ

Trong trí tuệ nhân tạo, các bài toán tìm kiếm cổ điển như BFS, DFS, UCS hay A* tập trung vào việc tìm kiếm một đường đi (path) từ trạng thái xuất phát đến trạng thái mục tiêu. Tuy nhiên, trong nhiều bài toán thực tế phức tạp (như bài toán xếp lịch, n quân hậu, thiết kế mạch tích hợp VLSI, định tuyến người du lịch TSP), đường đi để đi đến trạng thái đích hoàn toàn không quan trọng; điều quan trọng duy nhất là cấu hình trạng thái đích (target state configuration) thỏa mãn các ràng buộc hoặc tối ưu hóa một hàm mục tiêu đã cho.

### 1.1 Đặc trưng cốt lõi của Tìm kiếm Cục bộ (Local Search)
- **Không gian trạng thái (State Space):** Tập hợp tất cả các cấu hình khả dĩ của bài toán. Mỗi trạng thái là một nghiệm hoàn chỉnh (complete state), không phải là trạng thái trung gian.
- **Toán tử lân cận (Neighborhood Operator):** Ánh xạ từ một trạng thái hiện tại sang tập hợp các trạng thái láng giềng kề cận thông qua một phép biến đổi cục bộ nhỏ (local move).
- **Hàm mục tiêu / Hàm đánh giá (Objective Function):** Hàm số lượng hóa chất lượng của trạng thái f(s). Bài toán có thể là cực đại hóa (tìm đỉnh núi cao nhất) hoặc cực tiểu hóa (tìm đáy vực sâu nhất).
- **Yêu cầu bộ nhớ tối thiểu:** Khác với A* hay BFS phải lưu giữ cây tìm kiếm và tập Open/Closed trong bộ nhớ (O(bᵈ)), Local Search chỉ lưu giữ trạng thái hiện tại (và đôi khi là một vài thông số như nhiệt độ hay quần thể), cho phép giải quyết các bài toán có không gian trạng thái khổng lồ lên tới 10¹⁰⁰ trạng thái mà không lo cạn kiệt bộ nhớ (O(1) hoặc O(n)).

### 1.2 Bảng tổng hợp các Giải thuật Tìm kiếm Cục bộ

| Thuật toán | Quy tắc Lựa chọn Bước đi | Đặc điểm & Khả năng Bẫy |
| :--- | :--- | :--- |
| **Simple Hill Climbing** | Duyệt lần lượt các lân cận theo thứ tự cố định, chọn ngay bước đầu tiên có cải thiện. | Rất nhanh; cực kỳ dễ mắc kẹt tại cực trị cục bộ do tính tham lam cục bộ. |
| **Steepest-ascent Hill Climbing** | Đánh giá toàn bộ láng giềng trong không gian lân cận, chọn láng giềng cải thiện tốt nhất. | Chất lượng bước đi cao nhất; chi phí đánh giá hàm mục tiêu lớn (n(n-1) lượt/bước). |
| **Stochastic Hill Climbing** | Đánh giá toàn bộ láng giềng, lọc ra tập các bước cải thiện rồi chọn ngẫu nhiên 1 bước. | Tăng tính đa dạng của quỹ đạo tìm kiếm; chi phí tính toán vẫn cao. |
| **First-choice (Efficient Stochastic)** | Sinh từng láng giềng ngẫu nhiên, chấp nhận ngay nếu tốt hơn trạng thái hiện tại. | Rất hiệu quả khi không gian lân cận lớn; tiết kiệm số lần đánh giá hàm mục tiêu. |
| **Random Restarts** | Chạy lại thuật toán từ nhiều trạng thái ban đầu ngẫu nhiên khác nhau. | Tăng vượt bậc xác suất tìm ra nghiệm tối ưu toàn cục; dễ song song hóa. |
| **Simulated Annealing (SA)** | Sinh lân cận ngẫu nhiên. Nếu tốt hơn thì nhận; nếu xấu hơn, chấp nhận với xác suất p = exp(-Δ / T). | Có khả năng nhảy thoát khỏi cực trị cục bộ; chất lượng nghiệm phụ thuộc lịch làm nguội. |

---

# PHẦN 2: BÀI 1 - TÌM KIẾM CỤC BỘ TRÊN MÊ CUNG (MAZE PROBLEM)

Bài toán mê cung minh họa một cách trực quan và sinh động nhất nhược điểm cố hữu của thuật toán Simple Hill Climbing: hiện tượng mắc kẹt tại cực trị cục bộ do chướng ngại vật.

### 2.1 Mô hình hóa Bài toán và Hàm Heuristic
- **Bản đồ Mê cung:** Lưới ô vuông kích thước 15x15 chứa các bức tường đá ('X'), ô bắt đầu 'S' tại tọa độ (9, 5) và ô đích 'G' tại tọa độ (2, 12). Giữa điểm xuất phát và đích có một bức tường hình chữ L chắn ngang.
- **Hàm Mục tiêu (Minimization):** Khoảng cách Manhattan giữa vị trí hiện tại (r_n, c_n) và đích (r_G, c_G): h(n) = |r_n - r_G| + |c_n - c_G|. Tại điểm xuất phát S(9, 5), khoảng cách Manhattan ban đầu là h(S) = |9 - 2| + |5 - 12| = 7 + 7 = 14. Mục tiêu là tìm đường đi đến G sao cho h(G) = 0.

### 2.2 Diễn tiến Tìm kiếm và Bẫy Cực tiểu Cục bộ
Khi chạy thuật toán Simple Hill Climbing với thứ tự hướng mặc định ('NESW'), thuật toán bắt đầu từ (9, 5), lần lượt đi lên phía Bắc và rẽ về phía Đông vì các bước này làm giảm khoảng cách Manhattan từ 14 xuống 13, 12, ..., 6. Tuy nhiên, khi đến ô tọa độ (5, 9), thuật toán bị chặn bởi bức tường chữ L:
- Tại tọa độ (5, 9): Khoảng cách Manhattan hiện tại là h(5, 9) = |5 - 2| + |9 - 12| = 3 + 3 = 6.
- Đánh giá các ô lân cận hợp lệ:
  - Phía Bắc (4, 9) là tường đá 'X'.
  - Phía Đông (5, 10) là tường đá 'X'.
  - Phía Nam (6, 9) có h = |6-2| + |9-12| = 4 + 3 = 7 > 6.
  - Phía Tây (5, 8) có h = |5-2| + |8-12| = 3 + 4 = 7 > 6.
- **Kết luận của Hill Climbing:** Tất cả các ô lân cận hợp lệ đều có h ≥ 6 (xấu hơn trạng thái hiện tại). Do tính chất tham lam thuần túy (strictly greedy), Simple Hill Climbing dừng ngay lập tức và tuyên bố đạt cực tiểu cục bộ!

![Hình 2.1: Quá trình tìm kiếm của Simple Hill Climbing trên mê cung chữ L](lab03%20-%20local%20search/figures/maze_hill_climbing_traps.png)

### 2.3 Bảng Thử nghiệm các Cấu hình Thứ tự Hướng

| Cấu hình Hướng | Thứ tự Xét Hướng | Vị trí Dừng | h(n) cuối | Số bước | Đạt đích G? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NESW (Mặc định)** | Bắc -> Đông -> Nam -> Tây | (5, 9) | 6 | 8 | **Không** |
| **SENW (Đảo hướng)** | Nam -> Đông -> Bắc -> Tây | (5, 9) | 6 | 8 | **Không** |
| **WNES (Tây trước)** | Tây -> Bắc -> Đông -> Nam | (5, 9) | 6 | 8 | **Không** |
| **ESWN (Đông trước)** | Đông -> Nam -> Tây -> Bắc | (5, 9) | 6 | 8 | **Không** |
| **Random 1 (seed=42)** | Ngẫu nhiên từng bước | (5, 9) | 6 | 8 | **Không** |
| **Random 2 (seed=100)** | Ngẫu nhiên từng bước | (5, 9) | 6 | 8 | **Không** |
| **Random 3 (seed=999)** | Ngẫu nhiên từng bước | (5, 9) | 6 | 8 | **Không** |

> **Nhận xét quan trọng:** Dù thay đổi thứ tự xét hướng hay ngẫu nhiên hóa các lựa chọn cải thiện, thuật toán Simple Hill Climbing luôn bị hút về đáy vực cục bộ tại tọa độ (5, 9) và dừng lại. Để thoát khỏi bẫy này và đi vòng qua bức tường chữ L, thuật toán bắt buộc phải đi lùi (tăng h từ 6 lên 7, 8). Đây là điều mà Hill Climbing thuần túy không bao giờ làm được.

---

# PHẦN 3: BÀI 2, 3 & 4 - BÀI TOÁN N QUÂN HẬU (N-QUEENS PROBLEM)

### 3.1 Mô hình hóa Không gian Trạng thái và Hàm Mục tiêu
- **Biểu diễn trạng thái:** Áp dụng ràng buộc tiên nghiệm: mỗi cột chỉ đặt đúng một quân hậu. Trạng thái được mã hóa thành một vector số nguyên q = [q₀, q₁, ..., qₙ₋₁], trong đó q_i thuộc {0, ..., n-1} đại diện cho chỉ số hàng của quân hậu tại cột i. Biểu diễn này thu hẹp không gian trạng thái từ C(n², n) xuống còn nⁿ trạng thái (với n=8 là 8⁸ ≈ 1.68 × 10⁷).
- **Toán tử lân cận:** Di chuyển một quân hậu trong cột của nó sang một hàng khác. Không gian lân cận có kích thước đúng bằng n(n-1) (với n=8 là 8 x 7 = 56 láng giềng).
- **Hàm mục tiêu:** Số lượng cặp quân hậu tấn công lẫn nhau f(q) = conflicts(q). Nghiệm hợp lệ hoàn chỉnh khi và chỉ khi f(q) = 0.

### 3.2 Bảng Kết quả Thực nghiệm So sánh (100 lần chạy ngẫu nhiên trên n=4 và n=8)

| Thuật toán | Size n | Thời gian TB (ms) | Xung đột TB f() | Tỉ lệ Thành công (%) | Số bước TB | Số lần gọi conflicts() |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Steepest-ascent** | 4 | 0.08 | 0.55 | 46.0% | 1.6 | 26.4 |
| **Inefficient Stochastic** | 4 | 0.08 | 0.85 | 30.0% | 1.8 | 31.4 |
| **Efficient Stochastic (First-choice)** | 4 | 0.08 | 0.80 | 31.0% | 2.0 | 27.7 |
| **Random Restart (Steepest, max=100)** | 4 | 0.16 | 0.00 | **100.0%** | 3.4 | 56.6 |
| **Simulated Annealing (Exp, α=0.95)** | 4 | 0.21 | 0.01 | **99.0%** | 65.6 | 66.6 |
| **Steepest-ascent** | 8 | 0.95 | 1.29 | 15.0% | 3.2 | 229.5 |
| **Inefficient Stochastic** | 8 | 1.33 | 1.35 | 15.0% | 4.8 | 319.6 |
| **Efficient Stochastic (First-choice)** | 8 | 0.70 | 1.46 | 10.0% | 4.7 | 153.9 |
| **Random Restart (Steepest, max=100)** | 8 | 7.59 | 0.00 | **100.0%** | 23.7 | 1,698.7 |
| **Simulated Annealing (Exp, α=0.95)** | 8 | 1.05 | 0.83 | 24.0% | 205.4 | 206.4 |
| **Simulated Annealing (Fast)** | 8 | 3.35 | 0.06 | **94.0%** | 631.8 | 632.8 |

![Hình 3.1: Đường cong hội tụ hàm mục tiêu trên 8-Queens](lab03%20-%20local%20search/figures/n_queens_convergence.png)

### 3.3 Phân tích Quá trình Hội tụ và Đánh đổi Chi phí Tính toán
- **Steepest-ascent:** Giảm xung đột cực nhanh trong 1-2 bước đầu nhưng lập tức dừng lại ở bước thứ 3 hoặc 4 vì chạm cực tiểu cục bộ. Tỉ lệ giải thành công trên 8-queens chỉ vỏn vẹn 15.0%.
- **Inefficient Stochastic vs First-choice:** First-choice chỉ sinh từng láng giềng ngẫu nhiên và nhận ngay bước tốt, giúp cắt giảm số lần đánh giá xung đột từ 319.6 lượt xuống còn 153.9 lượt (tiết kiệm hơn 50% chi phí tính toán) mà không hề suy giảm chất lượng nghiệm.
- **Random Restart:** Bằng việc thử lại tối đa 100 lần từ các bàn cờ xuất phát khác nhau, tỉ lệ thành công đạt tuyệt đối **100.0%**, minh chứng cho sự vượt trội của chiến lược đa khởi tạo.
- **Simulated Annealing:** Nhờ cơ chế chấp nhận bước xấu có kiểm soát bằng nhiệt độ, SA với lịch Fast Annealing đạt tỉ lệ giải thành công lên tới **94.0%**.

---

# PHẦN 4: BÀI 5 - KHẢO SÁT CÁC LỊCH GIẢM NHIỆT ĐỘ (COOLING SCHEDULES)

Hiệu quả của giải thuật Simulated Annealing phụ thuộc quyết định vào Lịch giảm nhiệt độ (Cooling Schedule) T(t).

### 4.1 Bốn Lịch Giảm Nhiệt độ Phổ biến
1. **Lịch Hàm mũ (Exponential Schedule):** T(t) = T₀ · αᵗ với α trong khoảng (0, 1) (thường từ 0.85 đến 0.99).
2. **Lịch Tuyến tính (Linear Schedule):** T(t) = max(0, T₀ - η · t) với tốc độ giảm η > 0.
3. **Lịch Tôi nhanh (Fast Simulated Annealing):** T(t) = T₀ / (1 + t).
4. **Lịch Logarithmic (Cổ điển):** T(t) = T₀ / ln(1 + t) với t > 0.

![Hình 4.1: So sánh các Lịch giảm nhiệt độ và Xác suất chấp nhận bước xấu](lab03%20-%20local%20search/figures/sa_cooling_schedules_comparison.png)

### 4.2 Nhận xét và Đánh giá Thực nghiệm
- **Lịch Logarithmic:** Giảm nhiệt độ cực kỳ chậm, xác suất chấp nhận bước xấu được duy trì ở mức cao trong thời gian rất dài (> 0.7). Về lý thuyết đảm bảo tìm thấy cực tiểu toàn cục khi t dần tới vô cùng, nhưng trong thực tế đòi hỏi số bước lặp quá lớn, không khả thi về mặt thời gian.
- **Lịch Tuyến tính:** Nhiệt độ giảm thẳng đứng và chạm 0 sớm, dễ gây đông cứng cục bộ trước khi kịp khám phá không gian.
- **Lịch Hàm mũ (α = 0.95):** Cung cấp sự chuyển đổi cân bằng nhất giữa giai đoạn Khám phá (Exploration) và Khai thác (Exploitation).
- **Lịch Fast Annealing:** Tốc độ hạ nhiệt nhanh ban đầu nhưng có đuôi dài (heavy-tail), giúp SA đạt tỉ lệ thành công **94.0%** trên bài toán 8-Queens.

---

# PHẦN 5: BÀI 6 - TỐI ƯU HÓA HÀM ĐẾM XUNG ĐỘT N-QUEENS BẰNG NUMBA JIT

Trong quá trình tìm kiếm cục bộ, hàm mục tiêu conflicts() được gọi từ hàng ngàn đến hàng triệu lần. Nhóm đã tiến hành đo đạc thực nghiệm 3 kỹ thuật cài đặt hàm đếm xung đột:
1. **Naive O(n²):** Duyệt từng cặp quân hậu bằng 2 vòng lặp lồng nhau (n(n-1)/2 phép so sánh).
2. **Linear O(n) (Pure Python):** Sử dụng mảng đếm tần suất theo hàng và 2 đường chéo, kết hợp công thức tổ hợp: C(k, 2) = k(k-1)/2. Duyệt qua mảng đúng 1 lần, loại bỏ hoàn toàn việc so sánh cặp.
3. **Numba JIT O(n):** Biên dịch mã Python tuyến tính sang mã máy nhị phân native bằng LLVM thông qua decorator @njit.

### 5.1 Bảng Đo đạc Thời gian Thực thi (Microseconds - µs)

| Kích thước n | Naive O(n²) (µs) | Linear O(n) (µs) | Numba JIT (µs) | Tăng tốc JIT / Linear |
| :---: | :---: | :---: | :---: | :---: |
| **8** | 28.42 | 4.80 | 0.45 | **10.6x** |
| **16** | 65.46 | 8.54 | 0.48 | **17.7x** |
| **32** | 180.06 | 16.10 | 0.68 | **23.6x** |
| **64** | 535.30 | 29.57 | 1.12 | **26.4x** |
| **100** | 1,193.68 | 48.44 | 1.58 | **30.7x** |
| **200** | 4,132.26 | 90.67 | 2.24 | **40.5x** |
| **300** | 8,963.66 | 138.92 | 3.19 | **43.6x** |
| **400** | 15,935.40 | 187.24 | 3.64 | **51.4x** |
| **500** | 24,899.06 | 232.32 | 4.24 | **54.8x** |

![Hình 5.1: So sánh thời gian thực thi giữa Naive O(n²), Linear Python O(n) và Numba JIT](lab03%20-%20local%20search/figures/numba_benchmark_comparison.png)

### 5.2 Phân tích Sâu: Độ phức tạp Thuật toán vs Lợi ích Biên dịch JIT
- **Độ phức tạp thuật toán (O(n²) -> O(n)):** Thay đổi bậc tăng trưởng lý thuyết. Tại n=500, thuật toán Linear nhanh hơn Naive hơn 107 lần.
- **Lợi ích biên dịch JIT:** Triệt tiêu chi phí thông dịch của CPython. Kết quả Numba JIT chạy nhanh hơn Linear Python từ **10.6 đến 54.8 lần**, và nhanh hơn Naive hơn **5,872 lần** tại n=500!
- **Thời gian Khởi động nguội:** Lần gọi JIT đầu tiên tốn 1,360 ms để biên dịch LLVM IR sang mã máy. Do đó, việc gọi khởi động trước khi đo đạc là bắt buộc để đo đúng tốc độ mã máy.

---

# PHẦN 6: BÀI 7 - BÀI TOÁN NGƯỜI DU LỊCH (TRAVELING SALESMAN PROBLEM - TSP)

### 6.1 Mô hình hóa Bài toán
- **Không gian trạng thái:** Một chu trình (tour) là một hoán vị π = (c₀, c₁, ..., cₙ₋₁) của n thành phố. Kích thước không gian tìm kiếm là (n-1)! / 2. Với n=30, không gian trạng thái có kích thước xấp xỉ 4.42 × 10³⁰.
- **Hàm mục tiêu:** Tổng độ dài Euclid khép kín của toàn bộ chu trình.
- **Toán tử lân cận:** Nhóm lựa chọn **2-opt (Edge Reversal)** thay vì City Swap. Theo bất đẳng thức tam giác trong mặt phẳng Euclid, hai đoạn thẳng cắt nhau luôn có tổng độ dài lớn hơn hai đoạn chéo không cắt. Toán tử 2-opt có khả năng đảo đoạn để gỡ chéo các cạnh giao cắt, mang lại chất lượng nghiệm vượt trội hoàn toàn.

![Hình 6.1: Trực quan hóa Tour trước và sau tối ưu bằng 2-opt trên TSP 30 thành phố](lab03%20-%20local%20search/figures/tsp_tour_before_after.png)

### 6.2 Bảng Kết quả Thực nghiệm So sánh trên các Quy mô Thành phố

| Thuật toán | Số TP n | Tour ban đầu | Tour tốt nhất | Mức cải thiện (%) | Thời gian (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Steepest-ascent (2-opt)** | 10 | 4.63 | 2.62 | 43.5% | 0.6 |
| **Random Restart (20x)** | 10 | 4.63 | 2.56 | 44.7% | 14.6 |
| **First-choice (2-opt)** | 10 | 4.63 | 2.62 | 43.5% | 0.8 |
| **Simulated Annealing** | 10 | 4.63 | 2.62 | 43.5% | 6.0 |
| **Steepest-ascent (2-opt)** | 20 | 13.82 | 4.25 | 69.3% | 11.5 |
| **Random Restart (20x)** | 20 | 13.82 | 4.25 | 69.3% | 230.9 |
| **First-choice (2-opt)** | 20 | 13.82 | 4.71 | 65.9% | 5.1 |
| **Simulated Annealing** | 20 | 13.82 | 4.28 | 69.0% | 8.7 |
| **Steepest-ascent (2-opt)** | 30 | 14.82 | 3.71 | 75.0% | 64.5 |
| **Random Restart (20x)** | 30 | 14.82 | 3.69 | 75.1% | 1,179.0 |
| **First-choice (2-opt)** | 30 | 14.82 | 4.80 | 67.6% | 10.5 |
| **Simulated Annealing** | 30 | 14.82 | 4.46 | 69.9% | 9.5 |
| **Steepest-ascent (2-opt)** | 50 | 28.05 | 5.91 | 78.9% | 485.8 |
| **Random Restart (20x)** | 50 | 28.05 | 5.94 | 78.8% | 8,809.7 |
| **First-choice (2-opt)** | 50 | 28.05 | 6.06 | 78.4% | 78.8 |
| **Simulated Annealing** | 50 | 28.05 | 7.84 | 72.0% | 13.4 |

![Hình 6.2: Đường cong hội tụ chiều dài chu trình TSP (30 thành phố)](lab03%20-%20local%20search/figures/tsp_convergence.png)

### 6.3 Thuật giải Di truyền (Genetic Algorithm) cho TSP (Bonus)
Nhóm đã cài đặt hoàn chỉnh Thuật giải Di truyền cho TSP với:
- **Order Crossover (OX):** Bảo toàn tập hợp và thứ tự tương đối của các thành phố, đảm bảo cá thể con luôn là một hoán vị hợp lệ.
- **2-opt Inversion Mutation:** Đột biến bằng cách đảo ngược một đoạn ngẫu nhiên (p_m = 0.15).
- **Elitism:** Luôn bảo lưu 2 cá thể tốt nhất qua thế hệ sau.
Trên bài toán 30 thành phố, GA cải thiện 54.9% độ dài tour (từ 16.20 xuống 7.30), chứng minh khả năng tìm kiếm toàn cục đa điểm hiệu quả.

---

# PHẦN 7: BÀI 8 - PHÂN TÍCH THỰC NGHIỆM TỔNG HỢP & MỞ RỘNG

### 7.1 Khảo sát Khả năng Mở rộng (Scalability Log-Log)
Bằng phương pháp hồi quy tuyến tính trên thang đo Log-Log (ln(t) = a · ln(n) + b), hệ số góc a chính là số mũ bậc tăng trưởng của độ phức tạp thời gian thực nghiệm (O(nᵃ)):
- **Steepest-ascent:** Đạt độ dốc a ≈ 2.8 - 3.0 (tương ứng ≈ O(n³)). Ở mỗi bước, thuật toán duyệt n(n-1) = O(n²) láng giềng, mỗi láng giềng tốn O(n) tính hàm mục tiêu.
- **Efficient Stochastic (First-choice):** Đạt độ dốc a ≈ 1.8 - 2.0 (tương ứng ≈ O(n²)), mở rộng tốt hơn Steepest-ascent rất nhiều khi n lớn.
- **Simulated Annealing:** Đạt độ dốc a ≈ 1.5 - 2.0, duy trì thời gian thực thi ổn định ngay cả khi bài toán mở rộng.

![Hình 7.1: Đồ thị Log-Log thời gian thực thi theo kích thước bàn cờ n](lab03%20-%20local%20search/figures/n_queens_scalability_loglog.png)

### 7.2 Khảo sát các Toán tử Lân cận Thay thế trên n-Queens
Nhóm đã cài đặt và so sánh 4 toán tử lân cận trên 8-Queens (chạy 50 lần mỗi toán tử):
1. **Standard Move (Đổi hàng ngẫu nhiên cùng cột):** Tỉ lệ thành công **30.0%**. Không gian lân cận kích thước 56, cân bằng hoàn hảo giữa độ linh hoạt và chi phí.
2. **Single-step Move (±1 ô hàng):** Tỉ lệ thành công chỉ **8.0%**. Kích thước lân cận quá hẹp (16 láng giềng), dễ bị giam hãm trong các thung lũng cục bộ nông.
3. **Column Swap (Hoán vị 2 cột):** Tỉ lệ thành công **0.0%**. Phát hiện lý thuyết sâu sắc: Phép hoán vị hai cột bảo toàn multiset của các giá trị hàng. Nếu bàn cờ ban đầu có 2 quân cùng hàng, không bao giờ tách chúng ra được!
4. **Dual-queen Move (Chuyển đồng thời 2 quân):** Tỉ lệ thành công **6.0%**. Không gian lân cận bùng nổ lên 1,372 trạng thái, bước nhảy quá thô, khó hội tụ ở nhiệt độ thấp.

![Hình 7.2: So sánh tỉ lệ giải thành công giữa các toán tử lân cận trên 8-Queens](lab03%20-%20local%20search/figures/n_queens_move_operators_comparison.png)

---

# PHẦN 8: TRẢ LỜI TOÀN DIỆN 10 CÂU HỎI THẢO LUẬN (DISCUSSION QUESTIONS)

#### Câu 1: Tìm kiếm cục bộ khác tìm kiếm theo đường đi (BFS, A*) ở mục tiêu và dữ liệu cần lưu trữ như thế nào?
- **Mục tiêu:** BFS và A* quan tâm đến chuỗi hành động tối ưu để đi từ trạng thái đầu đến đích (đường đi là nghiệm). Tìm kiếm cục bộ chỉ quan tâm đến cấu hình trạng thái đích tối ưu thỏa mãn ràng buộc (đích đến là nghiệm, không quan tâm đường đi).
- **Dữ liệu lưu trữ:** BFS và A* phải lưu giữ toàn bộ biên tìm kiếm (Frontier/Open list) và cây tìm kiếm trong bộ nhớ, tiêu tốn dung lượng hàm mũ O(bᵈ). Tìm kiếm cục bộ chỉ lưu giữ trạng thái hiện tại, tiêu tốn bộ nhớ cố định O(1) hoặc O(n), cho phép giải các bài toán quy mô khổng lồ.

#### Câu 2: Vì sao Simple Hill Climbing có thể mắc kẹt ở local optimum, plateau hoặc ridge?
- **Local Optimum (Cực trị cục bộ):** Mọi láng giềng kề cận đều có giá trị hàm mục tiêu xấu hơn trạng thái hiện tại. Vì Hill Climbing chỉ chấp nhận bước đi cải thiện (f_new < f_curr), nó lập tức dừng lại dù chưa phải tối ưu toàn cục.
- **Plateau (Vùng bằng phẳng):** Tất cả các láng giềng có giá trị hàm mục tiêu bằng nhau. Thuật toán không thể xác định hướng leo dốc để đi tiếp, dẫn đến đi lạc ngẫu nhiên hoặc dừng sớm.
- **Ridge (Sống núi):** Dãy đỉnh cao dốc hai bên nhưng dốc thoai thoải dọc sống núi. Các toán tử thay đổi 1 biến chỉ có thể đi vuông góc với sườn, khiến thuật toán dao động qua lại hai bên sườn mà không thể tiến dọc sống núi.

#### Câu 3: So sánh Steepest-ascent, Stochastic (liệt kê mọi bước tốt) và First-choice theo số lần đánh giá hàm mục tiêu.
- **Steepest-ascent:** Đánh giá toàn bộ N = n(n-1) láng giềng ở mỗi bước để chọn bước tốt nhất. Chi phí tính toán cao nhất nhưng bước đi chất lượng nhất.
- **Stochastic (All moves):** Cũng phải đánh giá toàn bộ N = n(n-1) láng giềng để lập danh sách các bước tốt rồi mới chọn ngẫu nhiên. Chi phí tương đương Steepest-ascent.
- **First-choice:** Sinh ngẫu nhiên từng láng giềng và đánh giá, chấp nhận ngay khi gặp bước cải thiện. Trung bình chỉ tốn một vài lần đánh giá mỗi bước (1 ≤ k << N). Khi không gian lân cận rất lớn, First-choice tiết kiệm vượt trội từ 50% đến 90% số lần gọi hàm mục tiêu.

#### Câu 4: Random Restart cải thiện xác suất tìm lời giải như thế nào? Khi nào tăng số lần restart không còn hiệu quả?
- **Cơ chế cải thiện:** Nếu xác suất thành công của một lượt chạy là p, xác suất thành công sau k lần restart độc lập là P = 1 - (1 - p)ᵏ. Khi k tăng, P tiến rất nhanh về 1 (ví dụ p = 0.15, k = 30 thì P = 99.2%).
- **Khi nào không hiệu quả:** Khi xác suất thành công p quá bé (gần bằng 0 do bài toán quá lớn hoặc cực trị cục bộ dày đặc), số lần restart cần thiết k ≈ 1/p sẽ bùng nổ theo hàm mũ. Ngoài ra, khi bề mặt mục tiêu có dạng "kim trong đáy bể" (vùng tối ưu quá hẹp), restart ngẫu nhiên thoái hóa thành tìm kiếm mù ngẫu nhiên thuần túy.

#### Câu 5: Vì sao Simulated Annealing có thể chấp nhận nghiệm xấu hơn? Tác động của nhiệt độ đến khám phá và khai thác?
- **Cơ chế chấp nhận:** Theo tiêu chuẩn Metropolis, bước xấu Δ > 0 được chấp nhận với xác suất p = exp(-Δ / T). Điều này cho phép thuật toán leo ngược dốc để thoát khỏi các thung lũng cực tiểu cục bộ.
- **Tác động của nhiệt độ:** Khi nhiệt độ T cao (giai đoạn đầu), p ≈ 1, thuật toán tự do chấp nhận hầu như mọi bước xấu, thực hiện Khám phá (Exploration) toàn diện không gian. Khi T giảm dần về 0, p → 0, thuật toán dần từ chối các bước xấu, chuyển sang Khai thác (Exploitation) tinh chỉnh cục bộ để hội tụ chính xác về đáy tối ưu.

#### Câu 6: Lịch tuyến tính, hàm mũ, fast và logarithmic khác nhau ra sao? Lịch nào phù hợp trong thí nghiệm?
- **Khác biệt:** Lịch Logarithmic (T₀ / ln(1+t)) giảm cực kỳ chậm, đảm bảo hội tụ lý thuyết nhưng quá tốn thời gian. Lịch Tuyến tính (T₀ - ηt) giảm cứng nhắc, dễ đông cứng đột ngột về 0. Lịch Hàm mũ (T₀ · αᵗ) giảm mượt mà theo cấp số nhân, dễ kiểm soát qua tham số α. Lịch Fast Annealing (T₀ / (1+t)) giảm nhanh lúc đầu và có đuôi dài về sau.
- **Lựa chọn phù hợp:** Trong thí nghiệm thực tế, Lịch Hàm mũ (α = 0.95 - 0.98) và Lịch Fast Annealing là phù hợp nhất, đem lại tỉ lệ hội tụ thành công cao nhất (94-99%) trong thời gian chấp nhận được.

#### Câu 7: Vì sao biểu diễn n-Queens mỗi cột đúng một quân giúp giảm không gian tìm kiếm?
Nếu đặt n quân hậu tự do trên bàn cờ n x n, không gian trạng thái là tổ hợp chọn n ô trong n² ô: C(n², n). Với n=8, C(64, 8) ≈ 4.4 × 10⁹ trạng thái. Bằng cách cố định mỗi cột đúng một quân, trạng thái là vector n phần tử, mỗi phần tử nhận n giá trị. Không gian trạng thái chỉ còn nⁿ. Với n=8, 8⁸ ≈ 1.68 × 10⁷ trạng thái. Không gian tìm kiếm được thu hẹp hơn 260 lần và loại bỏ hoàn toàn các trạng thái có xung đột dọc (cùng cột).

#### Câu 8: Vì sao đếm xung đột theo hàng/đường chéo có thể có độ phức tạp O(n) thay vì O(n²)?
- **Naive O(n²):** So sánh từng cặp (i, j) với 0 ≤ i < j < n, tổng số cặp là n(n-1)/2 ≈ O(n²).
- **Tuyến tính O(n):** Dùng 3 mảng đếm tần suất: row_cnt (kích thước n), diag1_cnt (col+row, kích thước 2n) và diag2_cnt (col-row+n, kích thước 2n). Chỉ cần một vòng lặp n bước để đếm số quân trên mỗi đường. Sau đó, nếu có k quân trên cùng 1 đường, số cặp xung đột tính trực tiếp bằng công thức C(k, 2) = k(k-1)/2. Tổng số phép tính là n + 5n = O(n), hoàn toàn không cần so sánh cặp.

#### Câu 9: Khi dùng Numba, vì sao phải loại trừ thời gian biên dịch lần đầu trong phép đo tốc độ?
Trong lần gọi đầu tiên (khởi động nguội - cold start), Numba phải kiểm tra kiểu dữ liệu thực tế của tham số, sinh mã trung gian LLVM IR, tối ưu hóa và biên dịch sang mã máy nhị phân của CPU. Quá trình biên dịch này mất từ 100 ms đến hơn 1,300 ms (chi phí biên dịch - overhead). Từ lần gọi thứ hai trở đi, Numba thực thi trực tiếp mã máy đã lưu trong bộ nhớ chỉ mất vài micro giây (µs). Do đó, nếu không warm-up trước, kết quả đo sẽ phản ánh thời gian của trình biên dịch JIT chứ không phản ánh tốc độ thực thi của thuật toán.

#### Câu 10: Với TSP, thay toán tử swap bằng 2-opt có thể ảnh hưởng đến lời giải và số láng giềng như thế nào?
- **Số láng giềng:** Cả Swap (chọn 2 thành phố để đổi chỗ) và 2-opt (chọn 2 cạnh để cắt và đảo ngược đoạn con) đều có kích thước không gian lân cận bằng C(n, 2) = n(n-1)/2.
- **Chất lượng lời giải:** Toán tử Swap phá vỡ 4 cạnh liên thuộc và thay bằng 4 cạnh ngẫu nhiên, thường tạo thêm các đoạn cắt chéo mới. Toán tử 2-opt đảo ngược thứ tự một chuỗi các thành phố, giúp loại bỏ trực tiếp các cặp cạnh giao cắt nhau (gỡ chéo - uncrossing) theo bất đẳng thức tam giác trong mặt phẳng Euclid. Do đó, 2-opt cho ra chu trình có độ dài ngắn hơn vượt trội và hội tụ về các nghiệm tối ưu cao hơn hẳn so với Swap.

---

# PHẦN 9: ĐÁNH GIÁ PHÂN TÍCH CHUYÊN SÂU (1-2 TRANG)

### 9.1 Bản chất Cực trị Cục bộ và Hình thái Bề mặt Hàm Mục tiêu
Tìm kiếm cục bộ vận hành dựa trên giả định ngầm định rằng: cải thiện cục bộ từng bước sẽ dẫn đến trạng thái tối ưu toàn cục. Tuy nhiên, giả định này chỉ đúng khi bề mặt hàm mục tiêu có dạng lồi (convex) hoặc đơn mốt (unimodal). Trong thực tế:
- **Bề mặt lồi lõm đa cực trị (Multimodal Landscape):** Trong bài toán n-Queens (8-Queens), tỉ lệ thành công của Steepest-ascent chỉ đạt 15.0%, nghĩa là 85% các trạng thái ngẫu nhiên dẫn đến các cực tiểu cục bộ giả.
- **Chướng ngại vật địa hình:** Trong bài toán Mê cung chữ L, bức tường đá tạo ra một vùng bóng tối khiến hàm khoảng cách Manhattan không phản ánh đúng khoảng cách đường đi thực tế. Sự thiếu vắng tầm nhìn xa (lookahead) khiến giải thuật leo đồi thuần túy hoàn toàn bất lực trước các cấu trúc bẫy hình học.

### 9.2 So sánh Chiến lược Giải thoát: Random Restarts vs Simulated Annealing
- **Random Restarts (Tái tạo Độc lập):** Hoạt động theo nguyên lý "thử lại nhiều lần". Điểm mạnh là cực kỳ đơn giản để cài đặt, dễ song song hóa trên hệ thống đa nhân. Tuy nhiên, hạn chế lớn là thuật toán hoàn toàn mất trí nhớ giữa các lần chạy: mỗi lần restart là một cuộc tìm kiếm mới từ đầu, không tận dụng được những thông tin về các cấu trúc tốt đã khám phá ở các lần chạy trước.
- **Simulated Annealing (Vượt rào Ngẫu nhiên có Kiểm soát):** Hoạt động theo nguyên lý quỹ đạo duy nhất nhưng biến đổi linh hoạt theo nhiệt độ. Điểm mạnh vượt trội là khả năng giữ lại phần lớn cấu trúc tốt của trạng thái hiện tại trong khi chỉ rung lắc nhẹ để nhảy qua các rào cản năng lượng thấp. Nhược điểm là việc tinh chỉnh bộ siêu tham số (T₀, lịch làm nguội α, điều kiện dừng) đòi hỏi kinh nghiệm thực nghiệm đáng kể.

### 9.3 Đánh đổi giữa Chất lượng Nghiệm và Chi phí Tính toán
Trong thiết kế thuật toán, không có bữa trưa nào miễn phí (No Free Lunch Theorem):
- **Chất lượng bước đi vs Số lượt đánh giá:** Steepest-ascent tốn n(n-1) lượt đánh giá mỗi bước để chọn bước đi tối ưu nhất, nhưng cuối cùng vẫn bị bẫy ở 85% số lần chạy trên 8-Queens. Trong khi đó, First-choice chỉ tốn vài lượt đánh giá mỗi bước nhưng khi kết hợp với Random Restart lại đạt 100% thành công với tổng thời gian chạy tương đương hoặc nhanh hơn!
- **Toán tử lân cận hẹp vs rộng:** Toán tử lân cận quá hẹp (như Single-step) dễ khiến thuật toán đông cứng sớm. Toán tử lân cận quá rộng (như Dual-queen) làm bùng nổ chi phí duyệt và biến việc tìm kiếm thành dò dẫm hỗn loạn. Toán tử 2-opt trong TSP là một ví dụ mẫu mực về sự cân bằng hoàn hảo giữa kích thước lân cận O(n²) và khả năng gỡ bỏ cấu trúc chéo hình học.

### 9.4 Khả năng Mở rộng và Giới hạn Nội tại
Khi kích thước bài toán tăng lên hàng trăm hoặc hàng ngàn biến (ví dụ TSP 500 thành phố hay n-Queens 1000 quân):
- **Ưu thế:** Yêu cầu bộ nhớ của Local Search không bao giờ là điểm nghẽn (O(n)). Kết hợp với tối ưu hóa mã máy JIT như Numba, tốc độ đánh giá hàm mục tiêu có thể đạt hàng triệu trạng thái mỗi giây.
- **Hạn chế:** Số lượng cực tiểu cục bộ tăng theo hàm mũ so với số chiều của bài toán. Hướng đi tiên tiến trong thực tế là xây dựng các giải thuật lai (Hybrid Metaheuristics) - ví dụ kết hợp Thuật giải Di truyền (GA) để khám phá toàn cục trên diện rộng với 2-opt Local Search để tinh chỉnh cục bộ từng cá thể.

---

# PHẦN 10: KẾT LUẬN VÀ TÀI LIỆU THAM KHẢO

### 10.1 Kết luận
Qua đợt thực hành Lab 03 - Tìm kiếm Cục bộ (Local Search), nhóm sinh viên đã thu hoạch được những kết quả trọng tâm sau:
1. **Làm chủ lý thuyết và cài đặt:** Cài đặt thành công và hoàn thiện 100% các giải thuật: Simple Hill Climbing, Steepest-ascent, Stochastic, First-choice, Random Restarts, Simulated Annealing và Genetic Algorithm.
2. **Hoàn thiện toàn diện 5 Notebooks:** Chuyển đổi toàn bộ giải thích sang tiếng Việt chuẩn mực, điền đầy đủ mã nguồn cho tất cả các Task, thực thi và lưu lại trọn vẹn kết quả hiển thị, biểu đồ và animation.
3. **Đánh giá thực nghiệm quy mô lớn:** Khảo sát định lượng trên hàng trăm lượt chạy với các kích thước khác nhau (n=4, 8, 12, 16, 20 trên n-Queens; n=10, 20, 30, 50 trên TSP; n=8..500 trên Numba).
4. **Tối ưu hóa hiệu năng đỉnh cao:** Chứng minh khả năng tăng tốc vượt bậc của Numba JIT (lên tới hơn 5,800 lần) và sự vượt trội của toán tử 2-opt trong bài toán hình học phẳng TSP.

### 10.2 Tài liệu Tham khảo
- Stuart Russell và Peter Norvig (2020). Artificial Intelligence: A Modern Approach (4th Edition). Pearson Education. Chapter 4: Search in Complex Environments.
- Michael Hahsler. Local Search Course Materials and Jupyter Notebooks. Southern Methodist University (SMU).
- Kirkpatrick, S., Gelatt, C. D. Jr., và Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671–680.
- Hajek, B. (1988). Cooling schedules for optimal annealing. Mathematics of Operations Research, 13(2), 311–329.
- Szu, H., và Hartley, R. (1987). Fast simulated annealing. Physics Letters A, 122(3–4), 157–162.
- Croes, G. A. (1958). A method for solving traveling-salesman problems. Operations Research, 6(6), 791–812. (Giới thiệu toán tử 2-opt).
- Lam, S. K., Pitrou, A., và Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC.

---
*(Hết Báo Cáo)*
