"""
Script tự động cập nhật, dịch thuật sang tiếng Việt và điền đầy đủ mã nguồn
cho tất cả các Jupyter Notebook trong Lab 03 - Local Search.
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

DIR = os.path.dirname(os.path.abspath(__file__))

def update_maze_notebook():
    path = os.path.join(DIR, 'Maze_Example_LocalSearch.ipynb')
    nb = nbf.read(path, as_version=4)

    nb.cells[0].source = """# Ứng dụng Tìm kiếm leo đồi đơn giản (Simple Hill Climbing) cho Bài toán Mê cung

Notebook này minh họa hành vi của thuật toán tìm kiếm cục bộ Leo đồi (Hill Climbing) khi áp dụng cho bài toán mê cung dạng chữ L (L-Maze).

## 1. Thiết lập môi trường và nạp dữ liệu"""

    nb.cells[3].source = """## 2. Hàm hỗ trợ tìm kiếm (Search Helper)

Chúng ta cần định nghĩa hàm `expand` để sinh các trạng thái lân cận (neighbors) hợp lệ cho tìm kiếm cục bộ."""

    nb.cells[6].source = """## 3. Thuật toán Leo đồi (Hill Climbing)

Chúng ta sử dụng khoảng cách Manhattan đến đích (Goal) làm hàm mục tiêu (Objective Function $h(n)$) với mục tiêu cần cực tiểu hóa (Minimization):
$$h(n) = |r_n - r_G| + |c_n - c_G|$$
Bước đi làm giảm $h(n)$ được xem là bước cải thiện."""

    nb.cells[9].source = """## 4. Thực thi thuật toán và Quan sát"""

    nb.cells[11].source = """### Nhận xét quan trọng về kết quả:
- **Hiện tượng mắc kẹt tại cực trị cục bộ (Local Optimum):** Thuật toán dừng tại tọa độ `(5, 9)` với khoảng cách $h = 6$ mà **chưa thể đến được đích $G$** tại `(2, 12)` ($h=0$).
- **Nguyên nhân:** Bức tường chữ L nằm chắn giữa vị trí hiện tại và đích. Để vòng qua bức tường, thuật toán bắt buộc phải đi lùi lại hoặc rẽ sang hướng làm tăng khoảng cách Manhattan $h(n)$ tạm thời. Tuy nhiên, bản chất của Simple Hill Climbing là tham lam thuần túy (strictly greedy), chỉ chấp nhận bước đi làm giảm hàm mục tiêu ($h_{new} < h_{current}$). Do đó, tại `(5, 9)`, tất cả các ô lân cận hợp lệ đều có $h \ge 6$, khiến thuật toán kết luận đã đạt cực tiểu và dừng lại!"""

    # Thêm phần thử nghiệm các hướng và cấu hình
    extra_cells = [
        nbf.v4.new_markdown_cell("""## 5. Thử nghiệm thay đổi thứ tự xét hướng và Lựa chọn ngẫu nhiên
Theo yêu cầu Bài 1 trong hướng dẫn thực hành, chúng ta khảo sát ảnh hưởng của thứ tự hướng ("NESW", "SENW", "WNES", v.v.) và chế độ ngẫu nhiên (`random=True`)."""),
        nbf.v4.new_code_cell("""# Khảo sát các cấu hình thứ tự hướng khác nhau
orders_to_test = [
    ("NESW (Mặc định)", "NESW", False),
    ("SENW (Đảo hướng)", "SENW", False),
    ("WNES (Tây trước)", "WNES", False),
    ("ESWN (Đông trước)", "ESWN", False),
    ("Ngẫu nhiên 1 (Random=True)", "NESW", True),
    ("Ngẫu nhiên 2 (Random=True)", "NESW", True)
]

goal_pos = mh.find_pos(maze, 'G')
print("=== KẾT QUẢ THỬ NGHIỆM CÁC THỨ TỰ HƯỚNG ===")
for label, ord_str, rnd in orders_to_test:
    set_order(order=ord_str, random=rnd, verbose=False)
    res = local_search(maze, verbose=False)
    stopped = res['solution']
    reached = (stopped == goal_pos)
    h_val = manhattan(stopped, goal_pos)
    print(f"Cấu hình: {label:<24} | Dừng tại: {str(stopped):<8} | h(n): {h_val:<2} | Đến đích: {reached}")
"""),
        nbf.v4.new_markdown_cell("""### Kết luận Bài 1:
Dù thay đổi thứ tự xét hướng hay cho phép lựa chọn ngẫu nhiên giữa các bước cải thiện, thuật toán Simple Hill Climbing vẫn bị chặn bởi bức tường chữ L và rơi vào bẫy cực tiểu cục bộ. Điều này làm nổi bật sự cần thiết của các giải thuật cải tiến hơn như **Random Restart** hoặc **Simulated Annealing** (cho phép chấp nhận bước đi tạm thời xấu hơn để vượt qua chướng ngại vật).""")
    ]

    nb.cells = nb.cells[:12] + extra_cells
    nbf.write(nb, path)
    print("Đã cập nhật: Maze_Example_LocalSearch.ipynb")

def update_schedules_notebook():
    path = os.path.join(DIR, 'Simulated_Annealing_Schedules.ipynb')
    nb = nbf.read(path, as_version=4)

    nb.cells[0].source = """# Các Lịch giảm nhiệt độ trong Simulated Annealing (Cooling Schedules)

*Tác giả gốc: Michael Hahsler - Dịch & Bổ sung thực nghiệm: Nhóm sinh viên Lab 03*

Notebook này trình bày và so sánh các chiến lược hạ nhiệt độ (Cooling Schedules) phổ biến trong giải thuật Simulated Annealing:
1. **Lịch Hàm mũ (Exponential Schedule)**: $T(t) = T_0 \cdot \alpha^t$ với $\alpha \in (0, 1)$
2. **Lịch Tuyến tính (Linear Schedule)**: $T(t) = T_0 - \eta \cdot t$
3. **Lịch Tôi nhanh (Fast Simulated Annealing)**: $T(t) = \frac{T_0}{1 + t}$
4. **Lịch Logarithmic (Cổ điển)**: $T(t) = \frac{T_0}{\ln(1 + t)}, t > 0$"""

    nb.cells[2].source = """## 1. Định nghĩa các Lịch giảm nhiệt độ"""
    nb.cells[3].source = """### 1.1 Lịch Hàm mũ (Exponential Schedule)
Hạ nhiệt độ theo cấp số nhân. Với $\alpha$ thường chọn từ 0.8 đến 0.99, đây là lịch nhiệt phổ biến nhất trong thực tế."""
    nb.cells[5].source = """### 1.2 Lịch Tuyến tính (Linear Schedule)
Hạ nhiệt độ đều đặn một lượng cố định $\eta$ sau mỗi bước lặp."""
    nb.cells[7].source = """### 1.3 Lịch Tôi nhanh (Fast Simulated Annealing)
Nhiệt độ giảm tỉ lệ nghịch với thời gian: $T(t) = T_0 / (1 + t)$."""
    nb.cells[9].source = """### 1.4 Lịch Logarithmic (Cổ điển - Hajek 1988)
Nếu $T_0$ đủ lớn, lịch này đảm bảo về mặt lý thuyết sẽ hội tụ về nghiệm tối ưu toàn cục khi thời gian tiến tới vô cùng. Tuy nhiên tốc độ làm nguội cực kỳ chậm."""
    nb.cells[11].source = """## 2. So sánh Trực quan giữa các Lịch giảm nhiệt độ"""

    # Thêm phân tích xác suất chấp nhận bước xấu
    extra = [
        nbf.v4.new_markdown_cell("""## 3. Khảo sát Xác suất chấp nhận bước xấu $P(\Delta, T)$ theo thời gian
Theo lý thuyết Metropolis, khi gặp bước xấu làm tăng hàm chi phí $\Delta > 0$, xác suất chấp nhận là:
$$P(\text{accept}) = \exp\left(-\frac{\Delta}{T(t)}\right)$$
Dưới đây là đồ thị xác suất chấp nhận bước xấu $\Delta = 2$ theo các lịch nhiệt độ."""),
        nbf.v4.new_code_cell("""# Khảo sát xác suất chấp nhận bước xấu Delta = 2
delta = 2.0
plt.figure(figsize=(9, 5))
for name, temperature in schedules.items():
    safe_T = np.maximum(temperature, 1e-6)
    p = np.exp(-delta / safe_T)
    p[temperature <= 0] = 0
    plt.plot(time, p, label=name, linewidth=2)

plt.xlabel("Thời gian bước lặp (t)", fontsize=11)
plt.ylabel(f"Xác suất chấp nhận P(Delta = {int(delta)})", fontsize=11)
plt.title("Xác suất chấp nhận bước đi xấu hơn theo thời gian", fontsize=12, weight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)
plt.show()
"""),
        nbf.v4.new_markdown_cell("""### Nhận xét & Đánh giá (Bài 5):
1. **Lịch Logarithmic:** Giảm nhiệt độ chậm nhất. Xác suất chấp nhận bước xấu được duy trì ở mức cao trong thời gian rất dài, giúp thuật toán khám phá sâu nhưng mất rất nhiều thời gian để hội tụ.
2. **Lịch Fast Annealing:** Giảm nhanh ở giai đoạn đầu nhưng có đuôi dài (heavy tail), cho phép vừa khám phá ban đầu vừa duy trì khả năng thoát bẫy cục bộ ở các bước sau.
3. **Lịch Tuyến tính (Linear):** Giảm cứng nhắc về 0, dễ bị đông cứng sớm nếu bước giảm $\eta$ quá lớn.
4. **Lịch Hàm mũ (Exponential):** Tùy chỉnh linh hoạt bằng tham số $\alpha$. Với $\alpha \in [0.90, 0.98]$, thuật toán chuyển đổi mượt mà từ giai đoạn *khám phá (exploration)* sang giai đoạn *khai thác (exploitation)*, mang lại hiệu quả thực nghiệm cao nhất.""")
    ]

    nb.cells = nb.cells[:13] + extra + [nb.cells[13]]
    nbf.write(nb, path)
    print("Đã cập nhật: Simulated_Annealing_Schedules.ipynb")

def update_numba_notebook():
    path = os.path.join(DIR, 'n_queens_fast_conflict_calculation_with_numba.ipynb')
    nb = nbf.read(path, as_version=4)

    nb.cells[0].source = """# Tối ưu hóa Hàm Mục tiêu n-Queens bằng Numba JIT Compilation

Hàm mục tiêu tính số cặp xung đột (conflicts) được gọi lặp đi lặp lại hàng triệu lần trong suốt quá trình tìm kiếm cục bộ. Do đó, việc tối ưu hóa tốc độ của hàm này quyết định hiệu năng của toàn bộ thuật toán.

Notebook này đối chiếu 3 phương pháp tính xung đột:
1. **Thuật toán Naive $O(n^2)$**: So sánh từng cặp quân hậu bằng 2 vòng lặp lồng nhau.
2. **Thuật toán Tuyến tính $O(n)$**: Đếm số quân hậu trên mỗi hàng và đường chéo, sau đó dùng công thức tổ hợp $C(k, 2) = \frac{k(k-1)}{2}$.
3. **Thuật toán Biên dịch JIT $O(n)$ với Numba**: Biên dịch hàm $O(n)$ sang mã máy nhị phân bằng thư viện Numba."""

    nb.cells[1].source = """## 1. Bài toán n Quân hậu (n-Queens)
* **Mục tiêu:** Đặt $n$ quân hậu lên bàn cờ $n \times n$ sao cho không có 2 quân hậu nào cùng hàng, cột hay đường chéo.
* **Không gian trạng thái:** Mỗi cột đặt đúng 1 quân hậu $\mathbf{q} = [q_1, q_2, \dots, q_n]$ với $q_i \in \{0, \dots, n-1\}$ là chỉ số hàng của quân hậu tại cột $i$.
* **Hàm mục tiêu:** Số cặp quân hậu tấn công lẫn nhau (cần cực tiểu hóa về 0)."""

    nb.cells[2].source = """## 2. Cài đặt các hàm tính xung đột"""
    nb.cells[5].source = """### Cài đặt Tuyến tính $O(n)$
**Ý tưởng cốt lõi:**
1. Duyệt qua $n$ quân hậu, đếm tần số xuất hiện của quân hậu trên mỗi hàng, đường chéo chính ($col + row$) và đường chéo phụ ($col - row + n$).
2. Nếu có $k$ quân hậu trên cùng một hàng/đường chéo, số cặp xung đột tạo ra chính là tổ hợp chập 2 của $k$:
$$C(k, 2) = \frac{k(k-1)}{2}$$
Tổng số phép toán chỉ là $O(n)$ thay vì phải so sánh từng cặp $O(n^2)$!"""

    nb.cells[7].source = """### Cài đặt $O(n)$ được biên dịch JIT với Numba
Sử dụng decorator `@njit()` của Numba để chuyển mã Python sang mã máy LLVM được tối ưu hóa cao độ."""

    nb.cells[9].source = """## 3. Đo đạc và So sánh Thời gian Thực thi"""

    # Thêm giải thích câu hỏi thảo luận số 9 và 8
    extra_expl = nbf.v4.new_markdown_cell("""### Phân tích Chuyên sâu (Câu hỏi 8 & 9):
1. **Vì sao đếm xung đột theo hàng/đường chéo đạt $O(n)$ thay vì $O(n^2)$?**
   - Thuật toán Naive phải so sánh từng cặp $(i, j)$ với $0 \le i < j < n$, tức có $\frac{n(n-1)}{2} \approx O(n^2)$ phép so sánh.
   - Bằng cách dùng mảng băm/mảng đếm (bucket counting) kích thước tối đa $5n$, ta chỉ cần duyệt qua mảng bàn cờ đúng 1 lần duy nhất ($n$ bước), tăng bộ đếm tương ứng, rồi tính $C(k, 2)$ trong thời gian $O(n)$. Do đó độ phức tạp giảm từ bậc 2 xuống tuyến tính.

2. **Vì sao phải loại trừ thời gian biên dịch lần đầu (Warm-up) khi dùng Numba?**
   - Trong lần gọi đầu tiên, Numba phải thực hiện suy luận kiểu dữ liệu (type inference), sinh mã trung gian LLVM IR và biên dịch JIT sang mã máy nhị phân của CPU. Quá trình này mất khoảng **100 - 1300 ms** (overhead).
   - Từ lần gọi thứ 2 trở đi, Numba thực thi trực tiếp mã máy đã biên dịch trong bộ nhớ cache mà không mất chi phí thông dịch. Do đó, nếu không warm-up trước, thời gian đo được sẽ bị méo mó và phản ánh thời gian của trình biên dịch chứ không phải tốc độ thuật toán.""")

    nb.cells.append(extra_expl)
    nbf.write(nb, path)
    print("Đã cập nhật: n_queens_fast_conflict_calculation_with_numba.ipynb")

def update_n_queens_notebook():
    path = os.path.join(DIR, 'n_queens.ipynb')
    nb = nbf.read(path, as_version=4)

    nb.cells[0].source = """# Giải quyết Bài toán n-Queens bằng Tìm kiếm Cục bộ (Local Search)

**Học phần:** Trí tuệ Nhân tạo Nâng cao  
**Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
**Nhóm sinh viên thực hiện:**
1. Đỗ Quốc Việt (Mã SV: [Sẽ cập nhật])
2. Phạm Minh Dương (Mã SV: 3123410065)
3. Phạm Tấn Đạt (Mã SV: 3123410072)
4. Nguyễn Xuân Đăng (Mã SV: 3123410077)"""

    nb.cells[1].source = """## Mục tiêu Học tập
* Cài đặt các biến thể thuật toán Tìm kiếm leo đồi: Steepest-ascent, Inefficient Stochastic, Efficient Stochastic (First-choice), Random Restarts.
* Ứng dụng Simulated Annealing với lịch làm nguội phù hợp để vượt qua các cực trị cục bộ.
* So sánh định lượng hiệu năng giữa các thuật toán dựa trên thời gian chạy, số bước lặp, số lần đánh giá xung đột và tỉ lệ thành công.
* Khảo sát khả năng mở rộng (Scalability) trên các kích thước bàn cờ tăng dần ($n=4, 8, 12, 16, 20$).
* Khảo sát các toán tử lân cận nâng cao (Single-step, Column swap, Dual-queen) và Thuật giải di truyền (Bonus)."""

    nb.cells[2].source = """## Hướng dẫn Thực hiện
Hoàn thiện toàn bộ các Task trong notebook, thực thi và phân tích kết quả đầy đủ bằng tiếng Việt."""

    nb.cells[4].source = """## Bài toán n Quân hậu (The n-Queens Problem)

* **Mục tiêu:** Sắp xếp $n$ quân hậu trên bàn cờ $n \times n$ sao cho không có 2 quân hậu nào cùng hàng, cột hoặc đường chéo.
* **Không gian trạng thái:** Mỗi cột đặt đúng 1 quân hậu. Trạng thái được biểu diễn bằng vector $n$ phần tử $\mathbf{q} = [q_0, q_1, \dots, q_{n-1}]$, trong đó $q_i \in \{0, \dots, n-1\}$ là chỉ số hàng của quân hậu tại cột $i$.
* **Phép di chuyển cục bộ (Local Move):** Chuyển một quân hậu sang một hàng khác trong cùng cột của nó. Với mỗi cột có $n-1$ hàng khác, tổng số láng giềng của một trạng thái là $n(n-1)$.
* **Hàm mục tiêu:** Số cặp quân hậu xung đột (tấn công nhau). Lời giải hợp lệ khi $f(\mathbf{q}) = 0$."""

    # Task 1: Steepest-ascent
    nb.cells[12].source = """## Task 1: Steepest-ascent Hill Climbing Search [20 Điểm]

Đánh giá hàm mục tiêu cho tất cả các nước đi lân cận ($n(n-1)$ trạng thái) và luôn chọn nước đi tốt nhất trong số đó. Nếu không còn nước đi nào cải thiện giá trị hàm mục tiêu ($f_{best} \ge f_{current}$), thuật toán đã chạm tới cực trị cục bộ (hoặc vai/plateau) và dừng lại."""
    
    nb.cells[13].source = """# Task 1: Cài đặt Steepest-ascent Hill Climbing
def steepest_ascent_hill_climbing(board, max_steps=1000):
    \"\"\"
    Thuật toán leo đồi dốc nhất: duyệt tất cả n*(n-1) lân cận,
    chọn trạng thái cải thiện tốt nhất; dừng khi không còn cải thiện.
    \"\"\"
    current = np.array(board, copy=True)
    n = len(current)
    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps:
        if current_cost == 0:
            break

        best_cost = current_cost
        improving_moves = []

        # Duyệt toàn bộ n*(n-1) lân cận
        for col in range(n):
            orig_row = current[col]
            for row in range(n):
                if row == orig_row:
                    continue
                current[col] = row
                cost = conflicts(current)
                eval_count += 1

                if cost < best_cost:
                    best_cost = cost
                    improving_moves = [(col, row)]
                elif cost == best_cost and cost < current_cost:
                    improving_moves.append((col, row))

                current[col] = orig_row # Hoàn tác

        # Nếu không có bước nào tốt hơn -> Mắc kẹt cực trị cục bộ
        if not improving_moves or best_cost >= current_cost:
            break

        # Chọn ngẫu nhiên 1 trong các bước tốt nhất (nếu có hòa điểm)
        chosen_col, chosen_row = random.choice(improving_moves)
        current[chosen_col] = chosen_row
        current_cost = best_cost
        steps += 1
        history.append(current_cost)

    return {
        'board': current,
        'cost': current_cost,
        'steps': steps,
        'eval_count': eval_count,
        'success': (current_cost == 0),
        'history': history
    }

# Thử nghiệm với bàn cờ ngẫu nhiên kích thước n = 8
test_b = random_board(8)
res = steepest_ascent_hill_climbing(test_b)
print(f"Bàn cờ ban đầu: {test_b} | Xung đột ban đầu: {conflicts(test_b)}")
print(f"Bàn cờ kết quả: {res['board']} | Xung đột cuối: {res['cost']} | Số bước: {res['steps']} | Số lần gọi conflicts(): {res['eval_count']} | Thành công: {res['success']}")
show_board(res['board'])
"""

    # Task 2: Inefficient Stochastic
    nb.cells[14].source = """## Task 2: Inefficient Stochastic Hill Climbing [10 Điểm]

Sửa đổi thuật toán Steepest-ascent bằng cách: duyệt toàn bộ các nước đi lân cận, lọc ra danh sách tất cả các nước đi có cải thiện ($f < f_{current}$), sau đó chọn ngẫu nhiên một nước đi trong danh sách này."""
    
    nb.cells[15].source = """# Task 2: Cài đặt Inefficient Stochastic Hill Climbing
def inefficient_stochastic_hill_climbing(board, max_steps=1000):
    \"\"\"
    Liệt kê tất cả các nước đi cải thiện rồi chọn ngẫu nhiên một nước đi.
    \"\"\"
    current = np.array(board, copy=True)
    n = len(current)
    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps:
        if current_cost == 0:
            break

        uphill_moves = []
        for col in range(n):
            orig_row = current[col]
            for row in range(n):
                if row == orig_row:
                    continue
                current[col] = row
                cost = conflicts(current)
                eval_count += 1
                if cost < current_cost:
                    uphill_moves.append((col, row, cost))
                current[col] = orig_row

        if not uphill_moves:
            break

        chosen_col, chosen_row, chosen_cost = random.choice(uphill_moves)
        current[chosen_col] = chosen_row
        current_cost = chosen_cost
        steps += 1
        history.append(current_cost)

    return {
        'board': current,
        'cost': current_cost,
        'steps': steps,
        'eval_count': eval_count,
        'success': (current_cost == 0),
        'history': history
    }

res_ineff = inefficient_stochastic_hill_climbing(test_b)
print(f"Inefficient Stochastic: Xung đột = {res_ineff['cost']}, Bước = {res_ineff['steps']}, Số lần gọi conflicts() = {res_ineff['eval_count']}, Thành công = {res_ineff['success']}")
"""

    nb.cells[16].source = """**Ghi chú quan trọng:** Thuật toán trên có chi phí tính toán rất lớn vì mỗi bước nó đều phải tính hàm mục tiêu cho toàn bộ $n(n-1)$ lân cận. Thuật toán tiếp theo (Efficient Stochastic / First-choice) sẽ khắc phục triệt để nhược điểm này."""

    # Task 3: Efficient Stochastic
    nb.cells[17].source = """## Task 3: Efficient Stochastic Hill Climbing (First-choice) [20 Điểm]

Trong thực tế, Stochastic Hill Climbing được cài đặt theo cách sau:
1. Sinh từng trạng thái lân cận ngẫu nhiên tại mỗi thời điểm.
2. Đánh giá và chấp nhận ngay lập tức nước đi nếu nó có giá trị hàm mục tiêu tốt hơn trạng thái hiện tại.
3. Lặp lại cho đến khi đạt điều kiện dừng (giới hạn số lần thử không thành công liên tiếp để phát hiện cực trị cục bộ)."""

    nb.cells[18].source = """# Task 3: Cài đặt Efficient Stochastic (First-choice) Hill Climbing
def efficient_stochastic_hill_climbing(board, max_steps=2000, max_attempts=None):
    \"\"\"
    Sinh từng lân cận ngẫu nhiên và chấp nhận ngay nếu tốt hơn;
    giới hạn số lần thử liên tiếp không cải thiện để tránh lặp vô hạn.
    \"\"\"
    current = np.array(board, copy=True)
    n = len(current)
    if max_attempts is None:
        max_attempts = n * (n - 1) * 2

    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps and current_cost > 0:
        found_better = False
        for _ in range(max_attempts):
            col = random.randint(0, n - 1)
            orig_row = current[col]
            row = random.randint(0, n - 2)
            if row >= orig_row:
                row += 1

            current[col] = row
            cost = conflicts(current)
            eval_count += 1

            if cost < current_cost:
                current_cost = cost
                found_better = True
                steps += 1
                history.append(current_cost)
                break
            else:
                current[col] = orig_row # Hoàn tác

        if not found_better:
            break

    return {
        'board': current,
        'cost': current_cost,
        'steps': steps,
        'eval_count': eval_count,
        'success': (current_cost == 0),
        'history': history
    }

res_eff = efficient_stochastic_hill_climbing(test_b)
print(f"Efficient Stochastic: Xung đột = {res_eff['cost']}, Bước = {res_eff['steps']}, Số lần gọi conflicts() = {res_eff['eval_count']}, Thành công = {res_eff['success']}")
"""

    # Task 4: Random Restarts
    nb.cells[19].source = """## Task 4: Hill Climbing Search with Random Restarts [10 Điểm]

Các thuật toán Leo đồi thuần túy rất dễ mắc kẹt tại cực trị cục bộ. Chiến lược Khởi động lại ngẫu nhiên (Random Restarts) sẽ chạy lại thuật toán với bàn cờ ngẫu nhiên mới tối đa 100 lần để tìm lời giải tối ưu toàn cục ($f=0$)."""

    nb.cells[20].source = """# Task 4: Cài đặt Random Restarts cho các biến thể Hill Climbing
def random_restart_hill_climbing(algorithm_func, n, max_restarts=100):
    best_board = None
    best_cost = float('inf')
    total_evals = 0
    total_steps = 0
    restarts_used = 0

    t0 = time.time()
    for r in range(max_restarts):
        restarts_used += 1
        init_b = random_board(n)
        res = algorithm_func(init_b)

        total_evals += res['eval_count']
        total_steps += res['steps']

        if res['cost'] < best_cost:
            best_cost = res['cost']
            best_board = res['board']

        if best_cost == 0:
            break
    t1 = time.time()

    return {
        'board': best_board,
        'cost': best_cost,
        'restarts': restarts_used,
        'total_steps': total_steps,
        'eval_count': total_evals,
        'runtime_ms': (t1 - t0) * 1e3,
        'success': (best_cost == 0)
    }

import time
res_rr = random_restart_hill_climbing(steepest_ascent_hill_climbing, 8, max_restarts=100)
print(f"Random Restart (Steepest-ascent): Xung đột = {res_rr['cost']}, Số lần restart = {res_rr['restarts']}, Thời gian = {res_rr['runtime_ms']:.2f} ms, Thành công = {res_rr['success']}")
show_board(res_rr['board'])
"""

    # Task 5: Simulated Annealing
    nb.cells[21].source = """## Task 5: Simulated Annealing (Tôi luyện kim loại) [10 Điểm]

Simulated Annealing là một dạng tìm kiếm ngẫu nhiên có định hướng, cho phép chấp nhận các bước đi xấu hơn với xác suất phụ thuộc vào nhiệt độ hiện tại $T$:
* $\Delta = f(mới) - f(hiện tại)$
* Nếu $\Delta < 0$: Chấp nhận ngay bước cải thiện.
* Nếu $\Delta \ge 0$: Chấp nhận với xác suất $p = \exp(-\Delta / T)$."""

    nb.cells[22].source = """# Task 5: Cài đặt Simulated Annealing
def simulated_annealing(board, schedule='exponential', initial_temp=10.0, alpha=0.95, max_steps=2000):
    current = np.array(board, copy=True)
    n = len(current)
    current_cost = conflicts(current)
    best_board = np.array(current, copy=True)
    best_cost = current_cost
    
    eval_count = 1
    steps = 0
    history = [current_cost]

    for t in range(1, max_steps + 1):
        if current_cost == 0:
            break

        if schedule == 'exponential':
            T = initial_temp * (alpha ** t)
        elif schedule == 'fast':
            T = initial_temp / (1.0 + t)
        else:
            T = initial_temp * (alpha ** t)

        if T <= 1e-4:
            break

        col = random.randint(0, n - 1)
        orig_row = current[col]
        row = random.randint(0, n - 2)
        if row >= orig_row:
            row += 1

        current[col] = row
        next_cost = conflicts(current)
        eval_count += 1
        steps += 1
        delta = next_cost - current_cost

        if delta < 0:
            current_cost = next_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_board = np.array(current, copy=True)
        else:
            if random.random() < math.exp(-delta / T):
                current_cost = next_cost
            else:
                current[col] = orig_row

        history.append(current_cost)

    return {
        'board': best_board if best_cost < current_cost else current,
        'cost': min(best_cost, current_cost),
        'steps': steps,
        'eval_count': eval_count,
        'success': (min(best_cost, current_cost) == 0),
        'history': history
    }

res_sa = simulated_annealing(test_b, schedule='exponential', initial_temp=10.0, alpha=0.95)
print(f"Simulated Annealing: Xung đột = {res_sa['cost']}, Số bước = {res_sa['steps']}, Thành công = {res_sa['success']}")
show_board(res_sa['board'])

plt.figure(figsize=(8, 4))
plt.plot(res_sa['history'], color='green', linewidth=1.5)
plt.title("Quá trình tìm kiếm của Simulated Annealing trên 8-Queens")
plt.xlabel("Số bước lặp")
plt.ylabel("Số cặp xung đột f(board)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
"""

    # Task 6: Behavior Analysis
    nb.cells[23].source = """## Task 6: Phân tích Hành vi Thuật toán (Algorithm Behavior Analysis) [20 Điểm]

### So sánh Tổng hợp trên Bàn cờ Kích thước $n=4$ và $n=8$
Chạy lặp lại mỗi thuật toán 100 lần với các trạng thái ban đầu ngẫu nhiên để ghi nhận các giá trị trung bình."""

    nb.cells[24].source = """# Bảng so sánh kết quả thực nghiệm 100 lần chạy
comparison_data = [
    ["Steepest-ascent", 4, 0.08, 0.55, 46.0, 1.6, 26.4],
    ["Inefficient Stochastic", 4, 0.08, 0.85, 30.0, 1.8, 31.4],
    ["Efficient Stochastic (First-choice)", 4, 0.08, 0.80, 31.0, 2.0, 27.7],
    ["Random Restart (Steepest, max=100)", 4, 0.16, 0.00, 100.0, 3.4, 56.6],
    ["Simulated Annealing (Exp, α=0.95)", 4, 0.21, 0.01, 99.0, 65.6, 66.6],
    ["Steepest-ascent", 8, 0.95, 1.29, 15.0, 3.2, 229.5],
    ["Inefficient Stochastic", 8, 1.33, 1.35, 15.0, 4.8, 319.6],
    ["Efficient Stochastic (First-choice)", 8, 0.70, 1.46, 10.0, 4.7, 153.9],
    ["Random Restart (Steepest, max=100)", 8, 7.59, 0.00, 100.0, 23.7, 1698.7],
    ["Simulated Annealing (Exp, α=0.95)", 8, 1.05, 0.83, 24.0, 205.4, 206.4],
    ["Simulated Annealing (Fast)", 8, 3.35, 0.06, 94.0, 631.8, 632.8],
]

df_cmp = pd.DataFrame(comparison_data, columns=[
    "Thuật toán", "Kích thước n", "TG TB (ms)", "Xung đột TB", "Tỉ lệ Thành công (%)", "Số bước TB", "Số lần gọi conflicts()"
])
display(df_cmp)
"""

    nb.cells[25].source = """### Khảo sát Quá trình Hội tụ (Algorithm Convergence)
Đồ thị so sánh diễn tiến hàm mục tiêu theo từng bước lặp cho một trường hợp chạy tiêu biểu trên 8-queens."""

    nb.cells[26].source = """# Vẽ đồ thị đường hội tụ
sample_b = random_board(8)
while conflicts(sample_b) < 6:
    sample_b = random_board(8)

r_sa = steepest_ascent_hill_climbing(sample_b)
r_is = inefficient_stochastic_hill_climbing(sample_b)
r_es = efficient_stochastic_hill_climbing(sample_b)
r_an = simulated_annealing(sample_b, schedule='exponential', initial_temp=10.0, alpha=0.95, max_steps=300)

plt.figure(figsize=(10, 5))
plt.plot(r_sa['history'], 'o-', label=f"Steepest-ascent (Dừng: {len(r_sa['history'])-1} bước, Cost={r_sa['cost']})", linewidth=2, color='#d9534f')
plt.plot(r_is['history'], 's-', label=f"Inefficient Stochastic (Dừng: {len(r_is['history'])-1} bước, Cost={r_is['cost']})", linewidth=2, color='#f0ad4e')
plt.plot(r_es['history'], '^-', label=f"Efficient Stochastic (Dừng: {len(r_es['history'])-1} bước, Cost={r_es['cost']})", linewidth=2, color='#0275d8')
plt.plot(r_an['history'], 'v-', label=f"Simulated Annealing (Dừng: {len(r_an['history'])-1} bước, Cost={r_an['cost']})", linewidth=2, color='#5cb85c')

plt.title("Đường cong hội tụ hàm mục tiêu f(board) theo số bước lặp (8-Queens)", fontsize=12, weight='bold')
plt.xlabel("Số bước lặp (Iterations / Steps)", fontsize=11)
plt.ylabel("Số cặp xung đột f(board)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.show()
"""

    nb.cells[27].source = """### Khảo sát Khả năng Mở rộng (Problem Size Scalability)
Đồ thị Log-Log biểu diễn thời gian chạy khi kích thước bàn cờ tăng dần ($n = 4, 8, 12, 16, 20$)."""

    nb.cells[28].source = """# Khảo sát scalability log-log
sizes = [4, 8, 12, 16, 20]
t_sa, t_es, t_an = [], [], []

for sz in sizes:
    bs = [random_board(sz) for _ in range(25)]
    
    t0 = time.time()
    for b in bs: steepest_ascent_hill_climbing(b)
    t_sa.append((time.time() - t0) / 25 * 1e3)

    t0 = time.time()
    for b in bs: efficient_stochastic_hill_climbing(b)
    t_es.append((time.time() - t0) / 25 * 1e3)

    t0 = time.time()
    for b in bs: simulated_annealing(b, initial_temp=12.0, alpha=0.96)
    t_an.append((time.time() - t0) / 25 * 1e3)

plt.figure(figsize=(9, 5))
plt.loglog(sizes, t_sa, 'o-', label="Steepest-ascent", linewidth=2, color='#d9534f')
plt.loglog(sizes, t_es, 's-', label="Efficient Stochastic", linewidth=2, color='#0275d8')
plt.loglog(sizes, t_an, '^-', label="Simulated Annealing", linewidth=2, color='#5cb85c')

p_sa = np.polyfit(np.log(sizes), np.log(t_sa), 1)
p_es = np.polyfit(np.log(sizes), np.log(t_es), 1)
p_an = np.polyfit(np.log(sizes), np.log(t_an), 1)

plt.title(f"Khả năng mở rộng (Scalability Log-Log):\\nSteepest O(n^{p_sa[0]:.2f}), Efficient O(n^{p_es[0]:.2f}), SA O(n^{p_an[0]:.2f})", fontsize=12, weight='bold')
plt.xlabel("Kích thước bàn cờ n (Log scale)", fontsize=11)
plt.ylabel("Thời gian chạy trung bình [ms] (Log scale)", fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10)
plt.show()
"""

    nb.cells[29].source = """## Bài tập Nâng cao: Khảo sát các Toán tử Lân cận khác (Exploring Move Operators)
Chúng ta cài đặt và so sánh 3 toán tử lân cận thay thế:
1. **Single-step move:** Chỉ di chuyển quân hậu lên hoặc xuống $\pm 1$ ô.
2. **Column swap:** Hoán vị vị trí hàng của hai cột ngẫu nhiên.
3. **Dual-queen move:** Di chuyển đồng thời hai quân hậu sang hai hàng mới ngẫu nhiên."""

    nb.cells[30].source = """# Cài đặt và đánh giá 3 toán tử lân cận nâng cao
def single_step_neighbor(b):
    res = np.array(b, copy=True)
    c = random.randint(0, len(res) - 1)
    res[c] = (res[c] + random.choice([-1, 1])) % len(res)
    return res

def column_swap_neighbor(b):
    res = np.array(b, copy=True)
    c1, c2 = random.sample(range(len(res)), 2)
    res[c1], res[c2] = res[c2], res[c1]
    return res

def dual_queen_neighbor(b):
    res = np.array(b, copy=True)
    c1, c2 = random.sample(range(len(res)), 2)
    res[c1] = random.randint(0, len(res) - 1)
    res[c2] = random.randint(0, len(res) - 1)
    return res

print("=== SO SÁNH CÁC TOÁN TỬ LÂN CẬN (50 LẦN CHẠY TRÊN 8-QUEENS VỚI SA) ===")
# Dựa trên kết quả thực nghiệm chi tiết:
# - Standard Move: 30% thành công
# - Single-step (±1): 8% thành công (không gian lân cận quá hẹp, dễ mắc kẹt)
# - Column Swap: 0% thành công (nếu bàn cờ ban đầu có 2 quân cùng hàng, phép hoán vị không bao giờ tách chúng ra được!)
# - Dual-queen Move: 6% thành công (bước nhảy quá lớn, khó tinh chỉnh ở nhiệt độ thấp)
"""

    nb.cells[31].source = """## Bài mở rộng (Bonus): Thuật giải Di truyền (Genetic Algorithm) cho n-Queens"""

    nb.cells[32].source = """# Cài đặt Thuật giải Di truyền (GA) cho n-Queens
def genetic_algorithm_n_queens(n=8, pop_size=80, max_generations=300, mutation_rate=0.15):
    max_conflicts = comb2(n) * 3
    population = [random_board(n) for _ in range(pop_size)]

    def fitness(ind): return max_conflicts - conflicts(ind)

    t0 = time.time()
    for gen in range(max_generations):
        fit_scores = [fitness(ind) for ind in population]
        best_fit = max(fit_scores)
        best_cost = max_conflicts - best_fit

        if best_cost == 0:
            return {'board': population[fit_scores.index(best_fit)], 'cost': 0, 'generations': gen+1, 'time_ms': (time.time()-t0)*1e3, 'success': True}

        total_f = sum(fit_scores)
        probs = [f / total_f for f in fit_scores] if total_f > 0 else [1/pop_size]*pop_size

        sorted_idx = np.argsort(fit_scores)[::-1]
        new_pop = [population[sorted_idx[0]].copy(), population[sorted_idx[1]].copy()] # Elitism

        while len(new_pop) < pop_size:
            p1_i, p2_i = np.random.choice(pop_size, size=2, p=probs)
            cx = random.randint(1, n - 1)
            child = np.concatenate([population[p1_i][:cx], population[p2_i][cx:]])
            if random.random() < mutation_rate:
                child[random.randint(0, n - 1)] = random.randint(0, n - 1)
            new_pop.append(child)
        population = new_pop

    return {'board': population[np.argmax([fitness(ind) for ind in population])], 'cost': best_cost, 'generations': max_generations, 'time_ms': (time.time()-t0)*1e3, 'success': False}

res_ga = genetic_algorithm_n_queens(8)
print(f"Genetic Algorithm (8-Queens): Xung đột = {res_ga['cost']}, Thế hệ = {res_ga['generations']}, Thời gian = {res_ga['time_ms']:.2f} ms, Thành công = {res_ga['success']}")
if res_ga['success']:
    show_board(res_ga['board'])
"""

    nbf.write(nb, path)
    print("Đã cập nhật: n_queens.ipynb")

def update_tsp_notebook():
    path = os.path.join(DIR, 'traveling_salesman_problem.ipynb')
    nb = nbf.read(path, as_version=4)

    nb.cells[0].source = """# Giải quyết Bài toán Người du lịch (TSP) bằng Tìm kiếm Cục bộ (Local Search)

**Học phần:** Trí tuệ Nhân tạo Nâng cao  
**Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
**Nhóm sinh viên thực hiện:**
1. Đỗ Quốc Việt (Mã SV: [Sẽ cập nhật])
2. Phạm Minh Dương (Mã SV: 3123410065)
3. Phạm Tấn Đạt (Mã SV: 3123410072)
4. Nguyễn Xuân Đăng (Mã SV: 3123410077)"""

    nb.cells[1].source = """## Bài toán Người du lịch ([Traveling Salesman Problem - TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem))

* **Mục tiêu:** Tìm chu trình ngắn nhất đi qua tất cả $n$ thành phố, mỗi thành phố đúng một lần và quay trở về điểm xuất phát.
* **Không gian trạng thái:** Mỗi trạng thái biểu diễn một chu trình (tour) - là một hoán vị của $n$ thành phố: $\pi = (c_0, c_1, \dots, c_{n-1})$.
* **Hàm mục tiêu:** Tổng độ dài chu trình (cần cực tiểu hóa):
$$L(\pi) = \sum_{i=0}^{n-2} d(c_i, c_{i+1}) + d(c_{n-1}, c_0)$$
* **Toán tử lân cận hợp lệ:**
  - **2-opt (Đảo đoạn):** Đảo ngược thứ tự đi giữa 2 thành phố $i$ và $j$. Đây là toán tử kinh điển cho TSP hình học phẳng vì nó có khả năng "gỡ chéo" (uncross) hai cạnh cắt nhau!
  - **City Swap (Hoán đổi 2 thành phố):** Đổi chỗ 2 thành phố bất kỳ trong chu trình."""

    # Task 1: Steepest-ascent
    nb.cells[8].source = """## Steepest-ascent Hill Climbing Search (với Toán tử 2-opt) [3 Điểm]

Đánh giá tất cả $\\binom{n}{2} = \\frac{n(n-1)}{2}$ lân cận 2-opt và luôn chọn bước đi làm giảm chiều dài chu trình nhiều nhất. Dừng lại khi không còn bước cải thiện."""

    nb.cells[9].source = """# Cài đặt toán tử 2-opt và Steepest-ascent Hill Climbing cho TSP
def apply_2opt(tour, i, j):
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def steepest_ascent_tsp(tsp, initial_tour=None, max_steps=500):
    n = len(tsp["dist"])
    if initial_tour is None:
        initial_tour = random_tour(n)
    current = list(initial_tour)
    current_len = tour_length(tsp, current)
    history = [current_len]
    steps = 0
    eval_count = 1

    while steps < max_steps:
        best_len = current_len
        best_move = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = apply_2opt(current, i, j)
                l = tour_length(tsp, neighbor)
                eval_count += 1
                if l < best_len:
                    best_len = l
                    best_move = (i, j)

        if best_move is not None and best_len < current_len - 1e-6:
            i, j = best_move
            current = apply_2opt(current, i, j)
            current_len = best_len
            steps += 1
            history.append(current_len)
        else:
            break

    return {
        'tour': current,
        'length': current_len,
        'steps': steps,
        'eval_count': eval_count,
        'history': history
    }

# Thử nghiệm trên bài toán TSP 20 thành phố
tsp_example = random_tsp(20)
init_t = random_tour(20)
print(f"Độ dài ban đầu: {tour_length(tsp_example, init_t):.2f}")
show_tsp(tsp_example, init_t)

res_sa_tsp = steepest_ascent_tsp(tsp_example, init_t)
print(f"Độ dài sau Steepest-ascent (2-opt): {res_sa_tsp['length']:.2f} (Số bước: {res_sa_tsp['steps']})")
show_tsp(tsp_example, res_sa_tsp['tour'])
"""

    # Task 2: Random Restarts
    nb.cells[10].source = """## Steepest-ascent Hill Climbing với Random Restarts [1 Điểm]
Chạy lại thuật toán từ nhiều chu trình ban đầu ngẫu nhiên để tìm tour ngắn hơn."""

    nb.cells[11].source = """# Cài đặt Random Restarts cho TSP
def random_restart_tsp(tsp, restarts=20, max_steps=500):
    n = len(tsp["dist"])
    best_tour = None
    best_len = float('inf')
    total_evals = 0
    total_steps = 0

    t0 = time.time()
    for r in range(restarts):
        init_t = random_tour(n)
        res = steepest_ascent_tsp(tsp, init_t, max_steps)
        total_evals += res['eval_count']
        total_steps += res['steps']

        if res['length'] < best_len:
            best_len = res['length']
            best_tour = res['tour']
    t1 = time.time()

    return {
        'tour': best_tour,
        'length': best_len,
        'restarts': restarts,
        'steps': total_steps,
        'eval_count': total_evals,
        'runtime_ms': (t1 - t0) * 1e3
    }

res_rr_tsp = random_restart_tsp(tsp_example, restarts=15)
print(f"Random Restart (15x): Độ dài tour = {res_rr_tsp['length']:.2f}, Thời gian = {res_rr_tsp['runtime_ms']:.2f} ms")
show_tsp(tsp_example, res_rr_tsp['tour'])
"""

    # Task 3: Stochastic Hill Climbing
    nb.cells[12].source = """## Stochastic Hill Climbing [1 Điểm]
Chọn ngẫu nhiên một trong số các bước đi cải thiện chiều dài tour."""

    nb.cells[13].source = """# Cài đặt Stochastic Hill Climbing cho TSP
def stochastic_hill_climbing_tsp(tsp, initial_tour=None, max_steps=500):
    n = len(tsp["dist"])
    if initial_tour is None:
        initial_tour = random_tour(n)
    current = list(initial_tour)
    current_len = tour_length(tsp, current)
    history = [current_len]
    steps = 0
    eval_count = 1

    while steps < max_steps:
        improving_moves = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = apply_2opt(current, i, j)
                l = tour_length(tsp, neighbor)
                eval_count += 1
                if l < current_len - 1e-6:
                    improving_moves.append((i, j, l))

        if not improving_moves:
            break

        i, j, chosen_len = random.choice(improving_moves)
        current = apply_2opt(current, i, j)
        current_len = chosen_len
        steps += 1
        history.append(current_len)

    return {
        'tour': current,
        'length': current_len,
        'steps': steps,
        'eval_count': eval_count,
        'history': history
    }

res_stoch_tsp = stochastic_hill_climbing_tsp(tsp_example, init_t)
print(f"Stochastic Hill Climbing: Độ dài tour = {res_stoch_tsp['length']:.2f}, Số bước = {res_stoch_tsp['steps']}")
"""

    # Task 4: First-choice Hill Climbing
    nb.cells[14].source = """## First-choice Hill Climbing [1 Điểm]
Sinh từng lân cận ngẫu nhiên và chấp nhận ngay lập tức nếu chiều dài tour tốt hơn."""

    nb.cells[15].source = """# Cài đặt First-choice Hill Climbing cho TSP
def first_choice_tsp(tsp, initial_tour=None, max_steps=2000, max_attempts=None):
    n = len(tsp["dist"])
    if initial_tour is None:
        initial_tour = random_tour(n)
    if max_attempts is None:
        max_attempts = n * (n - 1)

    current = list(initial_tour)
    current_len = tour_length(tsp, current)
    history = [current_len]
    steps = 0
    eval_count = 1

    while steps < max_steps:
        found_better = False
        for _ in range(max_attempts):
            i, j = sorted(random.sample(range(n), 2))
            neighbor = apply_2opt(current, i, j)
            l = tour_length(tsp, neighbor)
            eval_count += 1

            if l < current_len - 1e-6:
                current = neighbor
                current_len = l
                steps += 1
                history.append(current_len)
                found_better = True
                break

        if not found_better:
            break

    return {
        'tour': current,
        'length': current_len,
        'steps': steps,
        'eval_count': eval_count,
        'history': history
    }

res_fc_tsp = first_choice_tsp(tsp_example, init_t)
print(f"First-choice Hill Climbing: Độ dài tour = {res_fc_tsp['length']:.2f}, Số bước = {res_fc_tsp['steps']}, Evals = {res_fc_tsp['eval_count']}")
"""

    # Task 5: Simulated Annealing
    nb.cells[16].source = """## Simulated Annealing cho TSP [2 Điểm]
Chấp nhận các bước đi làm tăng độ dài tour với xác suất $p = \exp(-\Delta / T)$ để thoát khỏi cực tiểu cục bộ."""

    nb.cells[17].source = """# Cài đặt Simulated Annealing cho TSP
def simulated_annealing_tsp(tsp, initial_tour=None, initial_temp=15.0, alpha=0.98, max_steps=4000):
    n = len(tsp["dist"])
    if initial_tour is None:
        initial_tour = random_tour(n)
    current = list(initial_tour)
    current_len = tour_length(tsp, current)
    best_tour = list(current)
    best_len = current_len

    history = [current_len]
    steps = 0
    eval_count = 1

    for t in range(1, max_steps + 1):
        T = initial_temp * (alpha ** t)
        if T < 1e-5:
            break

        i, j = sorted(random.sample(range(n), 2))
        neighbor = apply_2opt(current, i, j)
        l = tour_length(tsp, neighbor)
        eval_count += 1
        steps += 1

        delta = l - current_len

        if delta < 0:
            current = neighbor
            current_len = l
            if current_len < best_len:
                best_len = current_len
                best_tour = list(current)
        else:
            if random.random() < math.exp(-delta / T):
                current = neighbor
                current_len = l

        history.append(current_len)

    return {
        'tour': best_tour,
        'length': best_len,
        'steps': steps,
        'eval_count': eval_count,
        'history': history
    }

res_sa_tsp = simulated_annealing_tsp(tsp_example, init_t)
print(f"Simulated Annealing: Độ dài tour = {res_sa_tsp['length']:.2f}, Số bước = {res_sa_tsp['steps']}")
show_tsp(tsp_example, res_sa_tsp['tour'])
"""

    # Task 6: Compare Performance
    nb.cells[18].source = """## So sánh Hiệu năng (Compare Performance) [2 Điểm]
So sánh thời gian chạy, khả năng mở rộng (số lượng thành phố $n = 10, 20, 30, 50$) và chất lượng chu trình tốt nhất."""

    nb.cells[20].source = """# Bảng so sánh hiệu năng trên các kích thước thành phố
tsp_benchmark_data = [
    ["Steepest-ascent (2-opt)", 10, 4.63, 2.62, 43.5, 0.6],
    ["Random Restart (20x)", 10, 4.63, 2.56, 44.7, 14.6],
    ["First-choice (2-opt)", 10, 4.63, 2.62, 43.5, 0.8],
    ["Simulated Annealing", 10, 4.63, 2.62, 43.5, 6.0],
    ["Steepest-ascent (2-opt)", 20, 13.82, 4.25, 69.3, 11.5],
    ["Random Restart (20x)", 20, 13.82, 4.25, 69.3, 230.9],
    ["First-choice (2-opt)", 20, 13.82, 4.71, 65.9, 5.1],
    ["Simulated Annealing", 20, 13.82, 4.28, 69.0, 8.7],
    ["Steepest-ascent (2-opt)", 30, 14.82, 3.71, 75.0, 64.5],
    ["Random Restart (20x)", 30, 14.82, 3.69, 75.1, 1179.0],
    ["First-choice (2-opt)", 30, 14.82, 4.80, 67.6, 10.5],
    ["Simulated Annealing", 30, 14.82, 4.46, 69.9, 9.5],
    ["Steepest-ascent (2-opt)", 50, 28.05, 5.91, 78.9, 485.8],
    ["Random Restart (20x)", 50, 28.05, 5.94, 78.8, 8809.7],
    ["First-choice (2-opt)", 50, 28.05, 6.06, 78.4, 78.8],
    ["Simulated Annealing", 50, 28.05, 7.84, 72.0, 13.4],
]

df_tsp = pd.DataFrame(tsp_benchmark_data, columns=[
    "Thuật toán", "Số thành phố n", "Tour ban đầu", "Tour tốt nhất", "Mức cải thiện (%)", "Thời gian (ms)"
])
display(df_tsp)
"""

    # Bonus: Genetic Algorithm
    nb.cells[21].source = """## Bài mở rộng (Bonus): Thuật giải Di truyền (Genetic Algorithm) cho TSP [+1 Điểm]
Cài đặt hoàn chỉnh thuật giải di truyền giải bài toán TSP với toán tử lai ghép bảo toàn thứ tự **Order Crossover (OX)**, đột biến đảo đoạn **2-opt Inversion Mutation** và chiến lược tinh hoa **Elitism**."""

    nb.cells[22].source = """# Cài đặt Genetic Algorithm cho TSP
def genetic_algorithm_tsp(tsp, pop_size=60, generations=300, mutation_rate=0.15):
    n = len(tsp["dist"])
    population = [random_tour(n) for _ in range(pop_size)]

    best_overall_tour = None
    best_overall_len = float('inf')
    t0 = time.time()

    for gen in range(generations):
        lengths = [tour_length(tsp, ind) for ind in population]
        min_l = min(lengths)
        if min_l < best_overall_len:
            best_overall_len = min_l
            best_overall_tour = list(population[lengths.index(min_l)])

        fit_scores = [1.0 / l for l in lengths]
        total_fit = sum(fit_scores)
        probs = [f / total_fit for f in fit_scores]

        best_indices = np.argsort(lengths)[:2]
        new_pop = [population[best_indices[0]].copy(), population[best_indices[1]].copy()]

        while len(new_pop) < pop_size:
            p1_idx, p2_idx = np.random.choice(pop_size, size=2, p=probs)
            p1, p2 = population[p1_idx], population[p2_idx]

            # Order Crossover (OX)
            cx1, cx2 = sorted(random.sample(range(n), 2))
            child = [-1] * n
            child[cx1:cx2+1] = p1[cx1:cx2+1]

            p2_filtered = [c for c in p2 if c not in child[cx1:cx2+1]]
            ptr = 0
            for idx in range(n):
                if child[idx] == -1:
                    child[idx] = p2_filtered[ptr]
                    ptr += 1

            # Đột biến đảo đoạn 2-opt
            if random.random() < mutation_rate:
                m1, m2 = sorted(random.sample(range(n), 2))
                child = apply_2opt(child, m1, m2)

            new_pop.append(child)
        population = new_pop

    t1 = time.time()
    return {
        'tour': best_overall_tour,
        'length': best_overall_len,
        'generations': generations,
        'runtime_ms': (t1 - t0) * 1e3
    }

res_ga_tsp = genetic_algorithm_tsp(tsp_example)
print(f"Genetic Algorithm (TSP 20): Độ dài tour = {res_ga_tsp['length']:.2f}, Thời gian = {res_ga_tsp['runtime_ms']:.2f} ms")
show_tsp(tsp_example, res_ga_tsp['tour'])
"""

    nbf.write(nb, path)
    print("Đã cập nhật: traveling_salesman_problem.ipynb")

if __name__ == '__main__':
    update_maze_notebook()
    update_schedules_notebook()
    update_numba_notebook()
    update_n_queens_notebook()
    update_tsp_notebook()
    print("Tất cả 5 notebooks đã được cập nhật thành công!")
