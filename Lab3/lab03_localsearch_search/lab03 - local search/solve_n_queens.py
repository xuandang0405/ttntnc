"""
Thực nghiệm và giải pháp toàn diện cho Bài toán n-Queens (Lab 03 - Local Search)
Bao gồm:
- Task 1: Steepest-ascent Hill Climbing
- Task 2: Inefficient Stochastic Hill Climbing
- Task 3: Efficient Stochastic (First-choice) Hill Climbing
- Task 4: Hill Climbing with Random Restarts
- Task 5: Simulated Annealing (hỗ trợ nhiều lịch nhiệt)
- Task 6: Đánh giá thực nghiệm n=4, n=8 (100 lần lặp), Hội tụ, Khả năng mở rộng n=4..20
- Bài tập nâng cao: Khảo sát các toán tử lân cận (Single-step, Column swap, Dual-queen)
- Bonus: Thuật giải di truyền (Genetic Algorithm) cho n-Queens
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import sys
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Sử dụng comb2 và conflicts tối ưu O(n)
def comb2(n):
    return n * (n - 1) // 2

def conflicts(board):
    """Tính số xung đột O(n)"""
    n = len(board)
    horizontal_cnt = [0] * n
    diagonal1_cnt = [0] * (2 * n)
    diagonal2_cnt = [0] * (2 * n)

    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1

    total = 0
    for c in horizontal_cnt:
        if c > 1: total += comb2(c)
    for c in diagonal1_cnt:
        if c > 1: total += comb2(c)
    for c in diagonal2_cnt:
        if c > 1: total += comb2(c)
    return total

def random_board(n):
    return np.random.randint(0, n, size=n)

# ==============================================================================
# TASK 1: STEEPEST-ASCENT HILL CLIMBING
# ==============================================================================
def steepest_ascent_hill_climbing(board, max_steps=1000):
    current = np.array(board, copy=True)
    n = len(current)
    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps:
        if current_cost == 0:
            break

        best_move = None
        best_cost = current_cost
        improving_moves = []

        for col in range(n):
            original_row = current[col]
            for row in range(n):
                if row == original_row:
                    continue
                current[col] = row
                cost = conflicts(current)
                eval_count += 1
                
                if cost < best_cost:
                    best_cost = cost
                    improving_moves = [(col, row)]
                elif cost == best_cost and cost < current_cost:
                    improving_moves.append((col, row))
                
                current[col] = original_row

        if not improving_moves or best_cost >= current_cost:
            # Mắc kẹt tại cực trị cục bộ (hoặc vai/plateau)
            break

        # Chọn 1 trong các bước tốt nhất (ngẫu nhiên nếu có hòa điểm)
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

# ==============================================================================
# TASK 2: INEFFICIENT STOCHASTIC HILL CLIMBING
# ==============================================================================
def inefficient_stochastic_hill_climbing(board, max_steps=1000):
    current = np.array(board, copy=True)
    n = len(current)
    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps:
        if current_cost == 0:
            break

        uphill_moves = [] # Tất cả các bước cải thiện (cost < current_cost)
        
        for col in range(n):
            original_row = current[col]
            for row in range(n):
                if row == original_row:
                    continue
                current[col] = row
                cost = conflicts(current)
                eval_count += 1

                if cost < current_cost:
                    uphill_moves.append((col, row, cost))
                current[col] = original_row

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

# ==============================================================================
# TASK 3: EFFICIENT STOCHASTIC (FIRST-CHOICE) HILL CLIMBING
# ==============================================================================
def efficient_stochastic_hill_climbing(board, max_steps=2000, max_attempts=None):
    current = np.array(board, copy=True)
    n = len(current)
    if max_attempts is None:
        max_attempts = n * (n - 1) * 2  # Ngưỡng kiên nhẫn khi không tìm thấy lân cận tốt hơn
    
    current_cost = conflicts(current)
    eval_count = 1
    steps = 0
    history = [current_cost]

    while steps < max_steps and current_cost > 0:
        found_better = False
        for _ in range(max_attempts):
            col = random.randint(0, n - 1)
            original_row = current[col]
            # Chọn hàng khác hàng hiện tại
            row = random.randint(0, n - 2)
            if row >= original_row:
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
                current[col] = original_row # Rollback

        if not found_better:
            # Đã thử max_attempts lân cận ngẫu nhiên mà không cải thiện -> Dừng
            break

    return {
        'board': current,
        'cost': current_cost,
        'steps': steps,
        'eval_count': eval_count,
        'success': (current_cost == 0),
        'history': history
    }

# ==============================================================================
# TASK 4: HILL CLIMBING WITH RANDOM RESTARTS
# ==============================================================================
def random_restart_hill_climbing(algorithm_func, n, max_restarts=100, **kwargs):
    best_board = None
    best_cost = float('inf')
    total_evals = 0
    total_steps = 0
    restarts_used = 0
    history = []

    t0 = time.perf_counter()
    for r in range(max_restarts):
        restarts_used += 1
        initial_board = random_board(n)
        res = algorithm_func(initial_board, **kwargs)
        
        total_evals += res['eval_count']
        total_steps += res['steps']
        history.extend(res['history'])

        if res['cost'] < best_cost:
            best_cost = res['cost']
            best_board = res['board']

        if best_cost == 0:
            break
    t1 = time.perf_counter()

    return {
        'board': best_board,
        'cost': best_cost,
        'restarts': restarts_used,
        'total_steps': total_steps,
        'eval_count': total_evals,
        'runtime_ms': (t1 - t0) * 1e3,
        'success': (best_cost == 0),
        'history': history
    }

# ==============================================================================
# TASK 5: SIMULATED ANNEALING
# ==============================================================================
def simulated_annealing(board, schedule='exponential', initial_temp=10.0, alpha=0.95, eta=0.01, max_steps=2000):
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

        # Tính nhiệt độ T theo lịch
        if schedule == 'exponential':
            T = initial_temp * (alpha ** t)
        elif schedule == 'linear':
            T = max(1e-4, initial_temp - eta * t)
        elif schedule == 'fast':
            T = initial_temp / (1.0 + t)
        elif schedule == 'logarithmic':
            T = initial_temp / np.log(1.0 + t)
        else:
            T = initial_temp * (alpha ** t)

        if T <= 1e-4:
            break

        # Sinh một láng giềng ngẫu nhiên
        col = random.randint(0, n - 1)
        original_row = current[col]
        row = random.randint(0, n - 2)
        if row >= original_row:
            row += 1

        current[col] = row
        next_cost = conflicts(current)
        eval_count += 1
        steps += 1

        delta = next_cost - current_cost

        # Quy tắc Metropolis
        if delta < 0:
            current_cost = next_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_board = np.array(current, copy=True)
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_cost = next_cost
            else:
                current[col] = original_row # Reject move

        history.append(current_cost)

    return {
        'board': best_board if best_cost < current_cost else current,
        'cost': min(best_cost, current_cost),
        'steps': steps,
        'eval_count': eval_count,
        'success': (min(best_cost, current_cost) == 0),
        'history': history
    }

# ==============================================================================
# ADVANCED MOVE OPERATORS
# ==============================================================================
def single_step_neighbor(board):
    """Di chuyển 1 quân hậu chỉ 1 ô lên hoặc xuống"""
    n = len(board)
    b = np.array(board, copy=True)
    col = random.randint(0, n - 1)
    direction = random.choice([-1, 1])
    b[col] = (b[col] + direction) % n
    return b

def column_swap_neighbor(board):
    """Hoán vị hàng của 2 cột ngẫu nhiên"""
    b = np.array(board, copy=True)
    c1, c2 = random.sample(range(len(b)), 2)
    b[c1], b[c2] = b[c2], b[c1]
    return b

def dual_queen_neighbor(board):
    """Di chuyển đồng thời 2 quân hậu sang hàng mới"""
    n = len(board)
    b = np.array(board, copy=True)
    c1, c2 = random.sample(range(n), 2)
    b[c1] = random.randint(0, n - 1)
    b[c2] = random.randint(0, n - 1)
    return b

def sa_with_operator(board, operator_func, max_steps=2000, initial_temp=10.0, alpha=0.95):
    current = np.array(board, copy=True)
    current_cost = conflicts(current)
    best_cost = current_cost
    steps = 0
    history = [current_cost]

    for t in range(1, max_steps + 1):
        if current_cost == 0:
            break
        T = initial_temp * (alpha ** t)
        if T < 1e-4:
            break

        candidate = operator_func(current)
        cost_candidate = conflicts(candidate)
        steps += 1
        delta = cost_candidate - current_cost

        if delta < 0:
            current = candidate
            current_cost = cost_candidate
            if current_cost < best_cost:
                best_cost = current_cost
        else:
            if random.random() < math.exp(-delta / T):
                current = candidate
                current_cost = cost_candidate

        history.append(current_cost)

    return {'cost': best_cost, 'steps': steps, 'success': (best_cost == 0), 'history': history}

# ==============================================================================
# BONUS: GENETIC ALGORITHM FOR N-QUEENS
# ==============================================================================
def genetic_algorithm_n_queens(n, pop_size=100, max_generations=500, mutation_rate=0.1):
    max_conflicts = comb2(n) * 3
    
    # Khởi tạo quần thể
    population = [random_board(n) for _ in range(pop_size)]
    
    def fitness(individual):
        return max_conflicts - conflicts(individual)

    history = []
    t0 = time.perf_counter()

    for gen in range(max_generations):
        fitness_scores = [fitness(ind) for ind in population]
        best_fit = max(fitness_scores)
        best_cost = max_conflicts - best_fit
        history.append(best_cost)

        if best_cost == 0:
            t1 = time.perf_counter()
            best_ind = population[fitness_scores.index(best_fit)]
            return {
                'board': best_ind,
                'cost': 0,
                'generations': gen + 1,
                'success': True,
                'runtime_ms': (t1 - t0) * 1e3,
                'history': history
            }

        # Chọn lọc Roulette wheel / Rank selection
        total_fit = sum(fitness_scores)
        if total_fit == 0:
            probs = [1.0 / pop_size] * pop_size
        else:
            probs = [f / total_fit for f in fitness_scores]

        # Sinh sản quần thể mới (Elitism: giữ lại 2 cá thể tốt nhất)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_pop = [population[sorted_indices[0]].copy(), population[sorted_indices[1]].copy()]

        while len(new_pop) < pop_size:
            # Chọn 2 cha mẹ
            p1_idx, p2_idx = np.random.choice(pop_size, size=2, p=probs)
            p1, p2 = population[p1_idx], population[p2_idx]

            # Crossover (1-point crossover)
            cx_point = random.randint(1, n - 1)
            child = np.concatenate([p1[:cx_point], p2[cx_point:]])

            # Mutation
            if random.random() < mutation_rate:
                mut_col = random.randint(0, n - 1)
                child[mut_col] = random.randint(0, n - 1)

            new_pop.append(child)

        population = new_pop

    t1 = time.perf_counter()
    fitness_scores = [fitness(ind) for ind in population]
    best_fit = max(fitness_scores)
    best_ind = population[fitness_scores.index(best_fit)]
    return {
        'board': best_ind,
        'cost': max_conflicts - best_fit,
        'generations': max_generations,
        'success': (max_conflicts - best_fit == 0),
        'runtime_ms': (t1 - t0) * 1e3,
        'history': history
    }

# ==============================================================================
# HÀM CHẠY TẤT CẢ THỬ NGHIỆM VÀ XUẤT SỐ LIỆU + BIỂU ĐỒ
# ==============================================================================
def run_all_n_queens_experiments():
    random.seed(1234)
    np.random.seed(1234)
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("CHẠY THỰC NGHIỆM ĐÁNH GIÁ CÁC THUẬT TOÁN n-QUEENS TRÊN n=4 VÀ n=8 (100 LẦN)")
    print("=" * 80)

    algorithms = [
        ("Steepest-ascent", lambda b: steepest_ascent_hill_climbing(b)),
        ("Inefficient Stochastic", lambda b: inefficient_stochastic_hill_climbing(b)),
        ("Efficient Stochastic (First-choice)", lambda b: efficient_stochastic_hill_climbing(b)),
        ("Random Restart (Steepest, max=100)", lambda b: random_restart_hill_climbing(steepest_ascent_hill_climbing, len(b))),
        ("Random Restart (Efficient, max=100)", lambda b: random_restart_hill_climbing(efficient_stochastic_hill_climbing, len(b))),
        ("Simulated Annealing (Exp, α=0.95)", lambda b: simulated_annealing(b, schedule='exponential', initial_temp=10.0, alpha=0.95)),
        ("Simulated Annealing (Fast)", lambda b: simulated_annealing(b, schedule='fast', initial_temp=20.0)),
    ]

    benchmark_results = {}

    for n in [4, 8]:
        print(f"\n--- BÀN CỜ KÍCH THƯỚC n = {n} ---")
        reps = 100
        boards = [random_board(n) for _ in range(reps)]

        for name, algo in algorithms:
            runtimes = []
            final_costs = []
            steps_list = []
            evals_list = []
            success_count = 0

            for b in boards:
                t0 = time.perf_counter()
                res = algo(b)
                t1 = time.perf_counter()

                runtimes.append((t1 - t0) * 1e3) # ms
                final_costs.append(res['cost'])
                steps_list.append(res.get('steps', res.get('total_steps', 0)))
                evals_list.append(res['eval_count'])
                if res['success']:
                    success_count += 1

            avg_time = np.mean(runtimes)
            avg_cost = np.mean(final_costs)
            succ_rate = (success_count / reps) * 100
            avg_steps = np.mean(steps_list)
            avg_evals = np.mean(evals_list)

            benchmark_results[(name, n)] = {
                'avg_time_ms': avg_time,
                'avg_cost': avg_cost,
                'succ_rate': succ_rate,
                'avg_steps': avg_steps,
                'avg_evals': avg_evals
            }

            print(f"{name:<36} | Size {n:<2} | Time: {avg_time:6.2f} ms | Cost: {avg_cost:4.2f} | Success: {succ_rate:5.1f}% | Steps: {avg_steps:5.1f} | Evals: {avg_evals:7.1f}")

    # ==============================================================================
    # VẼ ĐỒ THỊ 1: ĐƯỜNG HỘI TỤ HÀM MỤC TIÊU TRÊN 8-QUEENS
    # ==============================================================================
    print("\nVẽ đồ thị hội tụ cho 8-queens...")
    plt.figure(figsize=(10, 6))
    sample_board = random_board(8)
    while conflicts(sample_board) < 6: # Chọn bàn cờ có số xung đột tương đối cao để quan sát
        sample_board = random_board(8)

    res_sa = steepest_ascent_hill_climbing(sample_board)
    res_is = inefficient_stochastic_hill_climbing(sample_board)
    res_es = efficient_stochastic_hill_climbing(sample_board)
    res_an = simulated_annealing(sample_board, schedule='exponential', initial_temp=10.0, alpha=0.95, max_steps=300)

    plt.plot(res_sa['history'], 'o-', label=f"Steepest-ascent (Dừng: {len(res_sa['history'])-1} bước, Cost={res_sa['cost']})", linewidth=2, color='#d9534f')
    plt.plot(res_is['history'], 's-', label=f"Inefficient Stochastic (Dừng: {len(res_is['history'])-1} bước, Cost={res_is['cost']})", linewidth=2, color='#f0ad4e')
    plt.plot(res_es['history'], '^-', label=f"Efficient Stochastic (Dừng: {len(res_es['history'])-1} bước, Cost={res_es['cost']})", linewidth=2, color='#0275d8')
    plt.plot(res_an['history'], 'v-', label=f"Simulated Annealing (Dừng: {len(res_an['history'])-1} bước, Cost={res_an['cost']})", linewidth=2, color='#5cb85c')

    plt.title("Đường cong hội tụ hàm mục tiêu $f(board)$ theo số bước lặp (8-Queens)", fontsize=13, weight='bold')
    plt.xlabel("Số bước lặp (Iterations / Steps)", fontsize=11)
    plt.ylabel("Số cặp xung đột $f(board)$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    conv_path = os.path.join(output_dir, 'n_queens_convergence.png')
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f"Đã lưu đồ thị hội tụ tại: {conv_path}")

    # ==============================================================================
    # VẼ ĐỒ THỊ 2: KHẢ NĂNG MỞ RỘNG (SCALABILITY LOG-LOG) n = 4, 8, 12, 16, 20
    # ==============================================================================
    print("\nKhảo sát khả năng mở rộng (Scalability n=4..20)...")
    sizes = [4, 8, 12, 16, 20]
    time_sa = []
    time_es = []
    time_anneal = []

    for sz in sizes:
        boards_sz = [random_board(sz) for _ in range(30)]
        
        t0 = time.perf_counter()
        for b in boards_sz: steepest_ascent_hill_climbing(b)
        time_sa.append((time.perf_counter() - t0) / 30 * 1e3)

        t0 = time.perf_counter()
        for b in boards_sz: efficient_stochastic_hill_climbing(b)
        time_es.append((time.perf_counter() - t0) / 30 * 1e3)

        t0 = time.perf_counter()
        for b in boards_sz: simulated_annealing(b, initial_temp=15.0, alpha=0.96)
        time_anneal.append((time.perf_counter() - t0) / 30 * 1e3)

    plt.figure(figsize=(9, 6))
    plt.loglog(sizes, time_sa, 'o-', label="Steepest-ascent", linewidth=2, color='#d9534f')
    plt.loglog(sizes, time_es, 's-', label="Efficient Stochastic", linewidth=2, color='#0275d8')
    plt.loglog(sizes, time_anneal, '^-', label="Simulated Annealing", linewidth=2, color='#5cb85c')

    # Ước lượng độ dốc Big O bằng polyfit trên log
    p_sa = np.polyfit(np.log(sizes), np.log(time_sa), 1)
    p_es = np.polyfit(np.log(sizes), np.log(time_es), 1)
    p_an = np.polyfit(np.log(sizes), np.log(time_anneal), 1)

    plt.title(f"Khả năng mở rộng (Scalability Log-Log):\nSteepest-ascent $\\approx O(n^{{{p_sa[0]:.2f}}})$, Efficient $\\approx O(n^{{{p_es[0]:.2f}}})$, SA $\\approx O(n^{{{p_an[0]:.2f}}})$", fontsize=13, weight='bold')
    plt.xlabel("Kích thước bàn cờ $n$ (Log scale)", fontsize=11)
    plt.ylabel("Thời gian thực thi trung bình [ms] (Log scale)", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    scale_path = os.path.join(output_dir, 'n_queens_scalability_loglog.png')
    plt.savefig(scale_path, dpi=200)
    plt.close()
    print(f"Đã lưu đồ thị scalability tại: {scale_path}")

    # ==============================================================================
    # VẼ ĐỒ THỊ 3: SO SÁNH CÁC TOÁN TỬ LÂN CẬN NÂNG CAO (ADVANCED MOVE OPERATORS)
    # ==============================================================================
    print("\nKhảo sát các toán tử lân cận nâng cao...")
    operators = [
        ("Standard Move (Hàng ngẫu nhiên cùng cột)", lambda b: simulated_annealing(b)['cost']),
        ("Single-step (±1 ô hàng)", lambda b: sa_with_operator(b, single_step_neighbor)['cost']),
        ("Column Swap (Hoán vị 2 cột)", lambda b: sa_with_operator(b, column_swap_neighbor)['cost']),
        ("Dual-queen Move (Chuyển 2 quân)", lambda b: sa_with_operator(b, dual_queen_neighbor)['cost']),
    ]

    op_success_rates = []
    op_names = []
    test_boards_8 = [random_board(8) for _ in range(50)]

    for name, op_eval in operators:
        succ = sum(1 for b in test_boards_8 if op_eval(b) == 0)
        rate = (succ / 50) * 100
        op_names.append(name)
        op_success_rates.append(rate)
        print(f"Toán tử: {name:<40} | Tỉ lệ thành công (8-queens): {rate:.1f}%")

    plt.figure(figsize=(10, 5))
    bars = plt.barh(op_names, op_success_rates, color=['#5cb85c', '#f0ad4e', '#0275d8', '#6f42c1'], height=0.55)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', weight='bold')

    plt.title("So sánh Hiệu quả của các Toán tử Lân cận (Local Move Operators) trên 8-Queens", fontsize=12, weight='bold')
    plt.xlabel("Tỉ lệ tìm thấy lời giải tối ưu f=0 (%)", fontsize=11)
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    op_path = os.path.join(output_dir, 'n_queens_move_operators_comparison.png')
    plt.savefig(op_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ toán tử lân cận tại: {op_path}")

    # ==============================================================================
    # THỰC NGHIỆM GENETIC ALGORITHM (BONUS)
    # ==============================================================================
    print("\nThử nghiệm Genetic Algorithm (Bonus) trên 8-queens...")
    ga_results = [genetic_algorithm_n_queens(8, pop_size=80, max_generations=300) for _ in range(20)]
    ga_succ = sum(1 for r in ga_results if r['success'])
    ga_avg_time = np.mean([r['runtime_ms'] for r in ga_results])
    ga_avg_gens = np.mean([r['generations'] for r in ga_results])
    print(f"GA (8-queens, 20 lần chạy): Thành công {ga_succ}/20 ({(ga_succ/20)*100:.1f}%) | TG trung bình: {ga_avg_time:.2f} ms | Thế hệ TB: {ga_avg_gens:.1f}")

    return benchmark_results

if __name__ == '__main__':
    run_all_n_queens_experiments()
