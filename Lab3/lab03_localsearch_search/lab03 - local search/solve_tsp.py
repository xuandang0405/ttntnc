"""
Thực nghiệm và giải pháp toàn diện cho Bài toán Người du lịch (TSP) - Lab 03
Bao gồm:
- Các toán tử lân cận: 2-opt (Edge Reversal) và City Swap
- Steepest-ascent Hill Climbing
- Steepest-ascent with Random Restarts
- Stochastic Hill Climbing
- First-choice Hill Climbing
- Simulated Annealing
- Thuật giải di truyền Genetic Algorithm cho TSP (Bonus)
- So sánh hiệu năng trên n=10, 20, 30, 50 thành phố
- Trực quan hóa Tour trước/sau và Đồ thị hội tụ
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Khởi tạo dữ liệu ngẫu nhiên lặp lại được
def random_tsp(n, seed=1234):
    np.random.seed(seed)
    pos = pd.DataFrame({
        "x": np.random.uniform(size=n),
        "y": np.random.uniform(size=n)
    })
    dist = squareform(pdist(pos))
    return {"pos": pos, "dist": dist}

def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def tour_length(tsp, tour):
    if not isinstance(tour, list):
        tour = tour.tolist()
    dist = tsp["dist"]
    tl = sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))
    tl += dist[tour[-1], tour[0]]
    return tl

# ==============================================================================
# CÁC TOÁN TỬ LÂN CẬN CHO TSP
# ==============================================================================
def apply_2opt(tour, i, j):
    """Đảo ngược đoạn đường từ vị trí i đến j (toán tử chuẩn 2-opt gỡ các cạnh chéo)"""
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def apply_swap(tour, i, j):
    """Hoán vị 2 thành phố tại chỉ số i và j"""
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# ==============================================================================
# THUẬT TOÁN 1: STEEPEST-ASCENT HILL CLIMBING (DÙNG 2-OPT)
# ==============================================================================
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

        # Khảo sát tất cả láng giềng n*(n-1)/2
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

# ==============================================================================
# THUẬT TOÁN 2: RANDOM RESTARTS CHO STEEPEST-ASCENT
# ==============================================================================
def random_restart_tsp(tsp, restarts=20, max_steps=500):
    n = len(tsp["dist"])
    best_tour = None
    best_len = float('inf')
    total_evals = 0
    total_steps = 0
    history = []

    t0 = time.perf_counter()
    for r in range(restarts):
        init_t = random_tour(n)
        res = steepest_ascent_tsp(tsp, init_t, max_steps)
        total_evals += res['eval_count']
        total_steps += res['steps']
        history.extend(res['history'])

        if res['length'] < best_len:
            best_len = res['length']
            best_tour = res['tour']
    t1 = time.perf_counter()

    return {
        'tour': best_tour,
        'length': best_len,
        'restarts': restarts,
        'steps': total_steps,
        'eval_count': total_evals,
        'runtime_ms': (t1 - t0) * 1e3,
        'history': history
    }

# ==============================================================================
# THUẬT TOÁN 3: STOCHASTIC HILL CLIMBING (ALL UPHILL MOVES)
# ==============================================================================
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

# ==============================================================================
# THUẬT TOÁN 4: FIRST-CHOICE HILL CLIMBING
# ==============================================================================
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

# ==============================================================================
# THUẬT TOÁN 5: SIMULATED ANNEALING CHO TSP
# ==============================================================================
def simulated_annealing_tsp(tsp, initial_tour=None, initial_temp=10.0, alpha=0.98, max_steps=5000):
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

# ==============================================================================
# BONUS: GENETIC ALGORITHM CHO TSP (ORDER CROSSOVER OX & 2-OPT MUTATION)
# ==============================================================================
def genetic_algorithm_tsp(tsp, pop_size=60, generations=300, mutation_rate=0.15):
    n = len(tsp["dist"])
    population = [random_tour(n) for _ in range(pop_size)]
    
    def fitness(ind):
        return 1.0 / (tour_length(tsp, ind) + 1e-6)

    best_overall_tour = None
    best_overall_len = float('inf')
    history = []
    t0 = time.perf_counter()

    for gen in range(generations):
        lengths = [tour_length(tsp, ind) for ind in population]
        min_l = min(lengths)
        history.append(min_l)

        if min_l < best_overall_len:
            best_overall_len = min_l
            best_overall_tour = list(population[lengths.index(min_l)])

        # Roulette selection
        fit_scores = [1.0 / l for l in lengths]
        total_fit = sum(fit_scores)
        probs = [f / total_fit for f in fit_scores]

        # Elitism: 2 tour tốt nhất
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

            # Mutation: 2-opt inversion
            if random.random() < mutation_rate:
                m1, m2 = sorted(random.sample(range(n), 2))
                child = apply_2opt(child, m1, m2)

            new_pop.append(child)

        population = new_pop

    t1 = time.perf_counter()
    return {
        'tour': best_overall_tour,
        'length': best_overall_len,
        'generations': generations,
        'runtime_ms': (t1 - t0) * 1e3,
        'history': history
    }

# ==============================================================================
# HÀM CHẠY THỰC NGHIỆM VÀ TRỰC QUAN HÓA CHO BÁO CÁO
# ==============================================================================
def run_all_tsp_experiments():
    random.seed(42)
    np.random.seed(42)
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("CHẠY THỰC NGHIỆM ĐÁNH GIÁ CÁC THUẬT TOÁN TSP")
    print("=" * 80)

    # 1. So sánh Tour ban đầu và Tour sau tối ưu trên TSP n=30
    tsp30 = random_tsp(30, seed=42)
    initial_t = random_tour(30)
    init_len = tour_length(tsp30, initial_t)

    res_sa = steepest_ascent_tsp(tsp30, initial_t)
    res_rr = random_restart_tsp(tsp30, restarts=20)
    res_ann = simulated_annealing_tsp(tsp30, initial_t, initial_temp=15.0, alpha=0.99, max_steps=5000)
    res_ga = genetic_algorithm_tsp(tsp30, pop_size=80, generations=300)

    print(f"Kích thước n=30:")
    print(f"Tour ban đầu (Random):   Chiều dài = {init_len:.3f}")
    print(f"Steepest-ascent:          Chiều dài = {res_sa['length']:.3f} (Cải thiện {(init_len - res_sa['length'])/init_len*100:.1f}%)")
    print(f"Random Restart (20x):     Chiều dài = {res_rr['length']:.3f} (Cải thiện {(init_len - res_rr['length'])/init_len*100:.1f}%)")
    print(f"Simulated Annealing:      Chiều dài = {res_ann['length']:.3f} (Cải thiện {(init_len - res_ann['length'])/init_len*100:.1f}%)")
    print(f"Genetic Algorithm:        Chiều dài = {res_ga['length']:.3f} (Cải thiện {(init_len - res_ga['length'])/init_len*100:.1f}%)")

    # Vẽ so sánh Tour ban đầu vs Tour tốt nhất
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def plot_tour_ax(ax, tsp, tour, title, color='blue'):
        pos = tsp["pos"]
        ax.scatter(pos["x"], pos["y"], color='red', s=60, zorder=3)
        for idx in range(len(pos)):
            ax.text(pos["x"][idx], pos["y"][idx] + 0.02, str(idx), fontsize=8, ha='center', color='black')
        
        pos_ = pos.reindex(tour)
        pos_ = pd.concat([pos_, pos_.head(1)])
        ax.plot(pos_["x"], pos_["y"], color=color, linewidth=2, alpha=0.8, zorder=2)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)

    plot_tour_ax(axes[0], tsp30, initial_t, f"Tour ban đầu (Ngẫu nhiên)\nĐộ dài: {init_len:.2f}", color='#d9534f')
    plot_tour_ax(axes[1], tsp30, res_sa['tour'], f"Steepest-ascent (2-opt)\nĐộ dài: {res_sa['length']:.2f}", color='#f0ad4e')
    plot_tour_ax(axes[2], tsp30, res_ann['tour'], f"Simulated Annealing (2-opt)\nĐộ dài: {res_ann['length']:.2f}", color='#5cb85c')

    plt.suptitle("Bài toán Người du lịch (TSP 30 thành phố): Trực quan hóa Tour trước và sau tối ưu", fontsize=13, weight='bold')
    plt.tight_layout()
    tour_path = os.path.join(output_dir, 'tsp_tour_before_after.png')
    plt.savefig(tour_path, dpi=200)
    plt.close()
    print(f"Đã lưu hình tour TSP tại: {tour_path}")

    # 2. Vẽ đường hội tụ Tour Length của các thuật toán
    plt.figure(figsize=(10, 6))
    plt.plot(res_sa['history'], 'o-', label=f"Steepest-ascent ({len(res_sa['history'])} bước, L={res_sa['length']:.2f})", color='#d9534f', linewidth=2)
    # Lấy 1 phần lịch sử của SA để biểu diễn tương quan
    sa_sampled = res_ann['history'][::20]
    plt.plot(range(0, len(res_ann['history']), 20), sa_sampled, label=f"Simulated Annealing (L={res_ann['length']:.2f})", color='#5cb85c', linewidth=2)
    plt.plot(res_ga['history'], label=f"Genetic Algorithm (L={res_ga['length']:.2f})", color='#0275d8', linewidth=2)

    plt.title("Đường cong hội tụ chiều dài chu trình TSP (30 thành phố)", fontsize=13, weight='bold')
    plt.xlabel("Số bước lặp / Thế hệ", fontsize=11)
    plt.ylabel("Tổng độ dài tour (Tour Length)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    conv_path = os.path.join(output_dir, 'tsp_convergence.png')
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f"Đã lưu đồ thị hội tụ TSP tại: {conv_path}")

    # 3. Benchmark trên các kích thước thành phố: n = 10, 20, 30, 50
    sizes = [10, 20, 30, 50]
    algo_dict = {
        "Steepest-ascent (2-opt)": lambda tsp: steepest_ascent_tsp(tsp),
        "Random Restart (20x)": lambda tsp: random_restart_tsp(tsp, restarts=20),
        "First-choice (2-opt)": lambda tsp: first_choice_tsp(tsp),
        "Simulated Annealing": lambda tsp: simulated_annealing_tsp(tsp, initial_temp=15.0, alpha=0.99, max_steps=4000),
    }

    tsp_benchmarks = []
    print("\n--- BẢNG KẾT QUẢ THÍ NGHIỆM TSP TRÊN NHIỀU KÍCH THƯỚC THÀNH PHỐ ---")
    print(f"{'Thuật toán':<26} | {'Kích thước n':<12} | {'Tour ban đầu':<14} | {'Tour tốt nhất':<14} | {'Cải thiện (%)':<14} | {'Thời gian (ms)':<14}")
    print("-" * 105)

    for sz in sizes:
        tsp = random_tsp(sz, seed=42 + sz)
        init_tour = random_tour(sz)
        init_l = tour_length(tsp, init_tour)

        for name, func in algo_dict.items():
            t0 = time.perf_counter()
            res = func(tsp)
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1e3
            best_l = res['length']
            improvement = (init_l - best_l) / init_l * 100

            tsp_benchmarks.append({
                'algorithm': name,
                'n': sz,
                'init_len': init_l,
                'best_len': best_l,
                'improvement': improvement,
                'time_ms': time_ms
            })

            print(f"{name:<26} | n = {sz:<8} | {init_l:<14.2f} | {best_l:<14.2f} | {improvement:<13.1f}% | {time_ms:<14.1f}")

    return tsp_benchmarks

if __name__ == '__main__':
    run_all_tsp_experiments()
