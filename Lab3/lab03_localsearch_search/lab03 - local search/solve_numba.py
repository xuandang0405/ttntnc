"""
Thực nghiệm và trực quan hóa Bài 6: Tối ưu hóa hàm đếm xung đột n-Queens với Numba JIT
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 1. Cài đặt Naive O(n^2)
def conflicts_naive(board):
    n = len(board)
    val = 0
    # Đếm xung đột hàng
    for i in range(n):
        val += math.comb(int(np.sum(board == i)), 2)
    
    # Đếm xung đột đường chéo
    for j in range(n):
        q_up = board[j]
        q_down = board[j]
        for jj in range(j + 1, n):
            q_up -= 1
            q_down += 1
            if board[jj] == q_up:
                val += 1
            if board[jj] == q_down:
                val += 1
    return val

# 2. Cài đặt Linear O(n) bằng Python thuần
def comb2(n):
    return n * (n - 1) // 2

def conflicts_linear(board):
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
        if c > 1:
            total += comb2(c)
    for c in diagonal1_cnt:
        if c > 1:
            total += comb2(c)
    for c in diagonal2_cnt:
        if c > 1:
            total += comb2(c)
    return total

# 3. Cài đặt Compiled O(n) với Numba njit
@njit
def comb2_jit(n):
    return n * (n - 1) // 2

@njit
def conflicts_jit(board):
    n = len(board)
    cnt = [0] * (5 * n)

    for i in range(n):
        cnt[board[i]] += 1
        cnt[n + (i + board[i])] += 1
        cnt[3 * n + (i - board[i] + n)] += 1

    total = 0
    for x in cnt:
        if x > 1:
            total += comb2_jit(x)
    return total

def run_numba_benchmarks():
    np.random.seed(1234)
    print("Warm-up Numba JIT (loại bỏ thời gian biên dịch lần đầu)...")
    t_compile_start = time.perf_counter()
    conflicts_jit(np.random.randint(0, 8, size=8))
    t_compile_end = time.perf_counter()
    print(f"Thời gian biên dịch JIT lần đầu: {(t_compile_end - t_compile_start)*1e3:.2f} ms")

    # Các kích thước bàn cờ thử nghiệm
    sizes = [8, 16, 32, 64, 100, 200, 300, 400, 500]
    reps = 30

    naive_times = []
    linear_times = []
    jit_times = []

    print("\n--- BẮT ĐẦU ĐO THỜI GIAN THỰC THI (Microseconds - µs) ---")
    print(f"{'Size (n)':<10} | {'Naive O(n^2) (µs)':<20} | {'Linear O(n) (µs)':<20} | {'Numba JIT (µs)':<18} | {'Tăng tốc JIT/Linear':<20}")
    print("-" * 95)

    for n in sizes:
        boards = [np.random.randint(0, n, size=n) for _ in range(reps)]

        # Đo naive (chỉ đo đến n=300 để tránh chờ đợi quá lâu nếu O(n^2) quá lớn)
        if n <= 300:
            t0 = time.perf_counter()
            for b in boards:
                conflicts_naive(b)
            t1 = time.perf_counter()
            t_naive = (t1 - t0) / reps * 1e6
        else:
            # Ước lượng ngoại suy lý thuyết O(n^2)
            t_naive = naive_times[-1] * ((n / sizes[sizes.index(n)-1])**2)
        naive_times.append(t_naive)

        # Đo linear O(n)
        t0 = time.perf_counter()
        for b in boards:
            conflicts_linear(b)
        t1 = time.perf_counter()
        t_linear = (t1 - t0) / reps * 1e6
        linear_times.append(t_linear)

        # Đo JIT O(n)
        t0 = time.perf_counter()
        for b in boards:
            conflicts_jit(b)
        t1 = time.perf_counter()
        t_jit = (t1 - t0) / reps * 1e6
        jit_times.append(t_jit)

        speedup = t_linear / t_jit if t_jit > 0 else 0
        print(f"{n:<10} | {t_naive:<20.2f} | {t_linear:<20.2f} | {t_jit:<18.2f} | {speedup:<18.1f}x")

    # Vẽ biểu đồ so sánh:
    # 1. So sánh cả 3 phương pháp (Linear Scale)
    # 2. So sánh Linear O(n) và Numba JIT (Zoom-in)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(sizes, np.array(naive_times) / 1000.0, 'o-', color='#d9534f', label='Naive $O(n^2)$ (Python)', linewidth=2)
    ax1.plot(sizes, np.array(linear_times) / 1000.0, 's-', color='#f0ad4e', label='Linear $O(n)$ (Pure Python)', linewidth=2)
    ax1.plot(sizes, np.array(jit_times) / 1000.0, '^-', color='#5cb85c', label='Linear $O(n)$ (Numba JIT)', linewidth=2)
    ax1.set_title("So sánh thời gian chạy: Naive vs Linear vs JIT (Tổng quan)", fontsize=11, weight='bold')
    ax1.set_xlabel("Kích thước bàn cờ ($n$)", fontsize=11)
    ax1.set_ylabel("Thời gian trung bình (ms)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # Biểu đồ Zoom-in Linear vs JIT
    ax2.plot(sizes, linear_times, 's-', color='#f0ad4e', label='Linear $O(n)$ (Pure Python)', linewidth=2)
    ax2.plot(sizes, jit_times, '^-', color='#5cb85c', label='Linear $O(n)$ (Numba JIT)', linewidth=2)
    ax2.set_title("So sánh Linear Python thuần vs Numba JIT (Zoom-in)", fontsize=11, weight='bold')
    ax2.set_xlabel("Kích thước bàn cờ ($n$)", fontsize=11)
    ax2.set_ylabel("Thời gian trung bình (µs - micro giây)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("Đánh giá hiệu năng và Độ phức tạp thuật toán: Hàm tính xung đột n-Queens", fontsize=14, weight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'numba_benchmark_comparison.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nĐã lưu biểu đồ Numba benchmark tại: {fig_path}")

    return {
        'sizes': sizes,
        'naive_times': naive_times,
        'linear_times': linear_times,
        'jit_times': jit_times
    }

if __name__ == '__main__':
    run_numba_benchmarks()
