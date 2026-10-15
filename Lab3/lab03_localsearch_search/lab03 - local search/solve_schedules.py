"""
Thực nghiệm và trực quan hóa Bài 5: Các lịch giảm nhiệt độ trong Simulated Annealing
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def run_schedule_experiments():
    time = np.arange(1, 101)
    T0 = 100.0

    # Các hàm lịch giảm nhiệt độ
    def temp_exponential(t, initial_T, alpha):
        return initial_T * (alpha ** t)

    def temp_linear(t, initial_T, eta):
        return np.maximum(0, initial_T - eta * t)

    def temp_fast(t, initial_T):
        return initial_T / (1.0 + t)

    def temp_logarithmic(t, initial_T):
        return initial_T / np.log(1.0 + t)

    schedules = {
        "Exponential (α = 0.85)": temp_exponential(time, T0, 0.85),
        "Exponential (α = 0.95)": temp_exponential(time, T0, 0.95),
        "Exponential (α = 0.98)": temp_exponential(time, T0, 0.98),
        "Linear (η = 1.0)": temp_linear(time, T0, 1.0),
        "Fast Annealing (T0/(1+t))": temp_fast(time, T0),
        "Logarithmic (T0/ln(1+t))": temp_logarithmic(time, T0)
    }

    # Vẽ 2 biểu đồ:
    # 1. Đường cong nhiệt độ chuẩn hóa T(t)/T0
    # 2. Xác suất chấp nhận bước xấu Δ = 2 theo thời gian: p = exp(-Δ / T(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors_list = ['#d9534f', '#f0ad4e', '#5cb85c', '#5bc0de', '#0275d8', '#6f42c1']
    
    for (name, T_vals), color in zip(schedules.items(), colors_list):
        ax1.plot(time, T_vals / T0, label=name, color=color, linewidth=2)

    ax1.set_title("So sánh các Lịch giảm nhiệt độ $T(t) / T_0$", fontsize=12, weight='bold')
    ax1.set_xlabel("Thời gian bước lặp ($t$)", fontsize=11)
    ax1.set_ylabel("Tỉ lệ nhiệt độ ($T(t) / T_0$)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')

    delta = 2.0  # Bước xấu làm tăng 2 xung đột
    for (name, T_vals), color in zip(schedules.items(), colors_list):
        # tránh chia cho 0
        safe_T = np.maximum(T_vals, 1e-6)
        prob = np.exp(-delta / safe_T)
        prob[T_vals <= 0] = 0
        ax2.plot(time, prob, label=name, color=color, linewidth=2)

    ax2.set_title(f"Xác suất chấp nhận bước xấu ($\\Delta={int(delta)}$) theo thời gian", fontsize=12, weight='bold')
    ax2.set_xlabel("Thời gian bước lặp ($t$)", fontsize=11)
    ax2.set_ylabel("Xác suất $P(\\Delta) = \\exp(-\\Delta / T(t))$", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=9, loc='upper right')

    plt.suptitle("Khảo sát các Lịch giảm nhiệt độ (Cooling Schedules) trong Simulated Annealing", fontsize=14, weight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'sa_cooling_schedules_comparison.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ lịch nhiệt tại: {fig_path}")

if __name__ == '__main__':
    run_schedule_experiments()
