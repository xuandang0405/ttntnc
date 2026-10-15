"""
Thực nghiệm và trực quan hóa Bài 1: Mê cung (Maze) với Simple Hill Climbing
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import maze_helper as mh

def run_maze_experiments():
    with open(os.path.join(os.path.dirname(__file__), 'L_maze.txt'), 'r') as f:
        maze_str = f.read()
    maze = mh.parse_maze(maze_str)

    DIR2POS = {
        'N': (-1, 0),
        'E': (0, 1),
        'S': (1, 0),
        'W': (0, -1)
    }

    def manhattan(p1, p2):
        return int(np.sum(np.abs(np.subtract(p1, p2))))

    start_pos = mh.find_pos(maze, 'S')
    goal_pos = mh.find_pos(maze, 'G')

    print(f"Bắt đầu tại Start: {start_pos}, Mục tiêu Goal: {goal_pos}")
    print(f"Khoảng cách Manhattan ban đầu: {manhattan(start_pos, goal_pos)}")

    def search_with_config(order='NESW', random_order=False, seed=42):
        if seed is not None:
            np.random.seed(seed)
        dirs = [DIR2POS[d] for d in order]

        def expand(pos):
            available = [[d, tuple(p)] for d, p in zip(order, np.add(pos, dirs)) if mh.look(maze, tuple(p)) != 'X']
            if random_order:
                np.random.shuffle(available)
            return available

        current = start_pos
        current_obj = manhattan(current, goal_pos)
        path = [current]
        evaluations = 0

        while True:
            neighbors = [n[1] for n in expand(current)]
            if not neighbors:
                break
            objs = [manhattan(n, goal_pos) for n in neighbors]
            evaluations += len(objs)
            best_obj = min(objs)
            best_neighbor = neighbors[objs.index(best_obj)]

            if best_obj < current_obj:
                current = best_neighbor
                current_obj = best_obj
                path.append(current)
            else:
                break

        return {
            'solution': current,
            'goal': goal_pos,
            'reached_goal': (current == goal_pos),
            'start': start_pos,
            'final_h': current_obj,
            'steps': len(path) - 1,
            'evaluations': evaluations,
            'path': path
        }

    configs = [
        ('NESW (Mặc định)', 'NESW', False, None),
        ('SENW (Đảo hướng)', 'SENW', False, None),
        ('WNES (Tây trước)', 'WNES', False, None),
        ('ESWN (Đông trước)', 'ESWN', False, None),
        ('Random 1 (seed=42)', 'NESW', True, 42),
        ('Random 2 (seed=100)', 'NESW', True, 100),
        ('Random 3 (seed=999)', 'NESW', True, 999),
    ]

    results = []
    print("\n--- KẾT QUẢ THỬ NGHIỆM TRÊN CÁC CẤU HÌNH ---")
    for name, ord_str, rnd, seed in configs:
        res = search_with_config(ord_str, rnd, seed)
        results.append((name, res))
        print(f"Cấu hình: {name:<22} | Dừng tại: {str(res['solution']):<8} | h(n): {res['final_h']:<2} | Số bước: {res['steps']:<2} | Đạt đích: {res['reached_goal']}")

    # Vẽ biểu đồ bản đồ mê cung cùng đường đi của các cấu hình
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])
    
    # 3 cấu hình tiêu biểu để vẽ
    selected_configs = [results[0], results[1], results[4]]
    
    for idx, (name, res) in enumerate(selected_configs):
        ax = axes[idx]
        maze_display = mh.maze_to_matrix(maze)
        
        # Vẽ đường đi
        for r, c in res['path']:
            if (r, c) != start_pos and (r, c) != goal_pos:
                maze_display[r, c] = 4 # Màu 'P' (đỏ)

        ax.imshow(maze_display, cmap=cmap, norm=colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
        ax.text(start_pos[1], start_pos[0], "S", fontsize=11, color="white", ha='center', va='center', weight='bold')
        ax.text(goal_pos[1], goal_pos[0], "G", fontsize=11, color="white", ha='center', va='center', weight='bold')
        
        # Đánh dấu vị trí kẹt
        stopped = res['solution']
        if stopped != goal_pos:
            ax.scatter([stopped[1]], [stopped[0]], s=120, color='yellow', edgecolors='red', linewidth=2, zorder=5)
            ax.text(stopped[1], stopped[0] - 0.6, "Trap!", color='yellow', weight='bold', fontsize=9, ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="darkred", alpha=0.8))

        ax.set_title(f"{name}\nDừng tại {stopped}, h={res['final_h']}, {res['steps']} bước", fontsize=11)
        ax.set_xticks(range(maze.shape[1]))
        ax.set_yticks(range(maze.shape[0]))
        ax.grid(color='gray', linestyle=':', linewidth=0.5)

    plt.suptitle("Bài toán Mê cung (L-Maze) với Simple Hill Climbing:\nHiện tượng mắc kẹt tại Cực trị cục bộ (Local Optima)", fontsize=13, weight='bold')
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'maze_hill_climbing_traps.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nĐã lưu ảnh phân tích mê cung tại: {fig_path}")

if __name__ == '__main__':
    run_maze_experiments()
