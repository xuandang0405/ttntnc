import os
import sys
import time
import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import Delaunay

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# PART 1: GRAPH COLORING CSP HELPER FUNCTIONS AND ALGORITHMS
# -------------------------------------------------------------

def generate_delaunay_graph(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    points = np.random.randint(100, size=(n, 2))
    order = np.argsort(points[:, 0])
    points = points[order, :]
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    
    adj = {}
    for k in range(len(indptr) - 1):
        adj[str(k)] = [str(x) for x in np.sort(indices[indptr[k]:indptr[k+1]])]
        
    variables = [str(i) for i in range(n)]
    constraints = []
    for i in range(n):
        for j in adj[str(i)]:
            if int(i) < int(j):
                constraints.append((str(i), str(j)))
                
    return points, tri, variables, adj, constraints

def create_csp(variables, adj, constraints, num_colors=4):
    palette = ['red', 'blue', 'green', 'orange', 'purple', 'yellow'][:num_colors]
    domains = {v: list(palette) for v in variables}
    return {
        'variables': variables,
        'domains': domains,
        'constraints': constraints,
        'adj': adj,
        'palette': palette
    }

def complete(assignment, csp):
    return len(assignment) == len(csp['variables'])

def consistent(assignment, csp):
    for u, v in csp['constraints']:
        if u in assignment and v in assignment:
            if assignment[u] == assignment[v]:
                return False
    return True

# 1. Standard Backtracking
def backtrack_baseline(csp):
    nodes = 0
    def bt(assignment):
        nonlocal nodes
        nodes += 1
        if complete(assignment, csp):
            return assignment
        unassigned = [v for v in csp['variables'] if v not in assignment]
        var = unassigned[0]
        for val in csp['domains'][var]:
            assignment[var] = val
            if consistent(assignment, csp):
                res = bt(assignment)
                if res is not None:
                    return res
            del assignment[var]
        return None
    
    t0 = time.perf_counter()
    sol = bt({})
    t1 = time.perf_counter()
    return sol, nodes, t1 - t0

# 2. MRV Backtracking
def get_legal_values(var, assignment, csp):
    legal = []
    for val in csp['domains'][var]:
        conflict = False
        for nb in csp['adj'][var]:
            if nb in assignment and assignment[nb] == val:
                conflict = True
                break
        if not conflict:
            legal.append(val)
    return legal

def select_mrv_var(assignment, csp):
    unassigned = [v for v in csp['variables'] if v not in assignment]
    if not unassigned:
        return None
    # MRV: min remaining legal values; tie break: highest degree (most unassigned neighbors)
    return min(unassigned, key=lambda v: (
        len(get_legal_values(v, assignment, csp)),
        -sum(1 for nb in csp['adj'][v] if nb not in assignment)
    ))

def backtrack_mrv(csp):
    nodes = 0
    def bt(assignment):
        nonlocal nodes
        nodes += 1
        if complete(assignment, csp):
            return assignment
        var = select_mrv_var(assignment, csp)
        for val in csp['domains'][var]:
            assignment[var] = val
            if consistent(assignment, csp):
                res = bt(assignment)
                if res is not None:
                    return res
            del assignment[var]
        return None
    
    t0 = time.perf_counter()
    sol = bt({})
    t1 = time.perf_counter()
    return sol, nodes, t1 - t0

# 3. MRV + LCV Backtracking
def order_domain_lcv(var, assignment, csp):
    unassigned_neighbors = [nb for nb in csp['adj'][var] if nb not in assignment]
    def count_eliminations(val):
        count = 0
        for nb in unassigned_neighbors:
            if val in get_legal_values(nb, assignment, csp):
                count += 1
        return count
    return sorted(csp['domains'][var], key=count_eliminations)

def backtrack_mrv_lcv(csp):
    nodes = 0
    def bt(assignment):
        nonlocal nodes
        nodes += 1
        if complete(assignment, csp):
            return assignment
        var = select_mrv_var(assignment, csp)
        for val in order_domain_lcv(var, assignment, csp):
            assignment[var] = val
            if consistent(assignment, csp):
                res = bt(assignment)
                if res is not None:
                    return res
            del assignment[var]
        return None
    
    t0 = time.perf_counter()
    sol = bt({})
    t1 = time.perf_counter()
    return sol, nodes, t1 - t0

# 4. Forward Checking Backtracking
def backtrack_forward_checking(csp):
    nodes = 0
    def bt_fc(assignment, cur_domains):
        nonlocal nodes
        nodes += 1
        if complete(assignment, csp):
            return assignment
        unassigned = [v for v in csp['variables'] if v not in assignment]
        # MRV on current domains
        var = min(unassigned, key=lambda v: (
            len(cur_domains[v]),
            -sum(1 for nb in csp['adj'][v] if nb not in assignment)
        ))
        
        for val in cur_domains[var]:
            assignment[var] = val
            # Forward check: prune val from neighbors
            new_domains = {v: list(cur_domains[v]) for v in csp['variables']}
            valid = True
            for nb in csp['adj'][var]:
                if nb not in assignment:
                    if val in new_domains[nb]:
                        new_domains[nb].remove(val)
                        if len(new_domains[nb]) == 0:
                            valid = False
                            break
            if valid:
                res = bt_fc(assignment, new_domains)
                if res is not None:
                    return res
            del assignment[var]
        return None
    
    t0 = time.perf_counter()
    sol = bt_fc({}, {v: list(csp['domains'][v]) for v in csp['variables']})
    t1 = time.perf_counter()
    return sol, nodes, t1 - t0

# 5. Min-Conflicts Local Search
def min_conflicts_graph(csp, max_steps=1000):
    t0 = time.perf_counter()
    variables = csp['variables']
    assignment = {v: random.choice(csp['domains'][v]) for v in variables}
    
    for step in range(max_steps):
        conflicted = [
            v for v in variables 
            if any(assignment[nb] == assignment[v] for nb in csp['adj'][v])
        ]
        if not conflicted:
            t1 = time.perf_counter()
            return assignment, step, t1 - t0, True
            
        var = random.choice(conflicted)
        counts = {
            val: sum(1 for nb in csp['adj'][var] if assignment[nb] == val)
            for val in csp['domains'][var]
        }
        min_c = min(counts.values())
        best_vals = [val for val, c in counts.items() if c == min_c]
        assignment[var] = random.choice(best_vals)
        
    t1 = time.perf_counter()
    return assignment, max_steps, t1 - t0, False

# -------------------------------------------------------------
# PART 2: N-QUEENS LOCAL SEARCH ALGORITHMS
# -------------------------------------------------------------

def comb2(n): 
    return n * (n - 1) // 2

def conflicts(board):
    n = len(board)
    h_cnt = [0] * n
    d1 = [0] * 2 * n
    d2 = [0] * 2 * n
    for i in range(n):
        h_cnt[board[i]] += 1
        d1[i + board[i]] += 1
        d2[i - board[i] + n] += 1
    return sum(map(comb2, h_cnt + d1 + d2))

def random_board(n):
    return list(np.random.randint(0, n, size=n))

# Task 1: Steepest-Ascent Hill Climbing
def steepest_ascent_hc(board):
    t0 = time.perf_counter()
    n = len(board)
    current = list(board)
    curr_h = conflicts(current)
    steps = 0
    evals = 0
    history = [curr_h]
    
    while curr_h > 0:
        best_neighbor = None
        best_h = curr_h
        for col in range(n):
            orig_row = current[col]
            for row in range(n):
                if row == orig_row:
                    continue
                current[col] = row
                evals += 1
                h = conflicts(current)
                if h < best_h:
                    best_h = h
                    best_neighbor = (col, row)
            current[col] = orig_row
            
        if best_neighbor is None:
            break
        current[best_neighbor[0]] = best_neighbor[1]
        curr_h = best_h
        steps += 1
        history.append(curr_h)
        
    t1 = time.perf_counter()
    return current, curr_h, steps, evals, t1 - t0, history

# Task 2: Inefficient Stochastic Hill Climbing
def inefficient_stochastic_hc(board):
    t0 = time.perf_counter()
    n = len(board)
    current = list(board)
    curr_h = conflicts(current)
    steps = 0
    evals = 0
    history = [curr_h]
    
    while curr_h > 0:
        improving_moves = []
        for col in range(n):
            orig_row = current[col]
            for row in range(n):
                if row == orig_row:
                    continue
                current[col] = row
                evals += 1
                h = conflicts(current)
                if h < curr_h:
                    improving_moves.append(((col, row), h))
            current[col] = orig_row
            
        if not improving_moves:
            break
        move, new_h = random.choice(improving_moves)
        current[move[0]] = move[1]
        curr_h = new_h
        steps += 1
        history.append(curr_h)
        
    t1 = time.perf_counter()
    return current, curr_h, steps, evals, t1 - t0, history

# Task 3: Efficient Stochastic Hill Climbing
def efficient_stochastic_hc(board, max_no_improve=None):
    t0 = time.perf_counter()
    n = len(board)
    if max_no_improve is None:
        max_no_improve = max(50, 4 * n * (n - 1))
    current = list(board)
    curr_h = conflicts(current)
    steps = 0
    evals = 0
    history = [curr_h]
    no_improve = 0
    
    while curr_h > 0 and no_improve < max_no_improve:
        col = random.randint(0, n - 1)
        orig_row = current[col]
        row = random.choice([r for r in range(n) if r != orig_row])
        current[col] = row
        evals += 1
        h = conflicts(current)
        if h < curr_h:
            curr_h = h
            steps += 1
            history.append(curr_h)
            no_improve = 0
        else:
            current[col] = orig_row
            no_improve += 1
            
    t1 = time.perf_counter()
    return current, curr_h, steps, evals, t1 - t0, history

# Task 4: Random Restart Hill Climbing
def random_restart_hc(algo_fn, n, max_restarts=100):
    t0 = time.perf_counter()
    total_steps = 0
    total_evals = 0
    best_h = float('inf')
    best_board = None
    
    for r in range(max_restarts):
        board = random_board(n)
        sol, h, steps, evals, _, _ = algo_fn(board)
        total_steps += steps
        total_evals += evals
        if h < best_h:
            best_h = h
            best_board = sol
        if best_h == 0:
            t1 = time.perf_counter()
            return best_board, best_h, r + 1, total_steps, total_evals, t1 - t0, True
            
    t1 = time.perf_counter()
    return best_board, best_h, max_restarts, total_steps, total_evals, t1 - t0, (best_h == 0)

# Task 5: Simulated Annealing
def simulated_annealing(board, schedule='geometric', T0=10.0, alpha=0.95, beta=0.01, max_steps=2000):
    t0 = time.perf_counter()
    n = len(board)
    current = list(board)
    curr_h = conflicts(current)
    best_board = list(current)
    best_h = curr_h
    history = [curr_h]
    T = T0
    evals = 0
    steps = 0
    
    for step in range(1, max_steps + 1):
        if curr_h == 0:
            break
        col = random.randint(0, n - 1)
        orig_row = current[col]
        row = random.choice([r for r in range(n) if r != orig_row])
        current[col] = row
        evals += 1
        h = conflicts(current)
        dh = h - curr_h
        
        if dh < 0 or (T > 1e-6 and random.random() < math.exp(-dh / T)):
            curr_h = h
            steps += 1
            if curr_h < best_h:
                best_h = curr_h
                best_board = list(current)
        else:
            current[col] = orig_row
            
        history.append(curr_h)
        
        if schedule == 'geometric':
            T = max(1e-4, T * alpha)
        elif schedule == 'linear':
            T = max(1e-4, T0 / (1 + beta * step))
            
    t1 = time.perf_counter()
    return best_board, best_h, steps, evals, t1 - t0, history

# Advanced Task: Alternative Local Move Operators
def operator_single_step(board):
    n = len(board)
    col = random.randint(0, n - 1)
    orig_row = board[col]
    delta = random.choice([-1, 1])
    new_row = orig_row + delta
    if 0 <= new_row < n:
        new_board = list(board)
        new_board[col] = new_row
        return new_board
    return None

def operator_column_swap(board):
    n = len(board)
    c1, c2 = random.sample(range(n), 2)
    new_board = list(board)
    new_board[c1], new_board[c2] = new_board[c2], new_board[c1]
    return new_board

def operator_dual_queen(board):
    n = len(board)
    c1, c2 = random.sample(range(n), 2)
    new_board = list(board)
    new_board[c1] = random.choice([r for r in range(n) if r != board[c1]])
    new_board[c2] = random.choice([r for r in range(n) if r != board[c2]])
    return new_board

def operator_adaptive(board):
    n = len(board)
    # Count conflicts per column
    col_conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                col_conflicts[i] += 1
                col_conflicts[j] += 1
    max_c = max(col_conflicts)
    if max_c > 0 and random.random() < 0.7:
        # Focus on worst conflicted queens
        worst_cols = [c for c, count in enumerate(col_conflicts) if count == max_c]
        col = random.choice(worst_cols)
        new_board = list(board)
        new_board[col] = random.choice([r for r in range(n) if r != board[col]])
        return new_board
    else:
        # Fallback to column swap
        return operator_column_swap(board)

def efficient_hc_with_operator(board, operator_fn, max_no_improve=None):
    t0 = time.perf_counter()
    n = len(board)
    if max_no_improve is None:
        max_no_improve = max(50, 4 * n * (n - 1))
    current = list(board)
    curr_h = conflicts(current)
    steps = 0
    evals = 0
    history = [curr_h]
    no_improve = 0
    
    while curr_h > 0 and no_improve < max_no_improve:
        neighbor = operator_fn(current)
        if neighbor is None:
            no_improve += 1
            continue
        evals += 1
        h = conflicts(neighbor)
        if h < curr_h:
            current = neighbor
            curr_h = h
            steps += 1
            history.append(curr_h)
            no_improve = 0
        else:
            no_improve += 1
            
    t1 = time.perf_counter()
    return current, curr_h, steps, evals, t1 - t0, history

# -------------------------------------------------------------
# EXECUTION OF EXPERIMENTS AND FIGURE GENERATION
# -------------------------------------------------------------

print("Running Graph Coloring Experiments...")

# 1. Graph Coloring Sample Visualization
points, tri, variables, adj, constraints = generate_delaunay_graph(10, seed=1111)
csp = create_csp(variables, adj, constraints, num_colors=4)
sol_baseline, nodes_base, _ = backtrack_baseline(csp)

fig, ax = plt.subplots(figsize=(7, 6))
ax.triplot(points[:, 0], points[:, 1], tri.simplices, color='darkgray', linewidth=1.5, zorder=1)
for i in range(len(points)):
    color = sol_baseline[str(i)]
    ax.plot(points[i, 0], points[i, 1], 'o', color=color, markersize=24, markeredgecolor='black', markeredgewidth=1.5, zorder=2)
    ax.annotate(str(i), (points[i, 0], points[i, 1]), color='white' if color in ['blue', 'green', 'purple', 'red'] else 'black',
                fontsize=11, weight='bold', ha='center', va='center', zorder=3)
ax.set_title("Kết quả tô màu đồ thị phẳng Delaunay (n = 10, 4 màu)", fontsize=13, pad=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('figures/fig1_graph_coloring_sample.png', dpi=300)
plt.close()
print("Saved figures/fig1_graph_coloring_sample.png")

# 2. Graph Coloring Performance across n
n_sizes = [10, 15, 20, 25, 30]
gc_results = {
    'n': n_sizes,
    'avg_edges': [],
    'nodes_base': [],
    'nodes_mrv': [],
    'nodes_mrv_lcv': [],
    'nodes_fc': [],
    'time_base': [],
    'time_mrv': [],
    'time_mrv_lcv': [],
    'time_fc': [],
    'time_mc': [],
    'success_mc_4col': [],
    'success_mc_3col': [],
    'success_fc_3col': []
}

for n in n_sizes:
    edges_list = []
    base_nodes_list = []
    mrv_nodes_list = []
    mrv_lcv_nodes_list = []
    fc_nodes_list = []
    
    base_time_list = []
    mrv_time_list = []
    mrv_lcv_time_list = []
    fc_time_list = []
    mc_time_list = []
    
    mc_4col_succ = 0
    mc_3col_succ = 0
    fc_3col_succ = 0
    
    num_instances = 15
    for s in range(num_instances):
        p, tr, v, a, c = generate_delaunay_graph(n, seed=100 + s * 13)
        edges_list.append(len(c))
        
        # 4 colors
        csp4 = create_csp(v, a, c, num_colors=4)
        _, nb, tb = backtrack_baseline(csp4)
        _, nm, tm = backtrack_mrv(csp4)
        _, nml, tml = backtrack_mrv_lcv(csp4)
        _, nfc, tfc = backtrack_forward_checking(csp4)
        _, _, tmc, ok_mc4 = min_conflicts_graph(csp4, max_steps=500)
        
        base_nodes_list.append(nb)
        mrv_nodes_list.append(nm)
        mrv_lcv_nodes_list.append(nml)
        fc_nodes_list.append(nfc)
        
        base_time_list.append(tb)
        mrv_time_list.append(tm)
        mrv_lcv_time_list.append(tml)
        fc_time_list.append(tfc)
        mc_time_list.append(tmc)
        if ok_mc4: mc_4col_succ += 1
        
        # 3 colors
        csp3 = create_csp(v, a, c, num_colors=3)
        sol_fc3, _, _ = backtrack_forward_checking(csp3)
        if sol_fc3 is not None: fc_3col_succ += 1
        _, _, _, ok_mc3 = min_conflicts_graph(csp3, max_steps=500)
        if ok_mc3: mc_3col_succ += 1
        
    gc_results['avg_edges'].append(np.mean(edges_list))
    gc_results['nodes_base'].append(np.mean(base_nodes_list))
    gc_results['nodes_mrv'].append(np.mean(mrv_nodes_list))
    gc_results['nodes_mrv_lcv'].append(np.mean(mrv_lcv_nodes_list))
    gc_results['nodes_fc'].append(np.mean(fc_nodes_list))
    
    gc_results['time_base'].append(np.mean(base_time_list) * 1000)
    gc_results['time_mrv'].append(np.mean(mrv_time_list) * 1000)
    gc_results['time_mrv_lcv'].append(np.mean(mrv_lcv_time_list) * 1000)
    gc_results['time_fc'].append(np.mean(fc_time_list) * 1000)
    gc_results['time_mc'].append(np.mean(mc_time_list) * 1000)
    
    gc_results['success_mc_4col'].append(mc_4col_succ / num_instances * 100)
    gc_results['success_mc_3col'].append(mc_3col_succ / num_instances * 100)
    gc_results['success_fc_3col'].append(fc_3col_succ / num_instances * 100)

# Save Fig 2: Node Count Comparison (Log scale)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n_sizes, gc_results['nodes_base'], 'o-', label='Baseline Backtracking', color='black', linewidth=1.5)
ax.plot(n_sizes, gc_results['nodes_mrv'], 's--', label='MRV Backtracking', color='dimgray', linewidth=1.5)
ax.plot(n_sizes, gc_results['nodes_mrv_lcv'], '^-.', label='MRV + LCV', color='gray', linewidth=1.5)
ax.plot(n_sizes, gc_results['nodes_fc'], 'd:', label='Forward Checking', color='silver', linewidth=2, markeredgecolor='black')
ax.set_xlabel('Số lượng đỉnh (n)', fontsize=11)
ax.set_ylabel('Số nút duyệt trung bình (COUNT)', fontsize=11)
ax.set_title('So sánh số nút duyệt (COUNT) giữa các chiến lược tìm kiếm (4 màu)', fontsize=12)
ax.set_yscale('log')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig2_graph_coloring_nodes.png', dpi=300)
plt.close()
print("Saved figures/fig2_graph_coloring_nodes.png")

# Save Fig 3: Runtime Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n_sizes, gc_results['time_base'], 'o-', label='Baseline Backtracking', color='black', linewidth=1.5)
ax.plot(n_sizes, gc_results['time_mrv'], 's--', label='MRV Backtracking', color='dimgray', linewidth=1.5)
ax.plot(n_sizes, gc_results['time_fc'], 'd:', label='Forward Checking', color='gray', linewidth=1.5)
ax.plot(n_sizes, gc_results['time_mc'], 'x-', label='Min-Conflicts (Local Search)', color='silver', linewidth=1.5, markeredgecolor='black')
ax.set_xlabel('Số lượng đỉnh (n)', fontsize=11)
ax.set_ylabel('Thời gian thực thi trung bình (ms)', fontsize=11)
ax.set_title('So sánh thời gian chạy thực nghiệm theo kích thước n (4 màu)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig3_graph_coloring_runtime.png', dpi=300)
plt.close()
print("Saved figures/fig3_graph_coloring_runtime.png")

# Save Fig 4: 3-color vs 4-color Success Rate
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.25
x = np.arange(len(n_sizes))
ax.bar(x - width, gc_results['success_mc_4col'], width, label='Min-Conflicts (4 màu)', color='dimgray', edgecolor='black')
ax.bar(x, gc_results['success_fc_3col'], width, label='Forward Checking (3 màu)', color='darkgray', edgecolor='black')
ax.bar(x + width, gc_results['success_mc_3col'], width, label='Min-Conflicts (3 màu)', color='lightgray', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(n_sizes)
ax.set_xlabel('Số lượng đỉnh (n)', fontsize=11)
ax.set_ylabel('Tỷ lệ tìm được nghiệm (%)', fontsize=11)
ax.set_title('Tỷ lệ thành công giải bài toán tô màu với 4 màu và 3 màu', fontsize=12)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig4_graph_coloring_3color_vs_4color.png', dpi=300)
plt.close()
print("Saved figures/fig4_graph_coloring_3color_vs_4color.png")

with open('figures/gc_results.json', 'w') as f:
    json.dump(gc_results, f, indent=2)

# -------------------------------------------------------------
# PART 3: N-QUEENS EXPERIMENTS
# -------------------------------------------------------------

print("Running N-Queens Experiments...")

# 1. Visualization of Solved 4-Queens and 8-Queens Boards
def plot_two_boards(b4, b8, filename='figures/fig5_nqueens_boards.png'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, b, title in zip(axes, [b4, b8], ['Nghiệm 4-Queens (h = 0)', 'Nghiệm 8-Queens (h = 0)']):
        n = len(b)
        display = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 != 0:
                    display[i, j] = 1
        cmap = colors.ListedColormap(['#f0d9b5', '#b58863'])
        ax.imshow(display, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, pad=8)
        font_sz = 32 if n == 4 else 22
        for j in range(n):
            ax.text(j, b[j], u"\u265B", fontsize=font_sz, color='black',
                    ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

b4_sol = [1, 3, 0, 2]
# Find an 8-queens solution
b8_sol = None
while b8_sol is None:
    _, h, _, _, _, _ = efficient_stochastic_hc(random_board(8))
    if h == 0:
        b8_sol, _, _, _, _, _ = efficient_stochastic_hc(random_board(8))
        if conflicts(b8_sol) == 0:
            break
        b8_sol = None
plot_two_boards(b4_sol, b8_sol)
print("Saved figures/fig5_nqueens_boards.png")

# 2. Benchmark 100 runs for n=4 and n=8 across algorithms
n_queens_table = []
algorithms = [
    ('Steepest asc. HC', lambda b: steepest_ascent_hc(b)),
    ('Ineff. Stochastic HC', lambda b: inefficient_stochastic_hc(b)),
    ('Efficient Stochastic HC', lambda b: efficient_stochastic_hc(b)),
    ('Simulated Annealing (Geometric)', lambda b: simulated_annealing(b, schedule='geometric')),
    ('Simulated Annealing (Linear)', lambda b: simulated_annealing(b, schedule='linear')),
    ('Random Restart (Steepest)', lambda b: random_restart_hc(steepest_ascent_hc, len(b), 100)),
    ('Random Restart (Ineff. Sto)', lambda b: random_restart_hc(inefficient_stochastic_hc, len(b), 100)),
    ('Random Restart (Eff. Sto)', lambda b: random_restart_hc(efficient_stochastic_hc, len(b), 100))
]

NUM_RUNS = 100
n_queens_stats = {}

for size in [4, 8]:
    n_queens_stats[size] = {}
    print(f"Testing Board Size n = {size}...")
    for name, fn in algorithms:
        runtimes = []
        final_conflicts = []
        evals_list = []
        success_count = 0
        
        for run_idx in range(NUM_RUNS):
            b_init = random_board(size)
            if 'Random Restart' in name:
                _, h, restarts, st, ev, dur, succ = fn(b_init)
            else:
                _, h, st, ev, dur, _ = fn(b_init)
                succ = (h == 0)
                
            runtimes.append(dur)
            final_conflicts.append(h)
            evals_list.append(ev)
            if succ:
                success_count += 1
                
        avg_time = np.mean(runtimes)
        avg_h = np.mean(final_conflicts)
        pct_optimal = (success_count / NUM_RUNS) * 100
        avg_evals = np.mean(evals_list)
        
        n_queens_stats[size][name] = {
            'avg_time': avg_time,
            'avg_h': avg_h,
            'pct_optimal': pct_optimal,
            'avg_evals': avg_evals
        }
        n_queens_table.append({
            'Algorithm': name,
            'Board size': size,
            'Avg. Run time (s)': round(avg_time, 5),
            'Avg. conflicts': round(avg_h, 3),
            '% optimal': round(pct_optimal, 1),
            'Avg. evals': round(avg_evals, 1)
        })

with open('figures/n_queens_stats.json', 'w') as f:
    json.dump(n_queens_stats, f, indent=2)

# 3. Algorithm Convergence (Fig 6)
# Run representative runs on 8-queens
rep_board = random_board(8)
_, _, _, _, _, hist_steepest = steepest_ascent_hc(list(rep_board))
_, _, _, _, _, hist_ineff = inefficient_stochastic_hc(list(rep_board))
_, _, _, _, _, hist_eff = efficient_stochastic_hc(list(rep_board))
_, _, _, _, _, hist_sa_geom = simulated_annealing(list(rep_board), schedule='geometric')
_, _, _, _, _, hist_sa_lin = simulated_annealing(list(rep_board), schedule='linear')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(hist_steepest, 'o-', label='Steepest-ascent HC', color='black', markevery=max(1, len(hist_steepest)//8))
ax.plot(hist_ineff, 's--', label='Ineff. Stochastic HC', color='dimgray', markevery=max(1, len(hist_ineff)//8))
ax.plot(hist_eff, '^-.', label='Efficient Stochastic HC', color='gray', markevery=max(1, len(hist_eff)//8))
ax.plot(hist_sa_geom[:150], 'd:', label='Simulated Annealing (Geometric)', color='darkgray', markevery=15)
ax.plot(hist_sa_lin[:150], 'x-', label='Simulated Annealing (Linear)', color='silver', markevery=15, markeredgecolor='black')
ax.set_xlabel('Số bước lặp cải thiện (Steps / Iterations)', fontsize=11)
ax.set_ylabel('Số cặp xung đột h(board)', fontsize=11)
ax.set_title('Đồ thị hội tụ hàm mục tiêu h của các thuật toán trên 8-Queens', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig6_convergence_8queens.png', dpi=300)
plt.close()
print("Saved figures/fig6_convergence_8queens.png")

# 4. SA Cooling Schedule Comparison (Fig 7)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(hist_sa_geom[:200], label='Lịch làm nguội hình học: T = T0 * 0.95^k', color='black', linewidth=1.5)
ax.plot(hist_sa_lin[:200], '--', label='Lịch làm nguội tuyến tính/điều hòa: T = T0 / (1 + 0.01k)', color='gray', linewidth=1.5)
ax.set_xlabel('Vòng lặp (Step)', fontsize=11)
ax.set_ylabel('Số xung đột h(board)', fontsize=11)
ax.set_title('So sánh động thái tìm kiếm của Simulated Annealing qua 2 lịch làm nguội (8-Queens)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig7_sa_cooling_comparison.png', dpi=300)
plt.close()
print("Saved figures/fig7_sa_cooling_comparison.png")

# 5. Problem Size Scalability Log-log Plot (Fig 8)
scalability_n = [4, 8, 12, 16, 20]
time_steepest_scale = []
time_eff_scale = []
time_sa_scale = []

for n in scalability_n:
    t_s, t_e, t_sa = [], [], []
    for _ in range(30):
        b = random_board(n)
        _, _, _, _, dur_s, _ = steepest_ascent_hc(b)
        _, _, _, _, dur_e, _ = efficient_stochastic_hc(b)
        _, _, _, _, dur_sa, _ = simulated_annealing(b, max_steps=1500)
        t_s.append(dur_s)
        t_e.append(dur_e)
        t_sa.append(dur_sa)
    time_steepest_scale.append(np.mean(t_s))
    time_eff_scale.append(np.mean(t_e))
    time_sa_scale.append(np.mean(t_sa))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(scalability_n, time_steepest_scale, 'o-', label='Steepest-ascent HC', color='black', linewidth=1.5)
ax.loglog(scalability_n, time_eff_scale, 's--', label='Efficient Stochastic HC', color='dimgray', linewidth=1.5)
ax.loglog(scalability_n, time_sa_scale, '^-.', label='Simulated Annealing', color='gray', linewidth=1.5)

# Fit slope for empirical Big O
log_n = np.log(scalability_n)
slope_s, _ = np.polyfit(log_n, np.log(time_steepest_scale), 1)
slope_e, _ = np.polyfit(log_n, np.log(time_eff_scale), 1)
slope_sa, _ = np.polyfit(log_n, np.log(time_sa_scale), 1)

ax.set_xlabel('Kích thước bàn cờ n (log scale)', fontsize=11)
ax.set_ylabel('Thời gian chạy trung bình (s) (log scale)', fontsize=11)
ax.set_title(f'Đồ thị Log-Log về độ co giãn kích thước bài toán n ∈ [4, 20]', fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig8_loglog_scalability.png', dpi=300)
plt.close()
print("Saved figures/fig8_loglog_scalability.png")
print(f"Slopes (empirical complexity exponent): Steepest={slope_s:.2f}, Eff={slope_e:.2f}, SA={slope_sa:.2f}")

# 6. Advanced Task: Move Operators Experiments (Figs 9, 10, 11)
print("Running Advanced Task: Move Operators...")
operators = [
    ('Single-step move', operator_single_step),
    ('Column swap', operator_column_swap),
    ('Dual-queen move', operator_dual_queen),
    ('Adaptive move', operator_adaptive)
]

op_results = {}
for n in [8, 12]:
    op_results[n] = {}
    for op_name, op_fn in operators:
        h_finals = []
        times = []
        success_times = []
        histories = []
        
        for _ in range(100):
            b = random_board(n)
            sol, h_final, steps, evals, dur, hist = efficient_hc_with_operator(b, op_fn)
            h_finals.append(h_final)
            times.append(dur)
            histories.append(hist)
            if h_final == 0:
                success_times.append(dur)
                
        # average quality over iterations (padded to max 100 steps)
        max_len = 80
        padded_h = []
        for h_track in histories:
            last = h_track[-1]
            pad = h_track + [last] * (max_len - len(h_track))
            padded_h.append(pad[:max_len])
        avg_curve = np.mean(padded_h, axis=0)
        
        op_results[n][op_name] = {
            'avg_curve': list(avg_curve),
            'h_finals': h_finals,
            'avg_time_success': np.mean(success_times) if success_times else None,
            'success_rate': len(success_times) / 100 * 100,
            'avg_h_final': np.mean(h_finals)
        }

# Fig 9: Average Solution Quality over Iterations (n=8)
fig, ax = plt.subplots(figsize=(7, 4.5))
markers = ['o', 's', '^', 'd']
linestyles = ['-', '--', '-.', ':']
colors_list = ['black', 'dimgray', 'gray', 'darkgray']
for (op_name, _), m, ls, c in zip(operators, markers, linestyles, colors_list):
    ax.plot(op_results[8][op_name]['avg_curve'], label=op_name, linestyle=ls, color=c, linewidth=1.5)
ax.set_xlabel('Số bước cải thiện (Iterations)', fontsize=11)
ax.set_ylabel('Chất lượng nghiệm trung bình h(board)', fontsize=11)
ax.set_title('Chất lượng nghiệm trung bình qua các bước trên 8-Queens', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig9_advanced_operators_quality.png', dpi=300)
plt.close()
print("Saved figures/fig9_advanced_operators_quality.png")

# Fig 10: Distribution of Final Solution Qualities (Boxplot n=8 and n=12)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, n in zip(axes, [8, 12]):
    data = [op_results[n][op_name]['h_finals'] for op_name, _ in operators]
    labels = ['Single-step', 'Col Swap', 'Dual-queen', 'Adaptive']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor='#e0e0e0', color='black', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.8)
    ax.set_ylabel('Số xung đột cuối cùng h', fontsize=11)
    ax.set_title(f'Phân phối số xung đột cuối ({n}-Queens)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/fig10_advanced_operators_distribution.png', dpi=300)
plt.close()
print("Saved figures/fig10_advanced_operators_distribution.png")

# Fig 11: Average Time to Solution for Successful Runs
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ['Single-step', 'Col Swap', 'Dual-queen', 'Adaptive']
times_8 = [op_results[8][name]['avg_time_success'] or 0 for name, _ in operators]
times_12 = [op_results[12][name]['avg_time_success'] or 0 for name, _ in operators]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, times_8, width, label='8-Queens', color='dimgray', edgecolor='black')
ax.bar(x + width/2, times_12, width, label='12-Queens', color='lightgray', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Thời gian trung bình đạt nghiệm (s)', fontsize=11)
ax.set_title('Thời gian trung bình tìm ra lời giải hoàn hảo (h = 0)', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig11_advanced_operators_runtime.png', dpi=300)
plt.close()
print("Saved figures/fig11_advanced_operators_runtime.png")

# Save table summary
print("\n=== SUMMARY TABLE OF N-QUEENS EXPERIMENTS ===")
for row in n_queens_table:
    print(row)

print("\nAll experiments finished successfully!")
