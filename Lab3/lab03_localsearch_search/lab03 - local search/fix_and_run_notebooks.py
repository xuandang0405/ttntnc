import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Sửa n_queens.ipynb
path_nq = os.path.join(DIR, 'n_queens.ipynb')
nb_nq = nbformat.read(path_nq, as_version=4)
nb_nq.cells[6].source = """import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import time
import math
import pandas as pd

np.random.seed(1234)
random.seed(1234)

def random_board(n):
    \"\"\"Creates a random board of size n x n. Note that only a single queen is placed in each column!\"\"\"
    return np.random.randint(0, n, size=n)

def comb2(n): return n*(n-1)//2 # this is n choose 2 equivalent to math.comb(n, 2); // is int division

def conflicts(board):
    \"\"\"Calculate the number of conflicts, i.e., the objective function.\"\"\"
    n = len(board)
    horizontal_cnt = [0] * n
    diagonal1_cnt = [0] * 2 * n
    diagonal2_cnt = [0] * 2 * n
    
    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1
    
    return sum(map(comb2, horizontal_cnt + diagonal1_cnt + diagonal2_cnt))

def show_board(board, cols = ['white', 'gray'], fontsize = 36):  
    \"\"\"display the board\"\"\"
    n = len(board)
    display = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (((i+j) % 2) != 0): 
                display[i, j] = 1
    
    cmap = colors.ListedColormap(cols)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(display, cmap=cmap, norm=colors.BoundaryNorm(range(len(cols)+1), cmap.N))
    ax.set_xticks([])
    ax.set_yticks([])
    
    for j in range(n):
        plt.text(j, board[j], u"\\u265B", fontsize=fontsize, 
                 horizontalalignment='center',
                 verticalalignment='center')
    
    print(f"Bàn cờ với {conflicts(board)} cặp xung đột.")
    plt.show()
"""
nbformat.write(nb_nq, path_nq)

# 2. Sửa traveling_salesman_problem.ipynb
path_tsp = os.path.join(DIR, 'traveling_salesman_problem.ipynb')
nb_tsp = nbformat.read(path_tsp, as_version=4)
nb_tsp.cells[3].source = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time

np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

# make the results repeatable
np.random.seed(1234)
random.seed(1234)
"""
nbformat.write(nb_tsp, path_tsp)
print("Đã sửa imports thành công! Đang tiến hành thực thi lại...")

# 3. Thực thi lại cả 5 notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
notebooks = [
    'Maze_Example_LocalSearch.ipynb',
    'Simulated_Annealing_Schedules.ipynb',
    'n_queens_fast_conflict_calculation_with_numba.ipynb',
    'n_queens.ipynb',
    'traveling_salesman_problem.ipynb'
]

for nb_name in notebooks:
    p = os.path.join(DIR, nb_name)
    print(f"\n>>> Thực thi {nb_name}...")
    with open(p, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if 'id' not in cell:
            cell['id'] = nbformat.v4.new_code_cell().id
    ep.preprocess(nb, {'metadata': {'path': DIR}})
    with open(p, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Thực thi thành công: {nb_name}")

print("\nHoàn tất 100% tất cả các notebooks với đầy đủ kết quả đầu ra!")
