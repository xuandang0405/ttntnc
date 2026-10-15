"""
Script thực thi toàn bộ các cells trong cả 5 notebooks và lưu lại kết quả hiển thị (outputs).
Tác giả: Nhóm sinh viên thực hiện Lab 03
"""

import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

DIR = os.path.dirname(os.path.abspath(__file__))

notebooks = [
    'Maze_Example_LocalSearch.ipynb',
    'Simulated_Annealing_Schedules.ipynb',
    'n_queens_fast_conflict_calculation_with_numba.ipynb',
    'n_queens.ipynb',
    'traveling_salesman_problem.ipynb'
]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_name in notebooks:
    nb_path = os.path.join(DIR, nb_name)
    print(f"\n>>> Đang thực thi notebook: {nb_name}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Đảm bảo tất cả cells có id theo chuẩn nbformat
    for cell in nb.cells:
        if 'id' not in cell:
            cell['id'] = nbformat.v4.new_code_cell().id

    try:
        ep.preprocess(nb, {'metadata': {'path': DIR}})
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Đã thực thi và lưu thành công: {nb_name}")
    except Exception as e:
        print(f"Lỗi khi thực thi {nb_name}: {e}")

print("\nHoàn tất thực thi toàn bộ 5 notebooks!")
