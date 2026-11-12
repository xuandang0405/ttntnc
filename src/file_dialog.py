import os
import subprocess
import sys
 
import pygame
 
from config import CKPT_DIR
 
_SCRIPT = r'''
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    sys.exit(3)
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
print(filedialog.askopenfilename(title=sys.argv[1], initialdir=sys.argv[2],
      filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")]))
'''

def pick_model_file(title="Select a model file"):
    start = os.path.abspath(CKPT_DIR)
    if not os.path.isdir(start):
        start = os.getcwd()
    try:
        r = subprocess.run([sys.executable, "-c", _SCRIPT, title, start], capture_output=True, text=True)
    except OSError as e:
        return None, f"Could not open file dialog: {e}"
    pygame.event.clear()

    if r.returncode == 3:
        return None, "File dialog unavailable - install tkinter (e.g. sudo apt install python3-tk)"
    return (r.stdout.strip() or None), None


def load_model_dialog(game, color):
    name = "White" if color else "Black"          # chess.WHITE is True
    path, err = pick_model_file(f"Select the {name} model file")
    if err:
        return err
    if not path:
        return ""
    try:
        game.load_model(color, path)
    except Exception as ex:
        return f"Load failed: {ex}"
    return f"Loaded {name}: {os.path.basename(path)}"