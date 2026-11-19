# Stockfish teacher and coach

Needs a Stockfish executable (https://stockfishchess.org/download/):
`$env:STOCKFISH_PATH = "C:\path\stockfish.exe"` (or put it on PATH).
The executable must be native and runnable in UCI mode. Stockfish binaries are not
downloaded or copied by the app.

Run teacher commands from the repository root:
```powershell
python src\stockfish\pretrain_stockfish.py generate --out data\sf.npz --games 400 --workers 8 --nodes 20000
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm AlphaZero --color white
python src\stockfish\pretrain_stockfish.py train --data data\sf.npz --algorithm DQN --color black
```

They can also be run after `cd src`:
```powershell
cd src
python -m stockfish.pretrain_stockfish generate --out ..\data\sf.npz --games 400 --workers 8 --nodes 20000
```

## Teacher: pretrain from Stockfish labels
Checkpoints (`white_alphazero.pt`, ...) load in the app like any other model.
`--init file.pt` continues an existing checkpoint (one colour at a time).
Use tens of thousands of positions or more: in a small test the value head learned
quickly, but move agreement needs much more data. DQN is trained by Q-value regression
on Stockfish's candidate moves plus a bound on all other legal moves.

## Coach: Stockfish evaluation as the reward-shaping potential
`$env:CHESS_AI_STOCKFISH_COACH = "1"` before starting the app. Material balance is replaced by
Stockfish's evaluation (pawns, clipped to +-10) at each of the agent's decision points; a finished
game has potential 0. Tune with `CHESS_AI_STOCKFISH_COACH_NODES` (default 2000).

## Process lifecycle
Every pool waits for running analyses and closes its UCI engine processes during
`close()` or context-manager exit. The app closes the shared coach in a `finally` block
when Pygame exits, and every pool also registers an `atexit` fallback. Teacher worker
processes close their pool explicitly.

If a process still appears in Task Manager after an abnormal termination, close it from Task
Manager; normal app, script, and worker exits should not leave one alive.

Settings: `stockfish_config.py`. Tests: `python -m unittest tests.test_stockfish_pipeline`
(skipped when Stockfish is not found).
