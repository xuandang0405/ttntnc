# algorithms-rs

Rust search-core primitives ported from the supplied Python AlphaZero code.

## Included in this first port

- `Node` / `Arena`: contiguous node storage with stable integer node IDs.
- `expand`: creates child nodes from legal action indices and already-computed priors.
- `select_puct`: same PUCT expression and opponent-view Q convention as Python `_select`.
- `backup`: same reverse-path visit/value update and alternating sign as Python `_backup`.

## Kept in Python

- Chess board transitions, legal move generation, terminal/draw checks: still owned by `python-chess`.
- Neural inference/training: stays in PyTorch/CUDA.
- Dirichlet sampling: remains in Python. Pass the post-noise priors to `expand` to preserve the existing NumPy RNG behavior.

The `algorithms_rs.MctsTree` PyO3 binding exposes node expansion, PUCT selection,
backup, and root visit counts. Python keeps `chess.Move` mappings and batches
network inference. This keeps the Python/Rust boundary outside neural inference
and lets AlphaZero use the native tree while preserving the Python implementation
as the default.

## Build and test

```bash
cargo test
python -m pip install .
```

Set `CHESS_AI_RUST_MCTS=1` before starting the application to opt into Rust MCTS
(in PowerShell: `$env:CHESS_AI_RUST_MCTS = "1"`).
The extension is optional; without the variable, AlphaZero uses Python MCTS.
If Rust mode is requested but the extension is not installed, startup fails
with an explicit error instead of silently switching backends.

## Semantics and data contract

`action_indices` must be in the same order as the Python `moves.keys()` order.
The Rust arena preserves that order, so exact-score ties select the first edge
just like the Python loop. `backup` expects a path ordered root-to-leaf and a
value expressed from the leaf side-to-move perspective. Values are flipped once
per ply.

`expand` expects normalized legal-action priors from the caller; it validates
finiteness and non-negativity but does not renormalize them. For noisy roots,
compute the same mixture in Python and pass the resulting values.
