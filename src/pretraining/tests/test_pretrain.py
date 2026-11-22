import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import chess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ["PRETRAIN_DEMO"] = "1"

from pretrain.common import cp_to_value, list_shards, read_jsonl_zst, split_of
from pretrain.dataset import PositionStream, collate
from make_synthetic import make_pgn

SF = shutil.which("stockfish") or "/usr/games/stockfish"
HAVE_SF = os.path.exists(SF)


def run(*args):
    env = dict(os.environ, PYTHONPATH=ROOT, PRETRAIN_DEMO="1")
    r = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@unittest.skipUnless(HAVE_SF, "stockfish not installed")
class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pgn = make_pgn(os.path.join(cls.tmp, "b.pgn.zst"), SF)
        cls.shards = os.path.join(cls.tmp, "shards")
        cls.out = run("pretrain.prepare", cls.pgn, "--out", cls.shards, "--min-elo", "2000", "--games-per-shard", "10")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def records(self, d=None):
        return [r for p in list_shards(d or self.shards) for r in read_jsonl_zst(p)]

    def test_prepare_filters(self):
        m = json.load(open(os.path.join(self.shards, "manifest.json")))
        st = m["stats"]
        self.assertEqual(st["unfinished"], 1)
        self.assertEqual(st["elo"], 1)
        self.assertEqual(st["short_draw"], 1)
        self.assertEqual(st["non_standard_start"], 1)
        self.assertEqual(st["duplicate"], 1)
        self.assertGreaterEqual(st["illegal_or_unparsable"], 1)
        self.assertGreaterEqual(st["length"], 1)                       # the zero-ply game
        self.assertEqual(m["games"], len(self.records()))
        self.assertGreater(m["games"], 30)

    def test_pgn_eval_alignment(self):
        recs = [r for r in self.records() if "sf" in r]
        self.assertTrue(recs, "half of the synthetic games carry [%eval]")
        r = recs[0]
        self.assertEqual(len(r["sf"]["cp"]), len(r["moves"].split()))
        self.assertIsNone(r["sf"]["cp"][0])
        # annotation == 100 * material (white POV) of the position AFTER each ply; the shard must store it
        # against the *next* position, in side-to-move POV.
        from pretrain.eval_join import material_white
        board, checked = chess.Board(), 0
        for p, uci in enumerate(r["moves"].split()):
            if r["sf"]["cp"][p] is not None:
                expected = 100 * material_white(board) * (1 if board.turn == chess.WHITE else -1)
                self.assertEqual(r["sf"]["cp"][p], expected, (p, board.turn))
                checked += 1
            board.push_uci(uci)
        self.assertGreater(checked, 20)

    def test_stockfish_labels_and_dataset(self):
        lab = os.path.join(self.tmp, "labeled")
        run("pretrain.stockfish_label", self.shards, "--out", lab, "--engine", SF, "--nodes", "50",
            "--every", "5", "--workers", "2")
        recs = self.records(lab)
        r = recs[0]
        n = len(r["moves"].split())
        self.assertEqual(r["sf"]["src"], "stockfish")
        self.assertEqual([i for i, c in enumerate(r["sf"]["cp"]) if c is not None], list(range(0, n, 5)))
        # best moves are legal in their positions
        board = chess.Board()
        for p, uci in enumerate(r["moves"].split()):
            if r["sf"]["best"][p]:
                self.assertIn(chess.Move.from_uci(r["sf"]["best"][p]), board.legal_moves)
            board.push_uci(uci)
        # dataset: split is by event, targets point at the played / best move, value blends result and SF
        tr = PositionStream(lab, "train", 20, demo=True, shuffle_buffer=64, seed=3)
        va = PositionStream(lab, "val", 20, demo=True, shuffle=False, shuffle_buffer=1)
        ev_tr = {r["event"] for r in recs if split_of(r["event"], 20) == "train"}
        ev_va = {r["event"] for r in recs if split_of(r["event"], 20) == "val"}
        self.assertFalse(ev_tr & ev_va)
        self.assertEqual(sum(1 for _ in tr), tr.count_positions())
        self.assertEqual(sum(1 for _ in va), va.count_positions())
        samples = list(tr)
        self.assertTrue(any(s["sf_target"] >= 0 for s in samples))
        for s in samples[:50]:
            self.assertEqual(s["state"].shape, (56, 8, 8))
            self.assertTrue(0 <= s["target"] < len(s["legal"]))
            self.assertTrue(np.all(np.diff(s["legal"].astype(int)) > 0))
            self.assertTrue(-1.0 <= s["value"] <= 1.0)
        b = collate(samples[:16])
        self.assertEqual(b["state"].shape[0], 16)
        self.assertEqual(b["legal"].dtype, torch.int16)

    def test_train_smoke_checkpoint(self):
        out = os.path.join(self.tmp, "run")
        log = run("pretrain.train", "--shards", self.shards, "--out", out, "--demo", "--epochs", "4", "--batch", "64",
                  "--workers", "0", "--val-pct", "20", "--warmup", "5", "--eval-every", "20", "--log-every", "10",
                  "--lr", "2e-3", "--val-batches", "4", "--max-steps", "60")
        self.assertIn("[val]", log)
        first = [float(l.split("pol ")[1].split()[0]) for l in log.splitlines() if l.startswith("step 10/")][0]
        last = [float(l.split("pol ")[1].split()[0]) for l in log.splitlines() if l.startswith("step 60/")][0]
        self.assertLess(last, first)                                    # policy loss goes down
        ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu")
        self.assertEqual(ck["format"], 3)
        sys.path.insert(0, os.path.join(ROOT, "pretrain", "demo_project"))
        from model import PolicyValueNet
        PolicyValueNet().load_state_dict(ck["net"])


class Units(unittest.TestCase):
    def test_load_project_uses_main_project_by_default(self):
        from pretrain.project import load_project
        proj = load_project(demo=False)
        self.assertFalse(proj.demo)
        self.assertEqual(proj.INPUT_CHANNELS, 56)
        self.assertEqual(proj.ACTION_SIZE, 20480)
        self.assertNotIn("demo_project", proj.PolicyValueNet.__module__)

    def test_cp_to_value_matches_lichess_curve(self):
        for cp in (-300, 0, 120, 800):
            win = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)
            self.assertAlmostEqual(cp_to_value(cp), win / 50 - 1, places=3)

    def test_eval_join_end_to_end(self):
        import zstandard as zstd
        from pretrain.common import ShardWriter
        tmp = tempfile.mkdtemp()
        try:
            w = ShardWriter(os.path.join(tmp, "s"), 10)
            w.add({"id": "g", "event": "E", "round": "1", "white": "a", "black": "b", "w_elo": 2500, "b_elo": 2500,
                   "result": 1.0, "moves": "e2e4 e7e5 g1f3 b8c6"})
            w.close()
            board, lines = chess.Board(), []
            # white is ahead in material in the 3rd position only: after 1.e4 e5 the eval DB says +40 for white
            for uci, cp in zip(["e2e4", "e7e5", "g1f3"], [30, 25, 45]):
                board.push_uci(uci)
            board = chess.Board()
            for ply, uci in enumerate(["e2e4", "e7e5", "g1f3", "b8c6"]):
                key = " ".join(board.fen().split()[:4])
                if ply in (1, 2):
                    # stm-POV convention: cp is relative to side to move
                    lines.append(json.dumps({"fen": key, "evals": [
                        {"knodes": 1, "depth": 20, "pvs": [{"cp": 30 if ply == 2 else 40, "line": "d2d4 d7d5"}]},
                        {"knodes": 1, "depth": 10, "pvs": [{"cp": -999, "line": "a2a3"}]}]}))
                board.push_uci(uci)
            lines.append(json.dumps({"fen": "8/8/8/8/8/8/8/K6k w - -", "evals": []}))
            evals = os.path.join(tmp, "e.jsonl.zst")
            open(evals, "wb").write(zstd.ZstdCompressor().compress("\n".join(lines).encode()))
            run("pretrain.eval_join", os.path.join(tmp, "s"), "--evals", evals, "--out", os.path.join(tmp, "o"),
                "--pov", "stm")
            rec = next(read_jsonl_zst(list_shards(os.path.join(tmp, "o"))[0]))
            self.assertEqual(rec["sf"]["cp"], [None, 40, 30, None])      # highest-depth eval wins
            self.assertEqual(rec["sf"]["best"][1], "d2d4")
            run("pretrain.eval_join", os.path.join(tmp, "s"), "--evals", evals, "--out", os.path.join(tmp, "o2"),
                "--pov", "white")
            rec = next(read_jsonl_zst(list_shards(os.path.join(tmp, "o2"))[0]))
            self.assertEqual(rec["sf"]["cp"], [None, -40, 30, None])     # black to move at ply 1 -> sign flips
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
