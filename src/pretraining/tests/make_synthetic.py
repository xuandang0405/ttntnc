import io
import random
import shutil

import chess
import chess.engine
import chess.pgn
import zstandard as zstd


def make_pgn(path, engine_path, n_games=40, plies=60, nodes=60, seed=1):
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    chunks = []

    def finish(game, board, result, w=2650, b=2610, event="Test Open", with_eval=False):
        game.headers.update({"Event": event, "Site": f"https://lichess.org/broadcast/x/r{rng.random():.6f}",
                             "White": f"W{rng.randint(0, 9)}", "Black": f"B{rng.randint(0, 9)}", "Result": result,
                             "WhiteElo": str(w), "BlackElo": str(b), "TimeControl": "5400+30"})
        chunks.append(game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)))

    for g in range(n_games):
        board, game = chess.Board(), chess.pgn.Game()
        node = game
        for ply in range(plies):
            if board.is_game_over():
                break
            if ply < 6:
                move = rng.choice(list(board.legal_moves))
                info = None
            else:
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                move = info["pv"][0]
            board.push(move)
            node = node.add_variation(move)
            node.comment = "[%clk 1:30:00]"
            if g % 2 == 0 and info is not None:                      # half of the games carry [%eval]
                mat = sum(v * (len(board.pieces(pt, True)) - len(board.pieces(pt, False)))
                          for pt, v in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9)))
                node.comment += f" [%eval {mat:.2f}]"             # exact function of the position (white POV)
        res = board.result(claim_draw=False) if board.is_game_over() else rng.choice(["1-0", "0-1", "1/2-1/2"])
        finish(game, board, res, event=f"Event {g % 5}")
    # deliberately bad / filtered records
    g = chess.pgn.Game(); finish(g, chess.Board(), "1-0")                                 # zero plies
    g = chess.pgn.Game(); b = chess.Board(); n = g
    for _ in range(40):
        m = rng.choice(list(b.legal_moves)); b.push(m); n = n.add_variation(m)
    finish(g, b, "*")                                                                      # unfinished
    g = chess.pgn.Game(); b = chess.Board(); n = g
    for _ in range(40):
        m = rng.choice(list(b.legal_moves)); b.push(m); n = n.add_variation(m)
    finish(g, b, "1-0", w=1500, b=1400)                                                    # low elo
    g = chess.pgn.Game(); b = chess.Board(); n = g
    for _ in range(24):
        m = rng.choice(list(b.legal_moves)); b.push(m); n = n.add_variation(m)
    finish(g, b, "1/2-1/2")                                                                # short draw
    g = chess.pgn.Game(); g.setup(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")); b = g.board(); n = g
    for _ in range(30):
        m = rng.choice(list(b.legal_moves)); b.push(m); n = n.add_variation(m)
        if b.is_game_over(): break
    finish(g, b, "1-0")                                                                    # custom start
    engine.quit()
    chunks.append(chunks[0])                                                               # duplicate game
    head, _, body = chunks[1].partition("\n\n")
    chunks.append(head + "\n\n" + body.replace("1. ", "1. Ke4 ", 1))                       # illegal first move
    text = "\n\n".join(chunks) + "\n\n"
    with open(path, "wb") as fh:
        fh.write(zstd.ZstdCompressor().compress(text.encode()))
    return path
