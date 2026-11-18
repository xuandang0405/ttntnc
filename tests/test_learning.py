import json
import chess
import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(2)

from backend.core.env import ChessEnv
from model.learning.agents import LearnedAlphaBetaAgent, NeuralAgent, value_moves
from model.learning.data import ChessDataset, collate, generate, split_records
from model.learning.encoding import ACTION_SIZE, CHANNELS, encode_board, id_move, move_id, position_key
from model.learning.models import ModelConfig, PolicyValueNet, load_model, masked_logits, save_model
from model.learning.selfplay import td_target


@pytest.fixture
def checkpoint(tmp_path):
    model = PolicyValueNet(ModelConfig('cnn', width=4, hidden=16))
    for parameter in model.parameters():
        parameter.data.zero_()
    path = tmp_path / 'model.pt'
    save_model(path, model, policy_trained=True, method='fixture')
    return path


def test_encoder_distinguishes_castling_ep_and_repetition():
    start = chess.Board()
    no_rights = start.copy()
    no_rights.castling_rights = 0
    assert encode_board(start).shape == (CHANNELS, 8, 8)
    assert np.array_equal(encode_board(start)[:12], encode_board(no_rights)[:12])
    assert not np.array_equal(encode_board(start)[13:17], encode_board(no_rights)[13:17])
    start.push_uci('e2e4')
    # A non-capturable EP square must match the FEN reconstruction at inference.
    assert np.array_equal(encode_board(start), encode_board(chess.Board(start.fen())))
    ep = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    assert encode_board(ep)[17].sum() == 1
    repeat = chess.Board()
    for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        repeat.push_uci(uci)
    assert encode_board(repeat)[19, 0, 0] == .5
    assert np.array_equal(encode_board(repeat), encode_board(chess.Board(repeat.fen()), .5))


def test_all_four_promotions_have_distinct_action_ids():
    moves = [chess.Move.from_uci('a7a8' + p) for p in 'nbrq']
    assert len({move_id(m) for m in moves}) == 4
    assert [id_move(move_id(m)) for m in moves] == moves


@pytest.mark.parametrize('kind', ['mlp', 'cnn'])
def test_models_support_batch_one_and_checkpoint_roundtrip(kind, tmp_path):
    model = PolicyValueNet(ModelConfig(kind, width=4, hidden=16)).eval()
    x = torch.from_numpy(encode_board(chess.Board())).unsqueeze(0)
    logits, value = model(x)
    assert logits.shape == (1, ACTION_SIZE) and value.shape == (1,)
    assert -1 <= value.item() <= 1
    path = tmp_path / 'weights.pt'
    save_model(path, model, policy_trained=True)
    loaded, metadata = load_model(path)
    assert torch.equal(logits, loaded(x)[0]) and metadata['policy_trained']


def test_mask_ignores_illegal_high_logit_and_preserves_underpromotion(checkpoint):
    model, _ = load_model(checkpoint)
    board = chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    legal_id = move_id(chess.Move.from_uci('a7a8n'))
    with torch.no_grad():
        model.policy.bias[0] = 1000  # Illegal a1a1.
        model.policy.bias[legal_id] = 100
    save_model(checkpoint, model, policy_trained=True)
    env = ChessEnv(fen=board.fen())
    action, info = NeuralAgent(checkpoint).select_action(env)
    assert action == 'a7a8n'
    assert sum(info['top_moves'].values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        masked_logits(torch.zeros(1, ACTION_SIZE), torch.zeros(1, ACTION_SIZE, dtype=torch.bool))


@pytest.mark.parametrize('black', [False, True])
def test_learned_search_and_value_find_mate_without_changing_history(checkpoint, black):
    board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    if black:
        board = board.mirror()
    env = ChessEnv(max_ply=100_000, fen=board.fen())
    before = env.board.fen()
    for agent in [NeuralAgent(checkpoint, 'value'), LearnedAlphaBetaAgent(checkpoint, depth=2)]:
        action, _ = agent.select_action(env, 500)
        after = env.board.copy()
        after.push_uci(action)
        assert after.is_checkmate()
        assert env.board.fen() == before and not env.board.move_stack


def test_search_deadline_returns_legal_fallback(checkpoint):
    env = ChessEnv()
    action, _ = LearnedAlphaBetaAgent(checkpoint, depth=8).select_action(env, 1)
    assert action in env.legal_actions() and not env.board.move_stack


def test_td_target_changes_perspective_and_bootstraps_truncation(checkpoint):
    v = torch.tensor([.8], requires_grad=True)
    assert td_target(v).item() == pytest.approx(-.8)
    assert not td_target(v).requires_grad
    assert td_target(v, terminal_reward=1).item() == 1
    assert td_target(v, terminal_reward=0).item() == 0
    model, _ = load_model(checkpoint)
    save_model(checkpoint, model, policy_trained=False, method='TD(0)-value')
    with pytest.raises(ValueError, match='value'):
        NeuralAgent(checkpoint, 'policy')


def test_generation_split_and_samples_do_not_leak_transpositions(tmp_path):
    data = tmp_path / 'samples.jsonl'
    info = generate(data, games=12, max_plies=12, teacher_ms=5, exploration=.5, seed=7)
    records = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    folds, manifest = split_records(records, seed=7)
    keys = [set(position_key(r['fen']) for r in fold) for fold in folds]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    games = [set(x['games']) for x in manifest.values()]
    assert not games[0] & games[1] and not games[0] & games[2]
    batch = collate([ChessDataset(folds[0])[0]])
    assert batch[3][0, batch[1][0]]
    assert all(g['result'] is None for g in info['game_summary'] if g['termination'] == 'truncated')
    with pytest.raises(FileExistsError):
        generate(data, games=3)
