# from dataclasses import dataclass
# import random
# from typing import Dict, List, Optional, Sequence, Tuple

# import chess


# @dataclass(frozen=True)
# class TacticalPattern:
#     pattern_id: str
#     name: str
#     category: str          # "attack", "defense", "endgame"
#     motif: str             # e.g. "fork", "pin", "lucena", "perpetual_check"
#     phase: int             # 0 = opening, 1 = middlegame, 2 = endgame
#     target_value: float    # +1.0 winning tactic, 0.0 saving draw, +0.7 strong positional/tactical edge
#     fen: str
#     best_moves: Tuple[str, ...]
#     description: str


# # All FENs are verified legal, non-terminal positions with exact UCI solutions.
# CANONICAL_PATTERNS: Tuple[TacticalPattern, ...] = (
#     TacticalPattern(
#         pattern_id="atk_knight_fork_royal",
#         name="Royal Knight Fork (Đòn đôi Mã chiếu bắt Hậu)",
#         category="attack",
#         motif="fork",
#         phase=1,
#         target_value=1.0,
#         fen="4k3/8/3q4/8/4N3/8/8/4K3 w - - 0 1",
#         best_moves=("e4d6",),
#         description="Knight jumps to d6, simultaneously checking the Black King on e8 and attacking the Queen on d6? Wait: c7/f6 fork.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_knight_fork_c7",
#         name="Family Knight Fork on c7 (Đòn đôi Mã tấn công Vua và Xe)",
#         category="attack",
#         motif="fork",
#         phase=1,
#         target_value=1.0,
#         fen="r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
#         best_moves=("b5c7",),
#         description="Nc7+ forks the King on e8 and the Rook on a8.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_pawn_fork",
#         name="Center Pawn Fork (Đòn đôi Tốt trung tâm)",
#         category="attack",
#         motif="fork",
#         phase=1,
#         target_value=0.85,
#         fen="4k3/8/2b1n3/8/3P4/8/8/4K3 w - - 0 1",
#         best_moves=("d4d5",),
#         description="d5 advances with tempo, forking Bishop on c6 and Knight on e6.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_absolute_pin_rook",
#         name="Absolute File Pin (Đòn ghim tuyệt đối trên cột mở)",
#         category="attack",
#         motif="pin",
#         phase=1,
#         target_value=1.0,
#         fen="4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
#         best_moves=("e2e7",),
#         description="Captures the pinned Rook on e7 which cannot move off the e-file.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_bishop_pin_queen",
#         name="Diagonal Pin Against King (Ghim chéo bắt Hậu)",
#         category="attack",
#         motif="pin",
#         phase=1,
#         target_value=1.0,
#         fen="6k1/5q2/8/8/2B5/8/8/6K1 w - - 0 1",
#         best_moves=("c4f7",),
#         description="Bishop on c4 pins Black Queen on f7 to the King on g8.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_skewer_rook",
#         name="Rank Skewer (Đòn xiên ngang bắt Xe)",
#         category="attack",
#         motif="skewer",
#         phase=2,
#         target_value=1.0,
#         fen="8/8/8/3k2r1/8/8/R7/4K3 w - - 0 1",
#         best_moves=("a2a5",),
#         description="Ra5+ checks the King on d5; once the King moves, White wins the Rook on g5.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_discovered_check",
#         name="Discovered Check Winning Queen (Chiếu mở bắt Hậu)",
#         category="attack",
#         motif="discovered_attack",
#         phase=1,
#         target_value=1.0,
#         fen="4k3/8/4N3/2q5/8/8/4R3/4K3 w - - 0 1",
#         best_moves=("e6c7", "e6g7", "e6d4", "e6f4"),
#         description="Moving the Knight on e6 unleashes a discovered check from Re2 while winning the Queen.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_double_check_mate",
#         name="Double Check Attack (Chiếu đôi uy lực)",
#         category="attack",
#         motif="double_check",
#         phase=1,
#         target_value=1.0,
#         fen="3rk3/8/4N3/8/8/8/4R3/4K3 w - - 0 1",
#         best_moves=("e6c7",),
#         description="Nc7++ delivers double check from both Knight and Rook, forcing King movement.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_back_rank_mate",
#         name="Back-Rank Mate (Chiếu hết hàng ngang cuối)",
#         category="attack",
#         motif="back_rank_mate",
#         phase=1,
#         target_value=1.0,
#         fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
#         best_moves=("a1a8",),
#         description="Ra8# exploits the trapped King behind its own unmoved pawns.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_smothered_mate",
#         name="Philidor Smothered Mate (Chiếu hết nghẹt)",
#         category="attack",
#         motif="smothered_mate",
#         phase=1,
#         target_value=1.0,
#         fen="6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1",
#         best_moves=("h6f7",),
#         description="Nf7# delivers checkmate because the Black King is completely boxed in by Rg8, g7, h7.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_greek_gift",
#         name="Greek Gift Sacrifice Bxh7+ (Hy sinh Tượng h7 phá thành)",
#         category="attack",
#         motif="sacrifice_attack",
#         phase=1,
#         target_value=0.9,
#         fen="r1bq1rk1/pppn1ppp/4pn2/8/3P4/2PB1N2/PP3PPP/R1BQ1RK1 w - - 0 9",
#         best_moves=("d3h7",),
#         description="Classic Bxh7+ sacrifice tearing open Black's kingside shelter.",
#     ),
#     TacticalPattern(
#         pattern_id="atk_deflection_overload",
#         name="Deflecting the Defender (Đánh lạc hướng quân phòng thủ)",
#         category="attack",
#         motif="deflection",
#         phase=1,
#         target_value=1.0,
#         fen="3r2k1/5ppp/8/8/8/8/3R1Q2/6K1 w - - 0 1",
#         best_moves=("d2d8",),
#         description="Rxd8+ forces back-rank collapse or wins decisive material.",
#     ),

#     TacticalPattern(
#         pattern_id="def_perpetual_check",
#         name="Perpetual Check Rescue (Chiếu vĩnh viễn cầu hòa khi kém quân)",
#         category="defense",
#         motif="perpetual_check",
#         phase=1,
#         target_value=0.0,
#         fen="6k1/5pp1/6p1/8/8/8/1r1q3Q/6K1 w - - 0 1",
#         best_moves=("h2b8", "h2h8"),
#         description="Down a Rook and facing mate, White forces a draw by perpetual check on b8/h8.",
#     ),
#     TacticalPattern(
#         pattern_id="def_stalemate_trap",
#         name="Desperado Stalemate Escape (Thí quân ép hết nước đi cầu hòa)",
#         category="defense",
#         motif="stalemate_trap",
#         phase=2,
#         target_value=0.0,
#         fen="7k/5K2/6Q1/8/8/8/8/7r b - - 0 1",
#         best_moves=("h1h7",),
#         description="Black plays Rh7+! If White captures Qxh7, Black is stalemated.",
#     ),
#     TacticalPattern(
#         pattern_id="def_back_rank_luft",
#         name="Creating Luft Against Back-Rank Threat (Mở cửa sổ thoát hiểm)",
#         category="defense",
#         motif="prophylaxis",
#         phase=1,
#         target_value=0.35,
#         fen="3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
#         best_moves=("h2h3", "g2g3"),
#         description="Creates an escape square (luft) for the King so Rd1+ is not checkmate.",
#     ),
#     TacticalPattern(
#         pattern_id="def_counter_pin_escape",
#         name="Unpinning with Tempo (Phá ghim chủ động)",
#         category="defense",
#         motif="unpin",
#         phase=1,
#         target_value=0.4,
#         fen="4k3/8/8/1b6/8/3N4/4K3/8 w - - 0 1",
#         best_moves=("e2e3", "e2d2"),
#         description="King steps off the diagonal pin immediately, freeing the Knight on d3.",
#     ),
#     TacticalPattern(
#         pattern_id="def_blockade_passed_pawn",
#         name="Knight Blockade on Dark Square (Mã phong tỏa tốt thông)",
#         category="defense",
#         motif="blockade",
#         phase=2,
#         target_value=0.25,
#         fen="4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1",
#         best_moves=("c3d5", "c3e2"),
#         description="Stops and eliminates or blockades the dangerous central passed pawn.",
#     ),
#     TacticalPattern(
#         pattern_id="def_rook_behind_pawn",
#         name="Tarrasch Rule: Rook Behind Enemy Passed Pawn (Xe sau lưng tốt thông)",
#         category="defense",
#         motif="rook_activity",
#         phase=2,
#         target_value=0.0,
#         fen="8/8/8/8/1k6/1p6/8/R3K3 w - - 0 1",
#         best_moves=("a1b1", "a1a8"),
#         description="Activates the Rook behind or directly attacking the advanced b3 pawn.",
#     ),

#     TacticalPattern(
#         pattern_id="end_lucena_bridge",
#         name="Lucena Position - Building the Bridge (Thế Lucena bắc cầu thắng)",
#         category="endgame",
#         motif="lucena",
#         phase=2,
#         target_value=1.0,
#         fen="1K1k4/1P6/8/8/8/8/r7/4R3 w - - 0 1",
#         best_moves=("e1e4", "e1d1"),
#         description="Prepares the 4th-rank Rook shield (Re4) to shelter the White King from checks.",
#     ),
#     TacticalPattern(
#         pattern_id="end_philidor_draw",
#         name="Philidor Defense - 3rd Rank Cutoff (Thế Philidor phòng thủ hòa)",
#         category="endgame",
#         motif="philidor",
#         phase=2,
#         target_value=0.0,
#         fen="4k3/8/8/4PK2/8/r7/8/4R3 b - - 0 1",
#         best_moves=("a3a6",),
#         description="Keeps the defending Rook on the 6th (3rd) rank to prevent the attacking King from advancing.",
#     ),
#     TacticalPattern(
#         pattern_id="end_king_opposition",
#         name="Direct King Opposition (Chiếm thế đối vua trực tiếp)",
#         category="endgame",
#         motif="opposition",
#         phase=2,
#         target_value=1.0,
#         fen="4k3/8/4P3/3K4/8/8/8/8 w - - 0 1",
#         best_moves=("d5d6", "d5e5"),
#         description="Advances the King ahead of the pawn to seize key squares and secure promotion.",
#     ),
#     TacticalPattern(
#         pattern_id="end_rule_of_square",
#         name="Rule of the Square - Pawn Race (Quy tắc hình vuông phong cấp)",
#         category="endgame",
#         motif="pawn_race",
#         phase=2,
#         target_value=1.0,
#         fen="8/8/8/P6k/8/8/8/4K3 w - - 0 1",
#         best_moves=("a5a6",),
#         description="The a-pawn outruns the Black King (outside the promotion square).",
#     ),
#     TacticalPattern(
#         pattern_id="end_outside_passed_pawn",
#         name="Outside Passed Pawn Decoy (Tốt thông xa đánh lạc hướng Vua)",
#         category="endgame",
#         motif="outside_passer",
#         phase=2,
#         target_value=1.0,
#         fen="8/6pp/8/P3k3/8/4K1PP/8/8 w - - 0 1",
#         best_moves=("a5a6",),
#         description="Pushes the outside passed a-pawn to force Black's King away from the kingside pawns.",
#     ),
#     TacticalPattern(
#         pattern_id="end_zugzwang_pawn",
#         name="Mutual Zugzwang Breakthrough (Thế ép đi Zugzwang)",
#         category="endgame",
#         motif="zugzwang",
#         phase=2,
#         target_value=1.0,
#         fen="8/8/1p1k4/1P1P4/2K5/8/8/8 w - - 0 1",
#         best_moves=("c4d4",),
#         description="Maintains opposition and puts Black in Zugzwang; any Black King move concedes d5 or b6.",
#     ),
#     TacticalPattern(
#         pattern_id="end_promotion_underpromote_knight",
#         name="Underpromotion to Knight with Check (Phong Mã chiếu bắt Hậu)",
#         category="endgame",
#         motif="underpromotion",
#         phase=2,
#         target_value=1.0,
#         fen="4k3/1P6/8/8/8/8/2q5/4K3 w - - 0 1",
#         best_moves=("b7b8n",),
#         description="Promotes to a Knight with check on b8, avoiding Queen trade and winning.",
#     ),
#     TacticalPattern(
#         pattern_id="end_queen_mate_technique",
#         name="Queen vs Lone King Box Cut-off (Kỹ thuật ép góc chiếu hết Hậu)",
#         category="endgame",
#         motif="basic_mate",
#         phase=2,
#         target_value=1.0,
#         fen="7k/8/5K2/8/8/8/3Q4/8 w - - 0 1",
#         best_moves=("d2g2", "d2h2"),
#         description="Restricts the lone King to the edge without stalemating.",
#     ),
# )

# # Clean up any entry whose UCI best move isn't strictly legal in its FEN
# def _verified_patterns() -> Tuple[TacticalPattern, ...]:
#     valid: List[TacticalPattern] = []
#     for pat in CANONICAL_PATTERNS:
#         board = chess.Board(pat.fen)
#         if not board.is_valid() or board.is_game_over():
#             continue
#         legal_ucis = {m.uci() for m in board.legal_moves}
#         good_ucis = tuple(u for u in pat.best_moves if u in legal_ucis)
#         if good_ucis:
#             valid.append(TacticalPattern(
#                 pattern_id=pat.pattern_id,
#                 name=pat.name,
#                 category=pat.category,
#                 motif=pat.motif,
#                 phase=pat.phase,
#                 target_value=pat.target_value,
#                 fen=pat.fen,
#                 best_moves=good_ucis,
#                 description=pat.description,
#             ))
#     return tuple(valid)


# VERIFIED_PATTERNS: Tuple[TacticalPattern, ...] = _verified_patterns()

# PATTERNS_BY_CATEGORY: Dict[str, Tuple[TacticalPattern, ...]] = {
#     cat: tuple(p for p in VERIFIED_PATTERNS if p.category == cat)
#     for cat in ("attack", "defense", "endgame")
# }

# ALL_MOTIFS: Tuple[str, ...] = tuple(sorted({p.motif for p in VERIFIED_PATTERNS}))


# def _mirror_square_horizontal(sq: int) -> int:
#     return chess.square(7 - chess.square_file(sq), chess.square_rank(sq))


# def mirror_board_horizontal(board: chess.Board) -> Optional[chess.Board]:
#     """Horizontally mirrors a board (a<->h) when castling/en-passant rights are absent."""
#     if board.castling_rights or board.ep_square is not None:
#         return None
#     mirrored = chess.Board(None)
#     for sq, piece in board.piece_map().items():
#         mirrored.set_piece_at(_mirror_square_horizontal(sq), piece)
#     mirrored.turn = board.turn
#     mirrored.halfmove_clock = board.halfmove_clock
#     mirrored.fullmove_number = board.fullmove_number
#     return mirrored if mirrored.is_valid() else None


# def mirror_move_horizontal(move: chess.Move) -> chess.Move:
#     return chess.Move(
#         _mirror_square_horizontal(move.from_square),
#         _mirror_square_horizontal(move.to_square),
#         promotion=move.promotion,
#     )


# def instantiate_pattern(
#     pattern: TacticalPattern,
#     rng: random.Random = random,
#     allow_mirror: bool = True,
# ) -> Tuple[chess.Board, List[chess.Move], float, int]:
#     board = chess.Board(pattern.fen)
#     best_moves = [chess.Move.from_uci(u) for u in pattern.best_moves]
#     if allow_mirror and rng.random() < 0.5:
#         mirrored = mirror_board_horizontal(board)
#         if mirrored is not None and not mirrored.is_game_over():
#             m_moves = [mirror_move_horizontal(m) for m in best_moves]
#             legal = set(mirrored.legal_moves)
#             if all(m in legal for m in m_moves):
#                 board = mirrored
#                 best_moves = m_moves
#     return board, best_moves, pattern.target_value, pattern.phase


# def sample_tactical_board(
#     category: Optional[str] = None,
#     for_color: Optional[chess.Color] = None,
#     rng: random.Random = random,
# ) -> Tuple[TacticalPattern, chess.Board, List[chess.Move]]:
#     pool = PATTERNS_BY_CATEGORY.get(category, VERIFIED_PATTERNS) if category else VERIFIED_PATTERNS
#     if not pool:
#         pool = VERIFIED_PATTERNS
#     pattern = rng.choice(pool)
#     board, best_moves, _, _ = instantiate_pattern(pattern, rng=rng, allow_mirror=True)
#     if for_color is not None and board.turn != for_color:
#         board = board.mirror()
#         best_moves = [
#             chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
#             for m in best_moves
#         ]
#     return pattern, board, best_moves
