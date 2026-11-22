INITIAL_ELO = 1200.0
ELO_K_FACTOR = 32.0


def update_elo(rating_a, rating_b, score_a, k_factor=ELO_K_FACTOR):
    if score_a not in (0.0, 0.5, 1.0):
        raise ValueError("score_a must be 0, 0.5, or 1")
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    change = k_factor * (score_a - expected_a)
    return rating_a + change, rating_b - change
