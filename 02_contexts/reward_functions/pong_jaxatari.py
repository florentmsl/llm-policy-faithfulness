def reward_fn(prev_state, state):
    """
    Representative Pong-style reward:
    +1 when the player scores, -1 when the player concedes, else 0.
    """
    if state["player_score"] > prev_state["player_score"]:
        return 1.0
    if state["opponent_score"] > prev_state["opponent_score"]:
        return -1.0
    return 0.0
