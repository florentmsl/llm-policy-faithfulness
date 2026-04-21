def reward_fn(prev_state, state):
    """
    Representative Seaquest-style reward:
    +1 whenever the in-game score increases (kill shark, rescue divers), else 0.
    """
    if state["score"] > prev_state["score"]:
        return 1.0
    return 0.0
