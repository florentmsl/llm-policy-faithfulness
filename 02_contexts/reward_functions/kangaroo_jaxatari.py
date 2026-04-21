def reward_fn(prev_state, state):
    """
    Representative Kangaroo-style reward:
    +1 whenever the in-game score increases (climbing up levels, punching
    fruit/bell, reaching the joey), else 0.
    """
    if state["score"] > prev_state["score"]:
        return 1.0
    return 0.0
