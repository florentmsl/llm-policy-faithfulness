def reward_fn(prev_state, state):
    """
    Representative Freeway-style reward:
    +1 when Chicken1 completes a crossing, else 0.
    """
    if state["Chicken1_crossed_top"] and not prev_state["Chicken1_crossed_top"]:
        return 1.0
    return 0.0
