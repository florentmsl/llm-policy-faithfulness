def reward_fn(prev_state, state):
    """
    Representative Skiing-style reward:
    Small negative reward each step (time penalty), large bonus when a gate
    is passed correctly, and a large positive reward when the finish line is
    reached. Missing a gate adds an extra time penalty.
    """
    r = -0.01  # constant time penalty per step
    if state["gates_passed"] > prev_state["gates_passed"]:
        r += 1.0
    if state["finished"] and not prev_state["finished"]:
        r += 10.0
    if state["gates_missed"] > prev_state["gates_missed"]:
        r -= 1.0
    return r
