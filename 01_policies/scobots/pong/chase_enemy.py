def play(state):
    delta = state['D(Player1, Enemy1).y']
    if delta > 4:
        return 3  # LEFT
    if delta < -4:
        return 2  # RIGHT
    return 0  # NOOP
