def play(state):
    delta = state['D(Agent, Obj_B).y']
    if delta > 4:
        return 3
    if delta < -4:
        return 2
    return 0
