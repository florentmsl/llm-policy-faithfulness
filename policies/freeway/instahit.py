def play(state):
    if state["ED(Agent, Hazard_1)"] <= 2:
        return 0
    if state["D(Agent, Hazard_1).y"] > 0:
        return 1
    if state["D(Agent, Hazard_1).y"] < 0:
        return 2
    return 0
