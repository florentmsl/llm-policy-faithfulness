def play(state):
    if state['Ball1.y'] <= 35:
        if state['C(Player1, Ball1).y'] <= 56:
            if state['ED(Player1, Ball1)'] <= 110:
                return 1  # FIRE
            else:
                return 3  # LEFT
        else:
            if state['C(Player1, Ball1).y'] <= 79:
                if state['DV(Player1).y'] <= -8:
                    return 0  # NOOP
                else:
                    if state['Player1.y[t-1]'] <= 144:
                        return 3  # LEFT
                    else:
                        if state['V(Player1).x'] <= 2:
                            return 3  # LEFT
                        else:
                            return 0  # NOOP
            else:
                return 0  # NOOP
    else:
        if state['Ball1.y#1'] <= 59:
            if state['Ball1.x'] <= 89:
                return 1  # FIRE
            else:
                return 2  # RIGHT
        else:
            return 2  # RIGHT
