def play(state):
    if state['Obj_A.y'] <= 35:
        if state['C(Agent, Obj_A).y'] <= 56:
            if state['ED(Agent, Obj_A)'] <= 110:
                return 1
            else:
                return 3
        else:
            if state['C(Agent, Obj_A).y'] <= 79:
                if state['DV(Agent).y'] <= -8:
                    return 0
                else:
                    if state['Agent.y[t-1]'] <= 144:
                        return 3
                    else:
                        if state['V(Agent).x'] <= 2:
                            return 3
                        else:
                            return 0
            else:
                return 0
    else:
        if state['Obj_A.y#1'] <= 59:
            if state['Obj_A.x'] <= 89:
                return 1
            else:
                return 2
        else:
            return 2
