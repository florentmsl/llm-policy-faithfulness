def play(state):
    if state['D(Agent, Hazard_2).x'] <= 111:
        if state['Hazard_3.x'] <= 146:
            if state['Hazard_9.x[t-1]'] <= 156:
                if state['Hazard_6.x[t-1]'] <= 154:
                    return 2
                else:
                    if state['Hazard_9.x'] <= 98:
                        return 2
                    else:
                        if state['Hazard_1.x[t-1]'] <= 109:
                            return 2
                        else:
                            if state['D(Agent, Hazard_3).x'] <= 32:
                                return 2
                            else:
                                return 0
            else:
                if state['Hazard_1.x#1'] <= 109:
                    return 2
                else:
                    if state['V(Hazard_3).x'] <= 1:
                        return 2
                    else:
                        return 0
        else:
            if state['Hazard_9.x#1'] <= 118:
                if state['Hazard_10.x[t-1]'] <= -2:
                    if state['D(Agent, Hazard_2).x'] <= 52:
                        return 2
                    else:
                        return 0
                else:
                    return 2
            else:
                if state['Hazard_1.x[t-1]'] <= 74:
                    return 2
                else:
                    if state['Hazard_10.x#1'] <= 17:
                        if state['Hazard_10.x[t-1]'] <= -1:
                            return 0
                        else:
                            return 2
                    else:
                        return 0
    else:
        if state['V(Hazard_1).x'] <= 59:
            if state['Hazard_3.x'] <= 129:
                return 0
            else:
                if state['Hazard_10.x#1'] <= 110:
                    return 1
                else:
                    return 0
        else:
            if state['Hazard_8.x[t-1]'] <= 77:
                if state['Hazard_8.x#1'] <= 25:
                    return 1
                else:
                    return 0
            else:
                return 2
