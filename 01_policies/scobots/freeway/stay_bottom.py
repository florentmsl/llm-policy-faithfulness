def play(state):
    if state['D(Chicken1, Car2).x'] <= 111:
        if state['Car3.x'] <= 146:
            if state['Car9.x[t-1]'] <= 156:
                if state['Car6.x[t-1]'] <= 154:
                    return 2  # DOWN
                else:
                    if state['Car9.x'] <= 98:
                        return 2  # DOWN
                    else:
                        if state['Car1.x[t-1]'] <= 109:
                            return 2  # DOWN
                        else:
                            if state['D(Chicken1, Car3).x'] <= 32:
                                return 2  # DOWN
                            else:
                                return 0  # NOOP
            else:
                if state['Car1.x#1'] <= 109:
                    return 2  # DOWN
                else:
                    if state['V(Car3).x'] <= 1:
                        return 2  # DOWN
                    else:
                        return 0  # NOOP
        else:
            if state['Car9.x#1'] <= 118:
                if state['Car10.x[t-1]'] <= -2:
                    if state['D(Chicken1, Car2).x'] <= 52:
                        return 2  # DOWN
                    else:
                        return 0  # NOOP
                else:
                    return 2  # DOWN
            else:
                if state['Car1.x[t-1]'] <= 74:
                    return 2  # DOWN
                else:
                    if state['Car10.x#1'] <= 17:
                        if state['Car10.x[t-1]'] <= -1:
                            return 0  # NOOP
                        else:
                            return 2  # DOWN
                    else:
                        return 0  # NOOP
    else:
        if state['V(Car1).x'] <= 59:
            if state['Car3.x'] <= 129:
                return 0  # NOOP
            else:
                if state['Car10.x#1'] <= 110:
                    return 1  # UP
                else:
                    return 0  # NOOP
        else:
            if state['Car8.x[t-1]'] <= 77:
                if state['Car8.x#1'] <= 25:
                    return 1  # UP
                else:
                    return 0  # NOOP
            else:
                return 2  # DOWN
