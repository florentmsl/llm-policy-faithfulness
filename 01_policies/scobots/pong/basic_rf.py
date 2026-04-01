def play(state):
    if state['D(Player1, Ball1).y'] <= -4:
        if state['DV(Ball1).y'] <= -3:
            if state['D(Player1, Ball1).y'] <= -9:
                if state['DV(Ball1).y'] <= -7:
                    if state['DV(Player1).y'] <= -19:
                        if state['D(Player1, Ball1).y'] <= -13:
                            if state['V(Player1).x'] <= 19:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['ED(Player1, Ball1)'] <= 92:
                                return 1  # FIRE
                            else:
                                return 2  # RIGHT
                    else:
                        if state['ED(Player1, Ball1)'] <= 33:
                            if state['C(Player1, Ball1).x'] <= 135:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            if state['D(Player1, Ball1).y'] <= -17:
                                return 2  # RIGHT
                            else:
                                return 1  # FIRE
                else:
                    if state['V(Player1).x'] <= 5:
                        if state['D(Player1, Ball1).y'] <= -18:
                            if state['D(Player1, Ball1).y'] <= -23:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['Ball1.x#1'] <= 134:
                                return 1  # FIRE
                            else:
                                return 2  # RIGHT
                    else:
                        if state['DV(Player1).y'] <= 21:
                            if state['V(Player1).x'] <= 13:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['Ball1.y'] <= 128:
                                return 0  # NOOP
                            else:
                                return 1  # FIRE
            else:
                if state['C(Player1, Ball1).x'] <= 131:
                    if state['V(Player1).x'] <= 14:
                        if state['ED(Player1, Ball1)'] <= 32:
                            if state['LT(Player1, Ball1).y'] <= -27:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Player1).y'] <= 1:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                    else:
                        if state['DV(Ball1).y'] <= -7:
                            if state['ED(Player1, Ball1)'] <= 46:
                                return 0  # NOOP
                            else:
                                return 1  # FIRE
                        else:
                            if state['DV(Player1).y'] <= -16:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                else:
                    if state['LT(Player1, Ball1).y'] <= 1:
                        if state['DV(Player1).y'] <= -7:
                            if state['Ball1.x'] <= 131:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['ED(Player1, Ball1)'] <= 9:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                    else:
                        if state['V(Player1).x'] <= 15:
                            if state['LT(Player1, Ball1).y'] <= 4:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['LT(Player1, Ball1).y'] <= 12:
                                return 2  # RIGHT
                            else:
                                return 0  # NOOP
        else:
            if state['Ball1.y#1'] <= 37:
                if state['Ball1.x[t-1]'] <= 7:
                    if state['Enemy1.y[t-1]'] <= 37:
                        if state['Ball1.y#1'] <= 18:
                            return 1  # FIRE
                        else:
                            return 1  # FIRE
                    else:
                        return 0  # NOOP
                else:
                    if state['DV(Player1).y'] <= -6:
                        if state['C(Player1, Ball1).y'] <= 38:
                            if state['LT(Player1, Ball1).y'] <= -109:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            return 2  # RIGHT
                    else:
                        if state['C(Player1, Ball1).y'] <= 43:
                            if state['Ball1.y#1'] <= 33:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['D(Player1, Ball1).y'] <= -18:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
            else:
                if state['DV(Player1).y'] <= 1:
                    if state['LT(Player1, Ball1).x'] <= 0:
                        if state['D(Player1, Ball1).y'] <= -5:
                            if state['C(Player1, Ball1).y'] <= 43:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['DV(Ball1).y'] <= 4:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                    else:
                        if state['V(Player1).x'] <= 3:
                            if state['D(Player1, Ball1).y'] <= -11:
                                return 2  # RIGHT
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Player1).y'] <= -5:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                else:
                    if state['D(Player1, Ball1).y'] <= -9:
                        if state['DV(Ball1).y'] <= 4:
                            if state['LT(Player1, Ball1).x'] <= 0:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            if state['Player1.y#1'] <= 116:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                    else:
                        if state['DV(Ball1).y'] <= 5:
                            if state['C(Player1, Ball1).x'] <= 136:
                                return 1  # FIRE
                            else:
                                return 0  # NOOP
                        else:
                            if state['V(Player1).x'] <= 7:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
    else:
        if state['DV(Ball1).y'] <= 0:
            if state['DV(Player1).y'] <= -6:
                if state['D(Player1, Ball1).y'] <= 6:
                    if state['ED(Player1, Ball1)'] <= 27:
                        if state['LT(Player1, Ball1).y'] <= 4:
                            if state['DV(Ball1).x'] <= 3:
                                return 2  # RIGHT
                            else:
                                return 0  # NOOP
                        else:
                            if state['ED(Player1, Ball1)'] <= 18:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                    else:
                        if state['Ball1.y#1'] <= 170:
                            if state['D(Player1, Ball1).y'] <= 4:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Ball1).y'] <= -5:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                else:
                    if state['D(Player1, Ball1).y'] <= 11:
                        if state['DV(Player1).y'] <= -13:
                            if state['V(Player1).x'] <= 19:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Ball1).y'] <= -8:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                    else:
                        if state['D(Player1, Ball1).y'] <= 12:
                            if state['DV(Ball1).y'] <= -7:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['Ball1.x'] <= 25:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
            else:
                if state['DV(Ball1).y'] <= -5:
                    if state['D(Player1, Ball1).y'] <= -2:
                        if state['DV(Player1).y'] <= -1:
                            if state['D(Player1, Ball1).y'] <= -2:
                                return 0  # NOOP
                            else:
                                return 3  # LEFT
                        else:
                            if state['DV(Ball1).y'] <= -7:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                    else:
                        if state['Ball1.y#1'] <= 37:
                            if state['Player1.y'] <= 37:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['Ball1.x[t-1]'] <= 155:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                else:
                    if state['DV(Player1).y'] <= 0:
                        if state['D(Player1, Ball1).y'] <= 3:
                            if state['Player1.y#1'] <= 35:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['C(Player1, Ball1).x'] <= 85:
                                return 1  # FIRE
                            else:
                                return 3  # LEFT
                    else:
                        if state['D(Player1, Ball1).y'] <= -2:
                            if state['DV(Player1).y'] <= 6:
                                return 1  # FIRE
                            else:
                                return 3  # LEFT
                        else:
                            if state['ED(Player1, Ball1)'] <= 3:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
        else:
            if state['D(Player1, Ball1).y'] <= 11:
                if state['DV(Player1).y'] <= 0:
                    if state['LT(Player1, Ball1).x'] <= 0:
                        if state['Ball1.x#1'] <= 123:
                            if state['DV(Player1).y'] <= -16:
                                return 2  # RIGHT
                            else:
                                return 0  # NOOP
                        else:
                            if state['V(Player1).x'] <= 15:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                    else:
                        if state['D(Player1, Ball1).y'] <= 6:
                            if state['DV(Ball1).y'] <= 0:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['DV(Ball1).y'] <= 6:
                                return 1  # FIRE
                            else:
                                return 2  # RIGHT
                else:
                    if state['Ball1.x'] <= 124:
                        if state['DV(Ball1).y'] <= 4:
                            if state['DV(Ball1).y'] <= 4:
                                return 0  # NOOP
                            else:
                                return 1  # FIRE
                        else:
                            if state['DV(Player1).y'] <= 4:
                                return 2  # RIGHT
                            else:
                                return 1  # FIRE
                    else:
                        if state['LT(Player1, Ball1).y'] <= -4:
                            if state['D(Player1, Ball1).y'] <= 1:
                                return 2  # RIGHT
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Player1).y'] <= 5:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
            else:
                if state['V(Player1).x'] <= 8:
                    if state['D(Player1, Ball1).y'] <= 18:
                        if state['Ball1.x[t-1]'] <= 117:
                            if state['V(Ball1).x'] <= 1:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Player1).y'] <= -1:
                                return 0  # NOOP
                            else:
                                return 3  # LEFT
                    else:
                        if state['D(Player1, Ball1).y'] <= 21:
                            if state['LT(Player1, Ball1).y'] <= 59:
                                return 3  # LEFT
                            else:
                                return 1  # FIRE
                        else:
                            if state['Ball1.x[t-1]'] <= 57:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                else:
                    if state['V(Ball1).x'] <= 9:
                        if state['DV(Player1).y'] <= -20:
                            if state['D(Player1, Ball1).y'] <= 32:
                                return 0  # NOOP
                            else:
                                return 3  # LEFT
                        else:
                            if state['DV(Player1).y'] <= 14:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                    else:
                        if state['DV(Player1).y'] <= 15:
                            if state['C(Player1, Ball1).x'] <= 132:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['Ball1.x#1'] <= 125:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
