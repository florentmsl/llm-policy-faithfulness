def play(state):
    if state['D(Player1, FallingCoconut1).x'] <= -2:
        if state['FallingCoconut1.y#1'] <= 117:
            if state['D(Monkey2, ThrownCoconut1).x'] <= -139:
                if state['Monkey2.x[t-1]'] <= 139:
                    if state['DV(Player1).x'] <= 0:
                        if state['Player1.x[t-1]'] <= 35:
                            if state['D(Player1, Monkey1).y'] <= -3:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['Player1.x[t-1]'] <= 46:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                    else:
                        if state['D(Player1, ThrownCoconut1).x'] <= -44:
                            if state['Player1.x[t-1]'] <= 46:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['Player1.y[t-1]'] <= 144:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                else:
                    if state['D(Player1, Monkey1).y'] <= -11:
                        if state['D(Monkey2, FallingCoconut1).x'] <= -122:
                            if state['D(Player1, Monkey2).x'] <= 132:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, Ladder1).x'] <= 25:
                                return 3  # RIGHT
                            else:
                                return 5  # DOWN
                    else:
                        if state['D(Monkey1, FallingCoconut1).x'] <= -95:
                            if state['D(Player1, ThrownCoconut1).y'] <= -136:
                                return 4  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['Player1.x'] <= 67:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
            else:
                if state['D(Player1, Monkey1).x'] <= 85:
                    if state['Monkey1.y[t-1]'] <= 45:
                        if state['D(Player1, FallingCoconut1).y'] <= -140:
                            if state['D(Player1, Monkey2).x'] <= -21:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['Child1.x[t-1]'] <= 127:
                                return 0  # NOOP
                            else:
                                return 4  # LEFT
                    else:
                        if state['D(Monkey2, ThrownCoconut1).x'] <= -50:
                            if state['D(Player1, Monkey2).x'] <= 122:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -7:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                else:
                    if state['FallingCoconut1.y[t-1]'] <= 77:
                        if state['D(Player1, ThrownCoconut1).x'] <= 82:
                            if state['D(Player1, Monkey2).y'] <= -144:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, Child1).x'] <= 93:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['D(Player1, Monkey2).x'] <= -27:
                            if state['D(Player1, Child1).y'] <= -132:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['DV(Player1).x'] <= 1:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
        else:
            if state['DV(Player1).x'] <= 1:
                if state['Monkey1.y[t-1]'] <= 57:
                    if state['D(Player1, Monkey2).x'] <= -22:
                        if state['D(Player1, FallingCoconut1).x'] <= -5:
                            if state['D(Player1, FallingCoconut1).x'] <= -8:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, Monkey1).y'] <= -99:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
                    else:
                        if state['D(Player1, Monkey2).x'] <= -21:
                            if state['D(Player1, Monkey1).y'] <= -99:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
                        else:
                            if state['Player1.y'] <= 152:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                else:
                    if state['D(Player1, Monkey2).x'] <= 121:
                        if state['D(Player1, Fruit2).y'] <= -68:
                            if state['D(Player1, FallingCoconut1).y'] <= -23:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, Child1).x'] <= 109:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                    else:
                        if state['D(Player1, Ladder2).x'] <= 0:
                            if state['ThrownCoconut1.x[t-1]'] <= 78:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['Player1.x[t-1]'] <= 19:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
            else:
                if state['D(Player1, FallingCoconut1).x'] <= -5:
                    if state['Player1.y[t-1]'] <= 152:
                        if state['D(Player1, FallingCoconut1).x'] <= -8:
                            if state['D(Player1, Child1).x'] <= 64:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                        else:
                            if state['DV(Player1).x'] <= 2:
                                return 4  # LEFT
                            else:
                                return 4  # LEFT
                    else:
                        if state['D(Player1, Platform1).x'] <= -8:
                            return 1  # FIRE
                        else:
                            return 1  # FIRE
                else:
                    if state['DV(Player1).x'] <= 2:
                        if state['D(Player1, FallingCoconut1).x'] <= -3:
                            if state['FallingCoconut1.y[t-1]'] <= 125:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, FallingCoconut1).y'] <= -15:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                    else:
                        if state['FallingCoconut1.y#1'] <= 133:
                            if state['D(Player1, Monkey1).x'] <= 132:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['D(Monkey1, ThrownCoconut1).y'] <= -101:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
    else:
        if state['D(Monkey2, Platform1).y'] <= 99:
            if state['D(Monkey2, Child1).x'] <= -32:
                if state['DV(Player1).x'] <= 0:
                    if state['D(Player1, ThrownCoconut1).x'] <= 80:
                        if state['D(Player1, FallingCoconut1).x'] <= 9:
                            if state['D(Player1, FallingCoconut1).x'] <= 5:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['DV(Player1).x'] <= -1:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                    else:
                        if state['Child1.x[t-1]'] <= 115:
                            if state['DV(Player1).x'] <= -1:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, Child1).x'] <= 95:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                else:
                    if state['D(Player1, FallingCoconut1).x'] <= 8:
                        if state['D(Player1, ThrownCoconut1).x'] <= -24:
                            if state['D(Player1, FallingCoconut1).x'] <= 6:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['Child1.x[t-1]'] <= 119:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['Child1.x[t-1]'] <= 119:
                            if state['D(Player1, FallingCoconut1).x'] <= 9:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= 9:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
            else:
                if state['D(Player1, FallingCoconut1).x'] <= 8:
                    if state['D(Player1, FallingCoconut1).y'] <= -39:
                        if state['D(Monkey2, Ladder2).y'] <= -52:
                            if state['DV(Player1).x'] <= 0:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, ThrownCoconut1).x'] <= -28:
                                return 4  # LEFT
                            else:
                                return 0  # NOOP
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= -2:
                            if state['DV(Player1).x'] <= 1:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                        else:
                            if state['DV(Player1).x'] <= 0:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                else:
                    if state['DV(Player1).x'] <= 0:
                        if state['D(Player1, FallingCoconut1).x'] <= 11:
                            if state['FallingCoconut1.x[t-1]'] <= 72:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= 12:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['Monkey1.x[t-1]'] <= 135:
                            if state['D(Player1, FallingCoconut1).x'] <= 9:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= 11:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
        else:
            if state['Monkey1.y[t-1]'] <= 57:
                if state['Child1.x[t-1]'] <= 123:
                    if state['D(Player1, Ladder2).y'] <= -59:
                        if state['DV(Player1).x'] <= 0:
                            if state['Child1.x[t-1]'] <= 115:
                                return 6  # UPRIGHT
                            else:
                                return 1  # FIRE
                        else:
                            if state['Monkey1.y[t-1]'] <= 17:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['DV(Child1).x'] <= -2:
                            if state['D(Player1, FallingCoconut1).y'] <= -55:
                                return 6  # UPRIGHT
                            else:
                                return 5  # DOWN
                        else:
                            if state['D(Player1, Platform1).y'] <= 36:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                else:
                    if state['D(Player1, Monkey1).x'] <= 128:
                        if state['FallingCoconut1.x[t-1]'] <= 24:
                            if state['DV(Player1).x'] <= 1:
                                return 5  # DOWN
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, Child1).x'] <= 103:
                                return 3  # RIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['FallingCoconut1.y'] <= 29:
                            if state['Monkey1.y[t-1]'] <= 2:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= 0:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
            else:
                if state['D(Monkey2, FallingCoconut1).y'] <= 27:
                    if state['D(Player1, FallingCoconut1).x'] <= 8:
                        if state['Monkey1.x[t-1]'] <= 144:
                            if state['D(Player1, Bell1).x'] <= 27:
                                return 3  # RIGHT
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, Ladder2).y'] <= -59:
                                return 4  # LEFT
                            else:
                                return 7  # UPLEFT
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= 12:
                            if state['D(Player1, FallingCoconut1).x'] <= 9:
                                return 5  # DOWN
                            else:
                                return 5  # DOWN
                        else:
                            if state['DV(Player1).x'] <= 0:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                else:
                    if state['D(Player1, FallingCoconut1).x'] <= 0:
                        if state['DV(Player1).x'] <= 1:
                            if state['FallingCoconut1.y#1'] <= 133:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Monkey1, Monkey2).y'] <= -65:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                    else:
                        if state['D(Monkey1, Monkey2).y'] <= -95:
                            if state['D(Player1, Bell1).x'] <= 73:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= 11:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
