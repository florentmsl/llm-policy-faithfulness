def play(state):
    if state['D(Monkey2, FallingCoconut1).x'] <= 9:
        if state['D(Player1, FallingCoconut1).y'] <= -87:
            if state['D(Player1, Ladder1).y'] <= -12:
                if state['DV(Monkey2).x'] <= -49:
                    return 6  # UPRIGHT
                else:
                    if state['DV(Monkey1).x'] <= -53:
                        if state['D(Player1, Fruit1).y'] <= -100:
                            return 1  # FIRE
                        else:
                            return 4  # LEFT
                    else:
                        if state['D(Player1, Monkey1).x'] <= -112:
                            if state['Monkey1.y[t-1]'] <= 79:
                                return 1  # FIRE
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['D(Player1, FallingCoconut1).y'] <= -103:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
            else:
                if state['Monkey2.y[t-1]'] <= 9:
                    if state['Child1.x[t-1]'] <= 127:
                        if state['DV(Player1).y'] <= -4:
                            return 1  # FIRE
                        else:
                            if state['D(Player1, Child1).x'] <= 32:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                    else:
                        if state['DV(Player1).x'] <= -1:
                            if state['D(Player1, Fruit2).y'] <= -52:
                                return 2  # UP
                            else:
                                return 4  # LEFT
                        else:
                            if state['D(Player1, Platform2).y'] <= -108:
                                return 6  # UPRIGHT
                            else:
                                return 4  # LEFT
                else:
                    if state['DV(Player1).y'] <= 4:
                        if state['D(Player1, Monkey2).y'] <= 21:
                            if state['DV(FallingCoconut1).y'] <= -9:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                        else:
                            if state['FallingCoconut1.y'] <= 21:
                                return 7  # UPLEFT
                            else:
                                return 5  # DOWN
                    else:
                        if state['D(Monkey2, ThrownCoconut1).y'] <= -65:
                            if state['D(Player1, FallingCoconut1).y'] <= -119:
                                return 1  # FIRE
                            else:
                                return 7  # UPLEFT
                        else:
                            return 4  # LEFT
        else:
            if state['D(Monkey2, FallingCoconut1).y'] <= 32:
                if state['DV(Child1).x'] <= -2:
                    if state['D(Monkey2, FallingCoconut1).y'] <= 24:
                        return 6  # UPRIGHT
                    else:
                        return 3  # RIGHT
                else:
                    if state['D(Player1, Bell1).y'] <= -108:
                        if state['DV(Child1).x'] <= 2:
                            if state['D(Player1, FallingCoconut1).x'] <= 1:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['D(Player1, Platform2).x'] <= -84:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                    else:
                        if state['DV(Player1).y'] <= -4:
                            if state['D(Player1, FallingCoconut1).x'] <= -2:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['D(Monkey2, Ladder1).y'] <= 83:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
            else:
                if state['D(Monkey1, FallingCoconut1).x'] <= -33:
                    if state['Player1.x[t-1]'] <= 100:
                        if state['FallingCoconut1.x#1'] <= 94:
                            if state['DV(Player1).y'] <= -4:
                                return 6  # UPRIGHT
                            else:
                                return 3  # RIGHT
                        else:
                            if state['DV(FallingCoconut1).y'] <= -4:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['FallingCoconut1.x'] <= 94:
                            if state['DV(Player1).y'] <= -4:
                                return 3  # RIGHT
                            else:
                                return 5  # DOWN
                        else:
                            if state['D(Player1, FallingCoconut1).y'] <= -15:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                else:
                    if state['Player1.x'] <= 110:
                        if state['D(Player1, Monkey2).y'] <= -59:
                            return 1  # FIRE
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -14:
                                return 3  # RIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['DV(FallingCoconut1).y'] <= -4:
                            if state['D(Player1, FallingCoconut1).y'] <= 9:
                                return 6  # UPRIGHT
                            else:
                                return 3  # RIGHT
                        else:
                            if state['DV(Player1).x'] <= -1:
                                return 5  # DOWN
                            else:
                                return 5  # DOWN
    else:
        if state['D(Monkey1, ThrownCoconut1).x'] <= -76:
            if state['D(Player1, Monkey1).y'] <= -59:
                if state['D(Player1, Ladder2).y'] <= -51:
                    if state['D(Player1, FallingCoconut1).x'] <= -19:
                        if state['D(Player1, Monkey1).x'] <= 76:
                            return 4  # LEFT
                        else:
                            return 6  # UPRIGHT
                    else:
                        if state['DV(Player1).y'] <= 4:
                            if state['D(Player1, FallingCoconut1).x'] <= -18:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['Child1.x[t-1]'] <= 127:
                                return 6  # UPRIGHT
                            else:
                                return 5  # DOWN
                else:
                    if state['DV(Child1).x'] <= -2:
                        if state['D(Player1, FallingCoconut1).x'] <= -10:
                            return 6  # UPRIGHT
                        else:
                            if state['DV(Player1).x'] <= -1:
                                return 6  # UPRIGHT
                            else:
                                return 1  # FIRE
                    else:
                        if state['D(Player1, Monkey2).x'] <= -60:
                            if state['DV(FallingCoconut1).y'] <= 4:
                                return 6  # UPRIGHT
                            else:
                                return 4  # LEFT
                        else:
                            return 6  # UPRIGHT
            else:
                if state['Child1.x'] <= 115:
                    if state['Player1.y[t-1]'] <= 144:
                        if state['Monkey1.y[t-1]'] <= 89:
                            return 1  # FIRE
                        else:
                            if state['Player1.y'] <= 136:
                                return 1  # FIRE
                            else:
                                return 3  # RIGHT
                    else:
                        if state['D(Player1, Fruit2).y'] <= -60:
                            return 6  # UPRIGHT
                        else:
                            return 4  # LEFT
                else:
                    if state['D(Player1, ThrownCoconut1).y'] <= -136:
                        if state['DV(Player1).y'] <= 4:
                            if state['DV(Player1).y'] <= -4:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['Monkey1.y#1'] <= 93:
                                return 4  # LEFT
                            else:
                                return 1  # FIRE
                    else:
                        if state['DV(Player1).x'] <= -1:
                            if state['Child1.x[t-1]'] <= 115:
                                return 2  # UP
                            else:
                                return 4  # LEFT
                        else:
                            if state['Player1.y[t-1]'] <= 136:
                                return 1  # FIRE
                            else:
                                return 4  # LEFT
        else:
            return 3  # RIGHT
