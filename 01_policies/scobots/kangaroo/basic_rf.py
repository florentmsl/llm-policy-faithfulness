def play(state):
    if state['D(Monkey2, FallingCoconut1).y'] <= 15:
        if state['Monkey3.y#1'] <= 41:
            if state['D(Player1, ThrownCoconut1).x'] <= -28:
                if state['DV(Player1).y'] <= -4:
                    if state['D(Monkey2, Child1).x'] <= -25:
                        if state['D(Player1, FallingCoconut1).x'] <= -19:
                            if state['D(Player1, Ladder2).y'] <= -59:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Monkey2, Child1).x'] <= -29:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= -83:
                            if state['D(Player1, Ladder2).x'] <= -71:
                                return 7  # UPLEFT
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['D(Player1, Child1).x'] <= 59:
                                return 2  # UP
                            else:
                                return 2  # UP
                else:
                    if state['Monkey3.y[t-1]'] <= 2:
                        if state['D(Player1, Monkey2).x'] <= 76:
                            if state['Player1.x#1'] <= 29:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Monkey2, FallingCoconut1).y'] <= -21:
                                return 2  # UP
                            else:
                                return 2  # UP
                    else:
                        if state['D(Player1, Fruit2).y'] <= -52:
                            if state['Player1.y[t-1]'] <= 144:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -49:
                                return 7  # UPLEFT
                            else:
                                return 7  # UPLEFT
            else:
                if state['D(Player1, FallingCoconut1).x'] <= -48:
                    if state['DV(Player1).y'] <= -4:
                        if state['Player1.y'] <= 144:
                            if state['D(Player1, Platform2).x'] <= -71:
                                return 6  # UPRIGHT
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['DV(Player1).x'] <= -2:
                                return 7  # UPLEFT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= -49:
                            if state['D(Player1, Monkey1).y'] <= 21:
                                return 6  # UPRIGHT
                            else:
                                return 5  # DOWN
                        else:
                            if state['FallingCoconut1.x[t-1]'] <= 38:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['D(Monkey1, FallingCoconut1).y'] <= -64:
                        if state['Player1.x[t-1]'] <= 84:
                            if state['ThrownCoconut1.x[t-1]'] <= 51:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['FallingCoconut1.x#1'] <= 84:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Player1, Fruit2).x'] <= -55:
                            if state['D(Monkey2, FallingCoconut1).x'] <= -73:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['FallingCoconut1.x[t-1]'] <= 87:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
        else:
            if state['D(Monkey1, ThrownCoconut1).y'] <= -89:
                if state['FallingCoconut1.x[t-1]'] <= 74:
                    if state['D(Player1, Monkey1).x'] <= 43:
                        if state['D(Player1, Child1).y'] <= -124:
                            if state['FallingCoconut1.x[t-1]'] <= 68:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -14:
                                return 7  # UPLEFT
                            else:
                                return 7  # UPLEFT
                    else:
                        if state['DV(Player1).x'] <= -1:
                            if state['FallingCoconut1.x#1'] <= 70:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, Child1).x'] <= 36:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['D(Player1, Monkey1).y'] <= -59:
                        if state['D(Player1, Monkey2).x'] <= 40:
                            if state['D(Player1, Ladder2).x'] <= -53:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['Child1.x[t-1]'] <= 119:
                                return 7  # UPLEFT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Player1, FallingCoconut1).y'] <= -119:
                            if state['Player1.y[t-1]'] <= 144:
                                return 7  # UPLEFT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Monkey1, FallingCoconut1).x'] <= -23:
                                return 2  # UP
                            else:
                                return 2  # UP
            else:
                if state['D(Monkey1, Child1).x'] <= 21:
                    if state['D(Monkey1, FallingCoconut1).x'] <= -23:
                        if state['D(Player1, Ladder2).y'] <= -51:
                            if state['Player1.x[t-1]'] <= 88:
                                return 7  # UPLEFT
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['D(Player1, Child1).x'] <= 46:
                                return 2  # UP
                            else:
                                return 2  # UP
                    else:
                        if state['D(Player1, Child1).y'] <= -156:
                            if state['DV(Child1).x'] <= -2:
                                return 2  # UP
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['D(Monkey1, ThrownCoconut1).x'] <= -10:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['D(Player1, ThrownCoconut1).y'] <= -19:
                        if state['D(Player1, FallingCoconut1).x'] <= -14:
                            if state['FallingCoconut1.x#1'] <= 82:
                                return 2  # UP
                            else:
                                return 2  # UP
                        else:
                            if state['DV(Child1).x'] <= -2:
                                return 2  # UP
                            else:
                                return 7  # UPLEFT
                    else:
                        if state['D(Monkey2, FallingCoconut1).x'] <= -59:
                            if state['D(Player1, Monkey2).x'] <= 45:
                                return 2  # UP
                            else:
                                return 7  # UPLEFT
                        else:
                            if state['D(Player1, FallingCoconut1).y'] <= -95:
                                return 7  # UPLEFT
                            else:
                                return 2  # UP
    else:
        if state['D(Monkey1, FallingCoconut1).y'] <= -8:
            if state['D(Monkey2, Child1).x'] <= 119:
                if state['DV(Player1).y'] <= -4:
                    if state['D(Player1, Child1).x'] <= 62:
                        if state['FallingCoconut1.x#1'] <= 58:
                            if state['DV(FallingCoconut1).y'] <= -4:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['DV(FallingCoconut1).y'] <= -4:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['DV(FallingCoconut1).x'] <= -1:
                            if state['D(Player1, Monkey2).x'] <= -46:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, Monkey1).y'] <= -59:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['DV(Player1).x'] <= -2:
                        if state['D(Player1, Bell1).x'] <= 31:
                            if state['FallingCoconut1.x[t-1]'] <= 58:
                                return 2  # UP
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -16:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Monkey1, FallingCoconut1).x'] <= -97:
                            if state['Player1.x'] <= 67:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['D(Monkey1, FallingCoconut1).x'] <= -94:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
            else:
                if state['Child1.x[t-1]'] <= 119:
                    if state['D(Player1, FallingCoconut1).x'] <= -9:
                        return 6  # UPRIGHT
                    else:
                        if state['D(Player1, Fruit2).y'] <= -56:
                            if state['DV(Player1).x'] <= -1:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, Platform2).x'] <= -40:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['Player1.x[t-1]'] <= 56:
                        if state['D(Player1, Ladder1).y'] <= -12:
                            if state['DV(Player1).y'] <= -4:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
                        else:
                            if state['DV(Player1).y'] <= -4:
                                return 2  # UP
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Monkey2, FallingCoconut1).y'] <= 93:
                            if state['Player1.x#1'] <= 56:
                                return 2  # UP
                            else:
                                return 2  # UP
                        else:
                            if state['DV(Player1).y'] <= -4:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
        else:
            if state['D(Player1, Ladder2).y'] <= -59:
                if state['D(Player1, Fruit1).x'] <= 20:
                    if state['D(Player1, FallingCoconut1).x'] <= -35:
                        if state['Player1.x[t-1]'] <= 55:
                            if state['D(Monkey2, FallingCoconut1).y'] <= 157:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Monkey2, FallingCoconut1).y'] <= 157:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Monkey2, FallingCoconut1).x'] <= -94:
                            if state['Monkey1.y'] <= 141:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, Child1).x'] <= 69:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['Player1.y[t-1]'] <= 144:
                        if state['DV(FallingCoconut1).x'] <= -1:
                            if state['D(Player1, Child1).x'] <= 88:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['FallingCoconut1.y'] <= 29:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= -1:
                            if state['D(Player1, FallingCoconut1).x'] <= -15:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, FallingCoconut1).x'] <= -1:
                                return 6  # UPRIGHT
                            else:
                                return 2  # UP
            else:
                if state['D(Player1, Platform1).y'] <= 36:
                    if state['D(Player1, Monkey1).x'] <= 117:
                        if state['DV(Monkey1).x'] <= -51:
                            if state['D(Player1, Platform2).x'] <= -21:
                                return 2  # UP
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, Monkey1).x'] <= 102:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                    else:
                        if state['DV(Player1).y'] <= 4:
                            if state['DV(Monkey1).x'] <= -51:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['Child1.x#1'] <= 119:
                                return 3  # RIGHT
                            else:
                                return 6  # UPRIGHT
                else:
                    if state['Child1.x[t-1]'] <= 127:
                        if state['FallingCoconut1.y[t-1]'] <= 21:
                            if state['D(Player1, Monkey1).x'] <= 130:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['D(Player1, Child1).x'] <= 70:
                                return 6  # UPRIGHT
                            else:
                                return 3  # RIGHT
                    else:
                        if state['D(Player1, FallingCoconut1).x'] <= -10:
                            if state['D(Player1, Monkey1).x'] <= 98:
                                return 2  # UP
                            else:
                                return 2  # UP
                        else:
                            if state['D(Player1, Fruit1).x'] <= 30:
                                return 6  # UPRIGHT
                            else:
                                return 3  # RIGHT
