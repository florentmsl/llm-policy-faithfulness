def play(state):
    if state['LT(Agent, Obj_A).y'] <= 6:
        if state['C(Obj_B, Agent).y'] <= 155:
            if state['D(Agent, Obj_A).y'] <= 7:
                if state['D(Obj_A, Obj_B).x'] <= -59:
                    if state['D(Agent, Obj_A).y'] <= -4:
                        if state['ED(Obj_B, Obj_A)'] <= 91:
                            if state['D(Obj_A, Agent).y'] <= 30:
                                return 0
                            else:
                                return 2
                        else:
                            if state['LT(Obj_B, Agent).y'] <= -28494:
                                return 2
                            else:
                                return 2
                    else:
                        if state['LT(Obj_A, Agent).y'] <= -1079:
                            if state['LT(Obj_A, Obj_B).y'] <= 3618:
                                return 0
                            else:
                                return 0
                        else:
                            if state['DV(Obj_A).y'] <= 2:
                                return 3
                            else:
                                return 2
                else:
                    if state['LT(Obj_A, Obj_B).y'] <= 2327:
                        if state['C(Obj_A, Agent).y'] <= 56:
                            if state['D(Obj_A, Obj_B).y'] <= 2:
                                return 3
                            else:
                                return 5
                        else:
                            if state['LT(Agent, Obj_B).y'] <= -2477:
                                return 4
                            else:
                                return 3
                    else:
                        if state['LT(Obj_A, Agent).y'] <= -9709:
                            if state['C(Obj_A, Agent).x'] <= 106:
                                return 5
                            else:
                                return 4
                        else:
                            if state['LT(Agent, Obj_A).y'] <= -102:
                                return 4
                            else:
                                return 0
            else:
                if state['LT(Obj_B, Obj_A).y'] <= 238:
                    if state['LT(Obj_A, Obj_B).y'] <= 3441:
                        if state['Obj_A.x'] <= 142:
                            if state['LT(Obj_A, Agent).x'] <= 0:
                                return 3
                            else:
                                return 1
                        else:
                            if state['ED(Obj_A, Agent)'] <= 11:
                                return 3
                            else:
                                return 3
                    else:
                        if state['Agent.y[t-1]'] <= 130:
                            if state['LT(Obj_A, Agent).y'] <= -12:
                                return 0
                            else:
                                return 2
                        else:
                            if state['LT(Obj_A, Agent).y'] <= -1499:
                                return 0
                            else:
                                return 1
                else:
                    if state['V(Agent).x'] <= 5:
                        if state['Obj_B.y[t-1]'] <= 140:
                            if state['LT(Agent, Obj_B).y'] <= -7430:
                                return 4
                            else:
                                return 1
                        else:
                            if state['Obj_A.x'] <= 139:
                                return 4
                            else:
                                return 4
                    else:
                        if state['ED(Obj_B, Agent)'] <= 125:
                            if state['Agent.y'] <= 120:
                                return 1
                            else:
                                return 4
                        else:
                            if state['LT(Obj_B, Agent).y'] <= -23579:
                                return 0
                            else:
                                return 1
        else:
            if state['ED(Obj_B, Obj_A)'] <= 125:
                if state['C(Agent, Obj_A).x'] <= 128:
                    if state['LT(Obj_B, Obj_A).y'] <= -72:
                        if state['LT(Obj_A, Obj_B).y'] <= -3300:
                            if state['LT(Obj_A, Agent).y'] <= -3742:
                                return 5
                            else:
                                return 4
                        else:
                            if state['LT(Agent, Obj_B).y'] <= 4932:
                                return 4
                            else:
                                return 3
                    else:
                        if state['LT(Obj_A, Obj_B).y'] <= 2464:
                            if state['LT(Agent, Obj_A).y'] <= -9:
                                return 0
                            else:
                                return 5
                        else:
                            if state['LT(Agent, Obj_A).y'] <= -38:
                                return 4
                            else:
                                return 3
                else:
                    if state['LT(Agent, Obj_A).y'] <= -32:
                        if state['ED(Obj_B, Agent)'] <= 125:
                            if state['LT(Obj_B, Obj_B).x'] <= -8:
                                return 4
                            else:
                                return 4
                        else:
                            if state['D(Obj_A, Agent).y'] <= 24:
                                return 4
                            else:
                                return 2
                    else:
                        if state['LT(Agent, Obj_A).y'] <= -5:
                            if state['DV(Agent).y'] <= 21:
                                return 4
                            else:
                                return 0
                        else:
                            if state['LT(Obj_A, Agent).y'] <= 139:
                                return 5
                            else:
                                return 4
            else:
                if state['Obj_A.y#1'] <= 183:
                    if state['Obj_B.y[t-1]'] <= 136:
                        if state['D(Agent, Obj_A).y'] <= -183:
                            if state['V(Agent).x'] <= 2:
                                return 3
                            else:
                                return 3
                        else:
                            if state['C(Obj_B, Agent).y'] <= 161:
                                return 2
                            else:
                                return 2
                    else:
                        if state['Obj_A.y'] <= 168:
                            if state['LT(Agent, Obj_B).y'] <= -8703:
                                return 3
                            else:
                                return 3
                        else:
                            if state['LT(Obj_B, Agent).y'] <= -620:
                                return 5
                            else:
                                return 5
                else:
                    if state['LT(Agent, Obj_A).y'] <= -15:
                        if state['LT(Obj_B, Agent).y'] <= 22:
                            if state['D(Obj_A, Obj_B).y'] <= -22:
                                return 5
                            else:
                                return 3
                        else:
                            if state['LT(Obj_B, Agent).y'] <= 2500:
                                return 5
                            else:
                                return 3
                    else:
                        if state['DV(Agent).y'] <= 3:
                            if state['LT(Obj_A, Obj_B).y'] <= -2711:
                                return 3
                            else:
                                return 4
                        else:
                            if state['LT(Agent, Obj_A).y'] <= 1:
                                return 5
                            else:
                                return 5
    else:
        if state['LT(Obj_B, Obj_A).y'] <= 233:
            if state['Agent.y[t-1]'] <= 138:
                if state['D(Obj_A, Agent).y'] <= -3:
                    if state['DV(Obj_A).x'] <= -1:
                        if state['ED(Obj_B, Obj_A)'] <= 103:
                            if state['D(Agent, Obj_A).y'] <= 14:
                                return 3
                            else:
                                return 3
                        else:
                            if state['LT(Agent, Obj_A).y'] <= 12:
                                return 3
                            else:
                                return 3
                    else:
                        if state['LT(Obj_B, Agent).x'] <= 0:
                            if state['D(Agent, Obj_A).y'] <= 30:
                                return 0
                            else:
                                return 5
                        else:
                            if state['ED(Agent, Obj_A)'] <= 24:
                                return 3
                            else:
                                return 3
                else:
                    if state['LT(Obj_B, Agent).y'] <= 3736:
                        if state['LT(Obj_B, Obj_A).y'] <= 81:
                            if state['ED(Agent, Obj_A)'] <= 29:
                                return 3
                            else:
                                return 0
                        else:
                            if state['C(Obj_A, Obj_B).y'] <= 48:
                                return 2
                            else:
                                return 3
                    else:
                        if state['ED(Agent, Obj_B)'] <= 126:
                            if state['LT(Obj_A, Obj_B).y'] <= -4838:
                                return 0
                            else:
                                return 2
                        else:
                            if state['D(Obj_A, Obj_B).x'] <= -64:
                                return 2
                            else:
                                return 5
            else:
                if state['ED(Agent, Obj_A)'] <= 40:
                    if state['Agent.y[t-1]'] <= 160:
                        if state['ED(Obj_B, Obj_A)'] <= 115:
                            if state['LT(Agent, Obj_A).y'] <= 18:
                                return 4
                            else:
                                return 5
                        else:
                            if state['Obj_B.y'] <= 164:
                                return 3
                            else:
                                return 5
                    else:
                        if state['ED(Obj_A, Agent)'] <= 30:
                            if state['Obj_B.y[t-1]'] <= 142:
                                return 3
                            else:
                                return 5
                        else:
                            if state['C(Agent, Obj_A).y'] <= 186:
                                return 5
                            else:
                                return 5
                else:
                    if state['LT(Agent, Obj_B).y'] <= -21193:
                        if state['LT(Obj_A, Obj_B).y'] <= -28470:
                            if state['LT(Agent, Obj_A).y'] <= 28:
                                return 3
                            else:
                                return 5
                        else:
                            if state['ED(Obj_A, Obj_B)'] <= 62:
                                return 0
                            else:
                                return 0
                    else:
                        if state['D(Obj_B, Obj_A).y'] <= 13:
                            if state['LT(Obj_A, Obj_B).y'] <= -5288:
                                return 2
                            else:
                                return 3
                        else:
                            if state['LT(Obj_A, Agent).x'] <= 0:
                                return 5
                            else:
                                return 5
        else:
            if state['C(Agent, Obj_A).y'] <= 122:
                if state['Agent.y#1'] <= 109:
                    if state['LT(Obj_B, Agent).y'] <= -11791:
                        return 3
                    else:
                        if state['ED(Obj_B, Agent)'] <= 125:
                            if state['LT(Obj_A, Agent).y'] <= 33:
                                return 2
                            else:
                                return 3
                        else:
                            if state['D(Obj_A, Agent).y'] <= -17:
                                return 3
                            else:
                                return 3
                else:
                    if state['D(Obj_B, Obj_A).x'] <= 131:
                        if state['DV(Obj_A).y'] <= 127:
                            if state['LT(Agent, Obj_B).y'] <= 9945:
                                return 4
                            else:
                                return 3
                        else:
                            if state['LT(Agent, Obj_B).y'] <= 9900:
                                return 4
                            else:
                                return 4
                    else:
                        if state['D(Agent, Obj_A).y'] <= 6:
                            if state['LT(Obj_A, Agent).x'] <= 0:
                                return 2
                            else:
                                return 2
                        else:
                            if state['LT(Obj_B, Agent).y'] <= 3703:
                                return 1
                            else:
                                return 2
            else:
                if state['LT(Obj_B, Agent).y'] <= -24829:
                    if state['Agent.y'] <= 123:
                        if state['C(Obj_A, Agent).y'] <= 137:
                            if state['LT(Obj_A, Obj_B).y'] <= 9483:
                                return 3
                            else:
                                return 3
                        else:
                            return 1
                    else:
                        if state['Obj_A.y[t-1]'] <= 157:
                            if state['C(Obj_A, Agent).y'] <= 139:
                                return 0
                            else:
                                return 3
                        else:
                            if state['LT(Agent, Obj_B).y'] <= 9948:
                                return 1
                            else:
                                return 1
                else:
                    if state['ED(Obj_A, Agent)'] <= 13:
                        if state['DV(Agent).y'] <= -6:
                            if state['Obj_A.y'] <= 137:
                                return 2
                            else:
                                return 4
                        else:
                            if state['D(Obj_A, Agent).y'] <= -10:
                                return 4
                            else:
                                return 4
                    else:
                        if state['Agent.y#1'] <= 128:
                            if state['LT(Agent, Obj_A).y'] <= 20:
                                return 1
                            else:
                                return 3
                        else:
                            if state['LT(Obj_B, Agent).y'] <= -11167:
                                return 1
                            else:
                                return 4
