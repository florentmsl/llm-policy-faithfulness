def play(state):
    if state['LT(Chicken1, Car1).x'] <= 0:
        if state['LT(Chicken1, Car2).x'] <= 0:
            if state['LT(Car8, Chicken1).y'] <= -3855:
                if state['D(Car1, Car4).x'] <= 55:
                    if state['ED(Car1, Chicken1)'] <= 79:
                        if state['C(Car6, Car2).x'] <= 91:
                            if state['ED(Car10, Chicken1)'] <= 131:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car2, Chicken1)'] <= 18:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Car8, Car9)'] <= 28:
                            if state['C(Car7, Chicken1).y'] <= 118:
                                return 0  # NOOP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car7, Chicken1)'] <= 65:
                                return 1  # UP
                            else:
                                return 1  # UP
                else:
                    if state['ED(Car2, Chicken1)'] <= 61:
                        if state['ED(Chicken2, Car9)'] <= 32:
                            if state['ED(Car1, Chicken1)'] <= 59:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car3, Chicken1)'] <= 36:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Car6, Chicken1)'] <= 68:
                            return 2  # DOWN
                        else:
                            if state['ED(Chicken1, Car4)'] <= 51:
                                return 1  # UP
                            else:
                                return 1  # UP
            else:
                if state['LT(Car9, Chicken1).x'] <= 0:
                    if state['D(Chicken1, Car1).y'] <= -71:
                        if state['D(Car2, Car3).x'] <= 112:
                            if state['LT(Car5, Chicken1).y'] <= -1867:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['ED(Car1, Car6)'] <= 91:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                    else:
                        if state['LT(Car10, Chicken1).y'] <= -4058:
                            if state['D(Car7, Car8).x'] <= 22:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['LT(Car6, Chicken1).y'] <= -3203:
                                return 1  # UP
                            else:
                                return 1  # UP
                else:
                    if state['ED(Chicken2, Car9)'] <= 32:
                        if state['ED(Car10, Chicken1)'] <= 125:
                            if state['ED(Chicken2, Car10)'] <= 47:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['ED(Car1, Car3)'] <= 43:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                    else:
                        if state['ED(Car5, Chicken1)'] <= 37:
                            if state['ED(Car5, Car2)'] <= 64:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['C(Car6, Car3).x'] <= 125:
                                return 1  # UP
                            else:
                                return 1  # UP
        else:
            if state['D(Car4, Car2).x'] <= -57:
                if state['ED(Car5, Chicken2)'] <= 103:
                    if state['ED(Car2, Chicken1)'] <= 110:
                        if state['ED(Chicken1, Car7)'] <= 28:
                            if state['D(Car10, Car3).x'] <= 39:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                        else:
                            if state['C(Car1, Car8).x'] <= 82:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Car1, Car8)'] <= 123:
                            if state['ED(Chicken1, Car10)'] <= 37:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car9, Car5)'] <= 66:
                                return 1  # UP
                            else:
                                return 1  # UP
                else:
                    if state['ED(Car7, Chicken1)'] <= 22:
                        if state['ED(Chicken1, Car3)'] <= 114:
                            if state['ED(Car8, Chicken1)'] <= 89:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                        else:
                            if state['ED(Car1, Car4)'] <= 48:
                                return 1  # UP
                            else:
                                return 0  # NOOP
                    else:
                        if state['ED(Car8, Car3)'] <= 93:
                            if state['ED(Car7, Chicken1)'] <= 29:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                        else:
                            if state['ED(Car9, Chicken1)'] <= 40:
                                return 1  # UP
                            else:
                                return 1  # UP
            else:
                if state['D(Car3, Car9).x'] <= 112:
                    if state['C(Car9, Car5).x'] <= 132:
                        if state['ED(Chicken1, Car9)'] <= 8:
                            if state['D(Car1, Car10).x'] <= -157:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['D(Car9, Car8).x'] <= -114:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Chicken1, Car8)'] <= 65:
                            if state['ED(Car1, Chicken1)'] <= 131:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car3, Chicken1)'] <= 91:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                else:
                    if state['D(Car10, Car1).x'] <= 48:
                        if state['LT(Car10, Chicken1).y'] <= 1562:
                            if state['D(Car1, Car4).x'] <= 4:
                                return 0  # NOOP
                            else:
                                return 2  # DOWN
                        else:
                            if state['Car9.x#1'] <= 140:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                    else:
                        if state['ED(Car9, Car1)'] <= 128:
                            if state['ED(Car4, Chicken1)'] <= 65:
                                return 2  # DOWN
                            else:
                                return 0  # NOOP
                        else:
                            if state['ED(Chicken1, Car5)'] <= 46:
                                return 2  # DOWN
                            else:
                                return 0  # NOOP
    else:
        if state['LT(Chicken2, Chicken1).y'] <= 636:
            if state['ED(Car6, Car2)'] <= 104:
                if state['D(Car1, Car6).x'] <= 93:
                    if state['D(Car6, Car8).x'] <= -51:
                        if state['ED(Car1, Car7)'] <= 120:
                            if state['ED(Chicken1, Car1)'] <= 171:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car1, Car4)'] <= 53:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Car8, Car9)'] <= 115:
                            if state['C(Car8, Car4).x'] <= 96:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['ED(Car7, Chicken1)'] <= 64:
                                return 0  # NOOP
                            else:
                                return 1  # UP
                else:
                    if state['ED(Chicken2, Car8)'] <= 55:
                        if state['D(Car8, Car5).x'] <= -67:
                            if state['Car10.x'] <= 156:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['LT(Car5, Chicken1).x'] <= 0:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                    else:
                        if state['D(Car7, Car3).x'] <= -39:
                            if state['C(Car10, Car8).x'] <= 153:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car2, Chicken1)'] <= 151:
                                return 2  # DOWN
                            else:
                                return 1  # UP
            else:
                if state['ED(Car9, Chicken1)'] <= 28:
                    if state['Car10.x#1'] <= -1:
                        if state['C(Car5, Car3).x'] <= 114:
                            return 0  # NOOP
                        else:
                            if state['LT(Car9, Chicken1).y'] <= 15:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Chicken1, Car1)'] <= 142:
                            if state['LT(Car8, Chicken1).y'] <= -295:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car1, Chicken1)'] <= 152:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
                else:
                    if state['C(Car10, Car6).x'] <= 109:
                        if state['ED(Car4, Car10)'] <= 100:
                            if state['C(Car5, Car9).x'] <= 59:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['Car1.x'] <= 28:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Car8, Car7)'] <= 68:
                            if state['D(Car1, Car4).x'] <= 36:
                                return 1  # UP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Car7, Car10)'] <= 87:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
        else:
            if state['ED(Car9, Car10)'] <= 127:
                if state['C(Car6, Car4).x'] <= 47:
                    if state['D(Car3, Car9).x'] <= 5:
                        if state['LT(Car9, Chicken1).y'] <= 26:
                            if state['C(Car1, Chicken1).x'] <= 100:
                                return 2  # DOWN
                            else:
                                return 1  # UP
                        else:
                            if state['LT(Chicken1, Car9).y'] <= -21:
                                return 1  # UP
                            else:
                                return 0  # NOOP
                    else:
                        if state['ED(Chicken1, Car9)'] <= 19:
                            if state['LT(Car7, Chicken1).y'] <= 2168:
                                return 0  # NOOP
                            else:
                                return 1  # UP
                        else:
                            if state['LT(Car2, Chicken1).y'] <= 2716:
                                return 1  # UP
                            else:
                                return 1  # UP
                else:
                    if state['D(Car8, Car7).x'] <= 96:
                        if state['D(Car9, Car3).x'] <= -106:
                            if state['ED(Car10, Car1)'] <= 160:
                                return 0  # NOOP
                            else:
                                return 1  # UP
                        else:
                            if state['C(Car8, Car9).x'] <= 72:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['ED(Chicken1, Car9)'] <= 21:
                            return 1  # UP
                        else:
                            if state['C(Car6, Car10).x'] <= 71:
                                return 2  # DOWN
                            else:
                                return 2  # DOWN
            else:
                if state['ED(Car3, Chicken1)'] <= 153:
                    if state['ED(Chicken1, Car8)'] <= 111:
                        if state['ED(Car6, Car9)'] <= 50:
                            if state['ED(Car5, Chicken1)'] <= 102:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['LT(Chicken2, Chicken1).y'] <= 2552:
                                return 0  # NOOP
                            else:
                                return 2  # DOWN
                    else:
                        if state['ED(Chicken1, Car4)'] <= 109:
                            if state['LT(Car5, Chicken1).y'] <= -739:
                                return 0  # NOOP
                            else:
                                return 2  # DOWN
                        else:
                            if state['ED(Chicken1, Car8)'] <= 114:
                                return 0  # NOOP
                            else:
                                return 2  # DOWN
                else:
                    if state['ED(Car10, Car5)'] <= 81:
                        if state['ED(Chicken1, Car1)'] <= 157:
                            if state['ED(Chicken1, Car3)'] <= 155:
                                return 2  # DOWN
                            else:
                                return 0  # NOOP
                        else:
                            if state['ED(Car10, Chicken1)'] <= 89:
                                return 1  # UP
                            else:
                                return 1  # UP
                    else:
                        if state['LT(Car6, Chicken1).y'] <= -49:
                            if state['ED(Chicken1, Car1)'] <= 149:
                                return 1  # UP
                            else:
                                return 1  # UP
                        else:
                            if state['LT(Car9, Chicken1).y'] <= -1373:
                                return 1  # UP
                            else:
                                return 1  # UP
