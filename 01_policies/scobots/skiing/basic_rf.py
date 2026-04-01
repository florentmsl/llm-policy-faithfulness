def play(state):
    if state['DV(Player1).x'] <= 0:
        if state['Player1.x#1'] <= 76:
            if state['Mogul3.y#1'] <= 145:
                if state['Tree4.y'] <= 33:
                    if state['Flag4.y[t-1]'] <= 53:
                        if state['O(Player1)'] <= 10:
                            return 2  # LEFT
                        else:
                            return 0  # NOOP
                    else:
                        return 0  # NOOP
                else:
                    if state['Tree1.y#1'] <= 159:
                        return 2  # LEFT
                    else:
                        return 0  # NOOP
            else:
                if state['Tree4.y[t-1]'] <= 124:
                    return 0  # NOOP
                else:
                    return 2  # LEFT
        else:
            if state['Mogul3.x#1'] <= 79:
                if state['Mogul1.x#1'] <= 126:
                    if state['Player1.x'] <= 78:
                        if state['Mogul3.x[t-1]'] <= 48:
                            if state['Mogul2.y#1'] <= 179:
                                return 2  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['Player1.x#1'] <= 77:
                                return 2  # LEFT
                            else:
                                return 2  # LEFT
                    else:
                        if state['Tree3.y[t-1]'] <= 171:
                            if state['O(Player1)'] <= 9:
                                return 2  # LEFT
                            else:
                                return 2  # LEFT
                        else:
                            if state['Player1.x[t-1]'] <= 82:
                                return 2  # LEFT
                            else:
                                return 2  # LEFT
                else:
                    if state['Tree4.y#1'] <= 51:
                        return 2  # LEFT
                    else:
                        return 0  # NOOP
            else:
                if state['Player1.x[t-1]'] <= 78:
                    if state['Tree2.y#1'] <= 22:
                        if state['Mogul1.y'] <= 86:
                            return 0  # NOOP
                        else:
                            return 2  # LEFT
                    else:
                        if state['Tree3.y[t-1]'] <= 169:
                            return 2  # LEFT
                        else:
                            return 0  # NOOP
                else:
                    if state['Tree4.y[t-1]'] <= 176:
                        if state['Mogul2.y'] <= 169:
                            if state['Tree2.y'] <= 150:
                                return 2  # LEFT
                            else:
                                return 2  # LEFT
                        else:
                            return 0  # NOOP
                    else:
                        if state['D(Player1, Flag1).y'] <= 113:
                            return 2  # LEFT
                        else:
                            if state['Player1.x'] <= 83:
                                return 0  # NOOP
                            else:
                                return 2  # LEFT
    else:
        if state['DV(Player1).x'] <= 0:
            if state['Player1.x[t-1]'] <= 76:
                if state['Flag3.x[t-1]'] <= 13:
                    if state['Player1.x[t-1]'] <= 75:
                        return 1  # RIGHT
                    else:
                        if state['Mogul2.y[t-1]'] <= 81:
                            if state['O(Player1)'] <= 11:
                                return 1  # RIGHT
                            else:
                                return 1  # RIGHT
                        else:
                            return 0  # NOOP
                else:
                    if state['Mogul1.y[t-1]'] <= 65:
                        if state['Tree1.y[t-1]'] <= 47:
                            if state['Tree4.y[t-1]'] <= 16:
                                return 0  # NOOP
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Tree4.x[t-1]'] <= 147:
                                return 0  # NOOP
                            else:
                                return 1  # RIGHT
                    else:
                        if state['Tree4.y#1'] <= 109:
                            if state['Flag3.y#1'] <= 85:
                                return 1  # RIGHT
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Mogul2.x'] <= 61:
                                return 0  # NOOP
                            else:
                                return 1  # RIGHT
            else:
                if state['Player1.x[t-1]'] <= 82:
                    if state['Mogul1.x[t-1]'] <= 96:
                        if state['Tree1.x[t-1]'] <= 19:
                            if state['Mogul1.y[t-1]'] <= 177:
                                return 0  # NOOP
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Mogul2.y[t-1]'] <= 168:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                    else:
                        if state['Tree1.y#1'] <= 77:
                            if state['Flag2.x[t-1]'] <= 125:
                                return 0  # NOOP
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Player1.x[t-1]'] <= 82:
                                return 1  # RIGHT
                            else:
                                return 0  # NOOP
                else:
                    if state['Mogul2.y[t-1]'] <= 91:
                        if state['Player1.x#1'] <= 85:
                            if state['Tree4.y[t-1]'] <= 145:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['D(Player1, Flag1).x'] <= -17:
                                return 2  # LEFT
                            else:
                                return 0  # NOOP
                    else:
                        if state['Tree2.x#1'] <= 74:
                            if state['Tree3.y[t-1]'] <= 138:
                                return 2  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['Mogul1.x[t-1]'] <= 49:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
        else:
            if state['DV(Flag1).x'] <= 76:
                if state['D(Player1, Flag1).x'] <= -86:
                    if state['Mogul1.y#1'] <= 29:
                        if state['Mogul2.y[t-1]'] <= 105:
                            return 0  # NOOP
                        else:
                            return 1  # RIGHT
                    else:
                        return 1  # RIGHT
                else:
                    if state['Flag3.y[t-1]'] <= 135:
                        if state['Tree2.y'] <= 180:
                            if state['Mogul1.y[t-1]'] <= 108:
                                return 1  # RIGHT
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Mogul1.y#1'] <= 52:
                                return 1  # RIGHT
                            else:
                                return 1  # RIGHT
                    else:
                        if state['Player1.x#1'] <= 84:
                            if state['Player1.x#1'] <= 83:
                                return 1  # RIGHT
                            else:
                                return 1  # RIGHT
                        else:
                            if state['Mogul1.y'] <= 83:
                                return 1  # RIGHT
                            else:
                                return 0  # NOOP
            else:
                if state['Player1.x#1'] <= 83:
                    if state['Flag4.y#1'] <= 120:
                        return 1  # RIGHT
                    else:
                        if state['Tree1.y[t-1]'] <= 178:
                            return 0  # NOOP
                        else:
                            return 0  # NOOP
                else:
                    if state['D(Player1, Flag1).x'] <= -86:
                        if state['Mogul2.y[t-1]'] <= 118:
                            return 0  # NOOP
                        else:
                            return 0  # NOOP
                    else:
                        return 0  # NOOP
