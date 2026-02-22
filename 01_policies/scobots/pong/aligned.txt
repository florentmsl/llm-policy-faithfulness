def play(state):
    if state['LT(Player1, Ball1).y'] <= 6:
        if state['C(Enemy1, Player1).y'] <= 155:
            if state['D(Player1, Ball1).y'] <= 7:
                if state['D(Ball1, Enemy1).x'] <= -59:
                    if state['D(Player1, Ball1).y'] <= -4:
                        if state['ED(Enemy1, Ball1)'] <= 91:
                            if state['D(Ball1, Player1).y'] <= 30:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            if state['LT(Enemy1, Player1).y'] <= -28494:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                    else:
                        if state['LT(Ball1, Player1).y'] <= -1079:
                            if state['LT(Ball1, Enemy1).y'] <= 3618:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                        else:
                            if state['DV(Ball1).y'] <= 2:
                                return 3  # LEFT
                            else:
                                return 2  # RIGHT
                else:
                    if state['LT(Ball1, Enemy1).y'] <= 2327:
                        if state['C(Ball1, Player1).y'] <= 56:
                            if state['D(Ball1, Enemy1).y'] <= 2:
                                return 3  # LEFT
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['LT(Player1, Enemy1).y'] <= -2477:
                                return 4  # RIGHTFIRE
                            else:
                                return 3  # LEFT
                    else:
                        if state['LT(Ball1, Player1).y'] <= -9709:
                            if state['C(Ball1, Player1).x'] <= 106:
                                return 5  # LEFTFIRE
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['LT(Player1, Ball1).y'] <= -102:
                                return 4  # RIGHTFIRE
                            else:
                                return 0  # NOOP
            else:
                if state['LT(Enemy1, Ball1).y'] <= 238:
                    if state['LT(Ball1, Enemy1).y'] <= 3441:
                        if state['Ball1.x'] <= 142:
                            if state['LT(Ball1, Player1).x'] <= 0:
                                return 3  # LEFT
                            else:
                                return 1  # FIRE
                        else:
                            if state['ED(Ball1, Player1)'] <= 11:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                    else:
                        if state['Player1.y[t-1]'] <= 130:
                            if state['LT(Ball1, Player1).y'] <= -12:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            if state['LT(Ball1, Player1).y'] <= -1499:
                                return 0  # NOOP
                            else:
                                return 1  # FIRE
                else:
                    if state['V(Player1).x'] <= 5:
                        if state['Enemy1.y[t-1]'] <= 140:
                            if state['LT(Player1, Enemy1).y'] <= -7430:
                                return 4  # RIGHTFIRE
                            else:
                                return 1  # FIRE
                        else:
                            if state['Ball1.x'] <= 139:
                                return 4  # RIGHTFIRE
                            else:
                                return 4  # RIGHTFIRE
                    else:
                        if state['ED(Enemy1, Player1)'] <= 125:
                            if state['Player1.y'] <= 120:
                                return 1  # FIRE
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['LT(Enemy1, Player1).y'] <= -23579:
                                return 0  # NOOP
                            else:
                                return 1  # FIRE
        else:
            if state['ED(Enemy1, Ball1)'] <= 125:
                if state['C(Player1, Ball1).x'] <= 128:
                    if state['LT(Enemy1, Ball1).y'] <= -72:
                        if state['LT(Ball1, Enemy1).y'] <= -3300:
                            if state['LT(Ball1, Player1).y'] <= -3742:
                                return 5  # LEFTFIRE
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['LT(Player1, Enemy1).y'] <= 4932:
                                return 4  # RIGHTFIRE
                            else:
                                return 3  # LEFT
                    else:
                        if state['LT(Ball1, Enemy1).y'] <= 2464:
                            if state['LT(Player1, Ball1).y'] <= -9:
                                return 0  # NOOP
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['LT(Player1, Ball1).y'] <= -38:
                                return 4  # RIGHTFIRE
                            else:
                                return 3  # LEFT
                else:
                    if state['LT(Player1, Ball1).y'] <= -32:
                        if state['ED(Enemy1, Player1)'] <= 125:
                            if state['LT(Enemy1, Enemy1).x'] <= -8:
                                return 4  # RIGHTFIRE
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['D(Ball1, Player1).y'] <= 24:
                                return 4  # RIGHTFIRE
                            else:
                                return 2  # RIGHT
                    else:
                        if state['LT(Player1, Ball1).y'] <= -5:
                            if state['DV(Player1).y'] <= 21:
                                return 4  # RIGHTFIRE
                            else:
                                return 0  # NOOP
                        else:
                            if state['LT(Ball1, Player1).y'] <= 139:
                                return 5  # LEFTFIRE
                            else:
                                return 4  # RIGHTFIRE
            else:
                if state['Ball1.y#1'] <= 183:
                    if state['Enemy1.y[t-1]'] <= 136:
                        if state['D(Player1, Ball1).y'] <= -183:
                            if state['V(Player1).x'] <= 2:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                        else:
                            if state['C(Enemy1, Player1).y'] <= 161:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                    else:
                        if state['Ball1.y'] <= 168:
                            if state['LT(Player1, Enemy1).y'] <= -8703:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Enemy1, Player1).y'] <= -620:
                                return 5  # LEFTFIRE
                            else:
                                return 5  # LEFTFIRE
                else:
                    if state['LT(Player1, Ball1).y'] <= -15:
                        if state['LT(Enemy1, Player1).y'] <= 22:
                            if state['D(Ball1, Enemy1).y'] <= -22:
                                return 5  # LEFTFIRE
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Enemy1, Player1).y'] <= 2500:
                                return 5  # LEFTFIRE
                            else:
                                return 3  # LEFT
                    else:
                        if state['DV(Player1).y'] <= 3:
                            if state['LT(Ball1, Enemy1).y'] <= -2711:
                                return 3  # LEFT
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['LT(Player1, Ball1).y'] <= 1:
                                return 5  # LEFTFIRE
                            else:
                                return 5  # LEFTFIRE
    else:
        if state['LT(Enemy1, Ball1).y'] <= 233:
            if state['Player1.y[t-1]'] <= 138:
                if state['D(Ball1, Player1).y'] <= -3:
                    if state['DV(Ball1).x'] <= -1:
                        if state['ED(Enemy1, Ball1)'] <= 103:
                            if state['D(Player1, Ball1).y'] <= 14:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Player1, Ball1).y'] <= 12:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                    else:
                        if state['LT(Enemy1, Player1).x'] <= 0:
                            if state['D(Player1, Ball1).y'] <= 30:
                                return 0  # NOOP
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['ED(Player1, Ball1)'] <= 24:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                else:
                    if state['LT(Enemy1, Player1).y'] <= 3736:
                        if state['LT(Enemy1, Ball1).y'] <= 81:
                            if state['ED(Player1, Ball1)'] <= 29:
                                return 3  # LEFT
                            else:
                                return 0  # NOOP
                        else:
                            if state['C(Ball1, Enemy1).y'] <= 48:
                                return 2  # RIGHT
                            else:
                                return 3  # LEFT
                    else:
                        if state['ED(Player1, Enemy1)'] <= 126:
                            if state['LT(Ball1, Enemy1).y'] <= -4838:
                                return 0  # NOOP
                            else:
                                return 2  # RIGHT
                        else:
                            if state['D(Ball1, Enemy1).x'] <= -64:
                                return 2  # RIGHT
                            else:
                                return 5  # LEFTFIRE
            else:
                if state['ED(Player1, Ball1)'] <= 40:
                    if state['Player1.y[t-1]'] <= 160:
                        if state['ED(Enemy1, Ball1)'] <= 115:
                            if state['LT(Player1, Ball1).y'] <= 18:
                                return 4  # RIGHTFIRE
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['Enemy1.y'] <= 164:
                                return 3  # LEFT
                            else:
                                return 5  # LEFTFIRE
                    else:
                        if state['ED(Ball1, Player1)'] <= 30:
                            if state['Enemy1.y[t-1]'] <= 142:
                                return 3  # LEFT
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['C(Player1, Ball1).y'] <= 186:
                                return 5  # LEFTFIRE
                            else:
                                return 5  # LEFTFIRE
                else:
                    if state['LT(Player1, Enemy1).y'] <= -21193:
                        if state['LT(Ball1, Enemy1).y'] <= -28470:
                            if state['LT(Player1, Ball1).y'] <= 28:
                                return 3  # LEFT
                            else:
                                return 5  # LEFTFIRE
                        else:
                            if state['ED(Ball1, Enemy1)'] <= 62:
                                return 0  # NOOP
                            else:
                                return 0  # NOOP
                    else:
                        if state['D(Enemy1, Ball1).y'] <= 13:
                            if state['LT(Ball1, Enemy1).y'] <= -5288:
                                return 2  # RIGHT
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Ball1, Player1).x'] <= 0:
                                return 5  # LEFTFIRE
                            else:
                                return 5  # LEFTFIRE
        else:
            if state['C(Player1, Ball1).y'] <= 122:
                if state['Player1.y#1'] <= 109:
                    if state['LT(Enemy1, Player1).y'] <= -11791:
                        return 3  # LEFT
                    else:
                        if state['ED(Enemy1, Player1)'] <= 125:
                            if state['LT(Ball1, Player1).y'] <= 33:
                                return 2  # RIGHT
                            else:
                                return 3  # LEFT
                        else:
                            if state['D(Ball1, Player1).y'] <= -17:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                else:
                    if state['D(Enemy1, Ball1).x'] <= 131:
                        if state['DV(Ball1).y'] <= 127:
                            if state['LT(Player1, Enemy1).y'] <= 9945:
                                return 4  # RIGHTFIRE
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Player1, Enemy1).y'] <= 9900:
                                return 4  # RIGHTFIRE
                            else:
                                return 4  # RIGHTFIRE
                    else:
                        if state['D(Player1, Ball1).y'] <= 6:
                            if state['LT(Ball1, Player1).x'] <= 0:
                                return 2  # RIGHT
                            else:
                                return 2  # RIGHT
                        else:
                            if state['LT(Enemy1, Player1).y'] <= 3703:
                                return 1  # FIRE
                            else:
                                return 2  # RIGHT
            else:
                if state['LT(Enemy1, Player1).y'] <= -24829:
                    if state['Player1.y'] <= 123:
                        if state['C(Ball1, Player1).y'] <= 137:
                            if state['LT(Ball1, Enemy1).y'] <= 9483:
                                return 3  # LEFT
                            else:
                                return 3  # LEFT
                        else:
                            return 1  # FIRE
                    else:
                        if state['Ball1.y[t-1]'] <= 157:
                            if state['C(Ball1, Player1).y'] <= 139:
                                return 0  # NOOP
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Player1, Enemy1).y'] <= 9948:
                                return 1  # FIRE
                            else:
                                return 1  # FIRE
                else:
                    if state['ED(Ball1, Player1)'] <= 13:
                        if state['DV(Player1).y'] <= -6:
                            if state['Ball1.y'] <= 137:
                                return 2  # RIGHT
                            else:
                                return 4  # RIGHTFIRE
                        else:
                            if state['D(Ball1, Player1).y'] <= -10:
                                return 4  # RIGHTFIRE
                            else:
                                return 4  # RIGHTFIRE
                    else:
                        if state['Player1.y#1'] <= 128:
                            if state['LT(Player1, Ball1).y'] <= 20:
                                return 1  # FIRE
                            else:
                                return 3  # LEFT
                        else:
                            if state['LT(Enemy1, Player1).y'] <= -11167:
                                return 1  # FIRE
                            else:
                                return 4  # RIGHTFIRE
