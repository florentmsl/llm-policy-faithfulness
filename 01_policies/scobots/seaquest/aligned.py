def play(state):
    if state['Player1.y[t-1]'] <= 118:
        if state['Shark1.x#1'] <= 97:
            if state['Shark4.x'] <= 50:
                if state['Diver2.x#1'] <= 0:
                    if state['Player1.x'] <= 96:
                        if state['Shark10.y'] <= 33:
                            if state['PlayerMissile1.y'] <= 81:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 8  # DOWNRIGHT
                        else:
                            if state['Shark6.x#1'] <= 21:
                                return 8  # DOWNRIGHT
                            else:
                                return 8  # DOWNRIGHT
                    else:
                        if state['Player1.y'] <= 94:
                            if state['Shark6.x'] <= 66:
                                return 9  # DOWNLEFT
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['PlayerMissile1.x'] <= 2:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                else:
                    if state['Shark6.x[t-1]'] <= 2:
                        if state['Player1.y[t-1]'] <= 75:
                            if state['Player1.x'] <= 55:
                                return 9  # DOWNLEFT
                            else:
                                return 17  # DOWNLEFTFIRE
                        else:
                            if state['PlayerMissile1.y#1'] <= 118:
                                return 13  # DOWNFIRE
                            else:
                                return 17  # DOWNLEFTFIRE
                    else:
                        if state['Player1.x#1'] <= 49:
                            if state['CollectedDiver2.y'] <= 89:
                                return 8  # DOWNRIGHT
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['PlayerMissile1.x#1'] <= 9:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 8  # DOWNRIGHT
            else:
                if state['Shark3.y[t-1]'] <= 69:
                    if state['Player1.y'] <= 111:
                        if state['PlayerMissile1.y[t-1]'] <= 116:
                            if state['PlayerMissile1.x'] <= 96:
                                return 11  # RIGHTFIRE
                            else:
                                return 5  # DOWN
                        else:
                            if state['PlayerMissile1.x'] <= 94:
                                return 11  # RIGHTFIRE
                            else:
                                return 11  # RIGHTFIRE
                    else:
                        if state['PlayerMissile1.x#1'] <= 80:
                            if state['Shark7.x#1'] <= 37:
                                return 11  # RIGHTFIRE
                            else:
                                return 14  # UPRIGHTFIRE
                        else:
                            if state['Player1.x#1'] <= 87:
                                return 11  # RIGHTFIRE
                            else:
                                return 7  # UPLEFT
                else:
                    if state['PlayerMissile1.x#1'] <= 1:
                        if state['Player1.y[t-1]'] <= 108:
                            if state['Shark4.x#1'] <= 149:
                                return 3  # RIGHT
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Shark9.x[t-1]'] <= 108:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 8  # DOWNRIGHT
                    else:
                        if state['Player1.x#1'] <= 66:
                            if state['Shark7.x'] <= 113:
                                return 8  # DOWNRIGHT
                            else:
                                return 17  # DOWNLEFTFIRE
                        else:
                            if state['Shark9.y'] <= 45:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 5  # DOWN
        else:
            if state['Shark6.x#1'] <= 7:
                if state['Shark4.x[t-1]'] <= 53:
                    if state['OxygenBar1.x[t-1]'] <= 25:
                        if state['Shark7.y#1'] <= 90:
                            if state['Submarine9.x#1'] <= 15:
                                return 13  # DOWNFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Shark1.y[t-1]'] <= 143:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 0  # NOOP
                    else:
                        if state['PlayerMissile1.x#1'] <= 58:
                            if state['Shark9.y[t-1]'] <= 91:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Player1.x[t-1]'] <= 89:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 9  # DOWNLEFT
                else:
                    if state['Player1.x[t-1]'] <= 59:
                        if state['PlayerMissile1.x#1'] <= 90:
                            if state['Shark7.y[t-1]'] <= 90:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['PlayerMissile1.y[t-1]'] <= 115:
                                return 5  # DOWN
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['PlayerMissile1.x#1'] <= 100:
                            if state['Shark4.x[t-1]'] <= 156:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 5  # DOWN
                        else:
                            if state['Diver1.x'] <= 140:
                                return 5  # DOWN
                            else:
                                return 16  # DOWNRIGHTFIRE
            else:
                if state['PlayerMissile1.y'] <= 100:
                    if state['Diver4.x#1'] <= 5:
                        if state['Player1.x'] <= 11:
                            if state['Submarine7.y#1'] <= 47:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['PlayerMissile1.x'] <= 14:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['PlayerMissile1.x'] <= 29:
                            if state['Shark1.y[t-1]'] <= 69:
                                return 9  # DOWNLEFT
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Player1.x#1'] <= 126:
                                return 5  # DOWN
                            else:
                                return 9  # DOWNLEFT
                else:
                    if state['Player1.x#1'] <= 108:
                        if state['Shark10.x[t-1]'] <= 6:
                            if state['Shark10.x[t-1]'] <= 2:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 5  # DOWN
                        else:
                            if state['PlayerMissile1.x'] <= 93:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['Shark6.x'] <= 51:
                            if state['Diver4.y[t-1]'] <= 71:
                                return 5  # DOWN
                            else:
                                return 9  # DOWNLEFT
                        else:
                            if state['Shark1.y[t-1]'] <= 137:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
    else:
        if state['Shark6.y'] <= 56:
            if state['Shark4.x#1'] <= 104:
                if state['Player1.x#1'] <= 73:
                    if state['PlayerMissile1.x'] <= 1:
                        if state['Shark3.y'] <= 69:
                            if state['Shark1.x#1'] <= 2:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 11  # RIGHTFIRE
                        else:
                            if state['Player1.x#1'] <= 33:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 17  # DOWNLEFTFIRE
                    else:
                        if state['Diver1.y'] <= 34:
                            if state['Player1.y[t-1]'] <= 134:
                                return 8  # DOWNRIGHT
                            else:
                                return 14  # UPRIGHTFIRE
                        else:
                            if state['Player1.x#1'] <= 35:
                                return 8  # DOWNRIGHT
                            else:
                                return 8  # DOWNRIGHT
                else:
                    if state['Shark1.x#1'] <= 23:
                        if state['PlayerMissile1.y#1'] <= 136:
                            if state['Shark3.y#1'] <= 69:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 17  # DOWNLEFTFIRE
                        else:
                            if state['Shark4.y'] <= 56:
                                return 9  # DOWNLEFT
                            else:
                                return 7  # UPLEFT
                    else:
                        if state['PlayerMissile1.x'] <= 1:
                            if state['PlayerMissile1.x[t-1]'] <= 1:
                                return 14  # UPRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Player1.y'] <= 136:
                                return 13  # DOWNFIRE
                            else:
                                return 10  # UPFIRE
            else:
                if state['Shark1.x#1'] <= 40:
                    if state['Shark3.y'] <= 69:
                        if state['PlayerMissile1.x'] <= 1:
                            if state['Player1.y#1'] <= 135:
                                return 14  # UPRIGHTFIRE
                            else:
                                return 14  # UPRIGHTFIRE
                        else:
                            if state['Player1.x[t-1]'] <= 82:
                                return 14  # UPRIGHTFIRE
                            else:
                                return 7  # UPLEFT
                    else:
                        if state['PlayerMissile1.y'] <= 61:
                            if state['Player1.y[t-1]'] <= 137:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 15  # UPLEFTFIRE
                        else:
                            if state['PlayerMissile1.y#1'] <= 139:
                                return 8  # DOWNRIGHT
                            else:
                                return 15  # UPLEFTFIRE
                else:
                    if state['PlayerMissile1.y'] <= 129:
                        if state['PlayerMissile1.x[t-1]'] <= 73:
                            if state['Player1.y'] <= 138:
                                return 11  # RIGHTFIRE
                            else:
                                return 14  # UPRIGHTFIRE
                        else:
                            if state['PlayerMissile1.x#1'] <= 35:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['Player1.y'] <= 134:
                            if state['Player1.x[t-1]'] <= 82:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 5  # DOWN
                        else:
                            if state['Player1.x[t-1]'] <= 64:
                                return 11  # RIGHTFIRE
                            else:
                                return 2  # UP
        else:
            if state['Shark1.x#1'] <= 59:
                if state['PlayerMissile1.x'] <= 2:
                    if state['Shark3.x'] <= 1:
                        if state['Player1.x#1'] <= 31:
                            if state['Shark6.x#1'] <= 12:
                                return 15  # UPLEFTFIRE
                            else:
                                return 14  # UPRIGHTFIRE
                        else:
                            if state['PlayerMissile1.y[t-1]'] <= 131:
                                return 15  # UPLEFTFIRE
                            else:
                                return 15  # UPLEFTFIRE
                    else:
                        if state['Player1.y'] <= 135:
                            if state['Shark6.x#1'] <= 83:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['Shark10.x[t-1]'] <= 4:
                                return 17  # DOWNLEFTFIRE
                            else:
                                return 15  # UPLEFTFIRE
                else:
                    if state['Shark3.x#1'] <= 1:
                        if state['Player1.x'] <= 82:
                            if state['Diver1.x#1'] <= 12:
                                return 6  # UPRIGHT
                            else:
                                return 6  # UPRIGHT
                        else:
                            if state['PlayerMissile1.x'] <= 49:
                                return 15  # UPLEFTFIRE
                            else:
                                return 15  # UPLEFTFIRE
                    else:
                        if state['PlayerMissile1.y#1'] <= 141:
                            if state['Shark7.x'] <= 85:
                                return 8  # DOWNRIGHT
                            else:
                                return 17  # DOWNLEFTFIRE
                        else:
                            if state['PlayerMissile1.x#1'] <= 45:
                                return 15  # UPLEFTFIRE
                            else:
                                return 3  # RIGHT
            else:
                if state['PlayerMissile1.y'] <= 131:
                    if state['PlayerMissile1.x[t-1]'] <= 139:
                        if state['Player1.x#1'] <= 28:
                            if state['Shark1.x#1'] <= 94:
                                return 8  # DOWNRIGHT
                            else:
                                return 16  # DOWNRIGHTFIRE
                        else:
                            if state['PlayerMissile1.x#1'] <= 59:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['PlayerMissile1.y[t-1]'] <= 139:
                            if state['Diver2.x[t-1]'] <= 148:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 11  # RIGHTFIRE
                        else:
                            if state['Diver4.x#1'] <= 128:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 11  # RIGHTFIRE
                else:
                    if state['PlayerMissile1.y#1'] <= 144:
                        if state['Player1.x'] <= 84:
                            if state['Player1.x'] <= 23:
                                return 14  # UPRIGHTFIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['Player1.x[t-1]'] <= 101:
                                return 16  # DOWNRIGHTFIRE
                            else:
                                return 16  # DOWNRIGHTFIRE
                    else:
                        if state['Player1.x'] <= 85:
                            if state['Player1.x'] <= 48:
                                return 11  # RIGHTFIRE
                            else:
                                return 3  # RIGHT
                        else:
                            if state['PlayerMissile1.y[t-1]'] <= 70:
                                return 10  # UPFIRE
                            else:
                                return 2  # UP
