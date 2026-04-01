"""Reward tracking the ball with the paddle (y-axis alignment).
Dense shaped reward that encourages the paddle to follow the ball vertically.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find DISTANCE(Player1, Ball1) indices
    dist_idxs = np.empty(0)
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "DISTANCE":
            input1 = feature_signature[0]
            input2 = feature_signature[1]
            if input1[0] == "POSITION" and input1[1] == "Player1" and \
               input2[0] == "POSITION" and input2[1] == "Ball1":
                dist_idxs = np.where(fv_backmap == i)[0]

    if not dist_idxs.any():
        return None

    def reward(fv):
        d_entries = fv[dist_idxs[0]:dist_idxs[-1] + 1]
        y_dist = abs(d_entries[1])  # y-distance between paddle and ball

        focus.reward_history[0] = focus.reward_history[1]
        focus.reward_history[1] = y_dist
        delta = focus.reward_history[0] - focus.reward_history[1]  # positive when getting closer
        return delta * 1.0

    return reward
