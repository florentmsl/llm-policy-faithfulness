"""Reward passing through flag gates (staying between flag pairs).
Dense shaped reward for horizontal alignment with flag center.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find DISTANCE(Player1, Flag*) or CENTER(Flag1, Flag2) indices
    # Try to find player position and flag positions
    player_idxs = np.empty(0)
    flag_idxs_list = []

    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "POSITION" and feature_signature == "Player1":
            player_idxs = np.where(fv_backmap == i)[0]
        elif feature_name == "EUCLIDEAN_DISTANCE":
            input1 = feature_signature[0]
            input2 = feature_signature[1]
            if input1[0] == "POSITION" and input1[1] == "Player1" and \
               input2[0] == "POSITION" and "Flag" in str(input2[1]):
                idxs = np.where(fv_backmap == i)[0]
                if idxs.any():
                    flag_idxs_list.append(idxs)

    if not player_idxs.any():
        return None

    # If we have euclidean distances to flags, use them
    if flag_idxs_list:
        def reward(fv):
            min_dist = float("inf")
            for idxs in flag_idxs_list:
                dist = abs(fv[idxs[0]])
                if dist < min_dist:
                    min_dist = dist
            focus.reward_history[0] = focus.reward_history[1]
            focus.reward_history[1] = min_dist
            delta = focus.reward_history[0] - focus.reward_history[1]
            return delta * 0.5
        return reward

    # Fallback: reward being near center of screen (x ~ 80)
    CENTER_X = 80.0

    def reward(fv):
        x = fv[player_idxs[0]]  # x coordinate
        dist_to_center = abs(x - CENTER_X)
        return max(0, (1.0 - dist_to_center / CENTER_X)) * 0.01

    return reward
