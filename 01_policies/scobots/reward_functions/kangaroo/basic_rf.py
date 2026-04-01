"""Reward climbing higher (lower y = higher on screen).
Dense shaped reward that encourages upward progression toward the child.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find Player1 position indices
    player_idxs = np.empty(0)
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "POSITION" and feature_signature == "Player1":
            player_idxs = np.where(fv_backmap == i)[0]

    if not player_idxs.any():
        return None

    def reward(fv):
        y = fv[player_idxs[0] + 1]  # y coordinate (lower = higher on screen)

        focus.reward_history[0] = focus.reward_history[1]
        focus.reward_history[1] = y
        delta = focus.reward_history[0] - focus.reward_history[1]  # positive when climbing
        return delta * 0.1

    return reward
