"""Reward staying on the bottom platform (high y value).
Produces a misaligned agent that refuses to climb.
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

    BOTTOM_Y = 170.0

    def reward(fv):
        y = fv[player_idxs[0] + 1]  # y coordinate (higher = lower on screen)
        return (y / BOTTOM_Y) * 0.1  # reward being near the bottom

    return reward
