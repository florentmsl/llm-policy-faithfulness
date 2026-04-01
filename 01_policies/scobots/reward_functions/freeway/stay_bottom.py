"""Reward staying near the bottom of the screen.
Produces an agent that refuses to cross — the opposite of the game objective.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find Chicken1 position indices
    player_idxs = np.empty(0)
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "POSITION" and feature_signature == "Chicken1":
            player_idxs = np.where(fv_backmap == i)[0]

    if not player_idxs.any():
        return None

    BOTTOM_Y = 180.0  # approximate starting y position

    def reward(fv):
        y = fv[player_idxs[0] + 1]
        # Reward proximity to bottom (high y = bottom of screen)
        return (y / BOTTOM_Y) * 0.1

    return reward
