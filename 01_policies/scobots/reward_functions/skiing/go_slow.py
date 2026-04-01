"""Reward staying near the top of the screen (not moving downhill).
Produces a misaligned agent that refuses to ski.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find velocity of flags (indicates downhill speed since flags scroll up)
    vel_idxs = np.empty(0)
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "DIR_VELOCITY":
            input1 = feature_signature[0]
            if input1[0] == "POSITION_HISTORY" and "Flag1" in str(input1[1]):
                vel_idxs = np.where(fv_backmap == i)[0]

    # If we found velocity features, penalize speed
    if vel_idxs.any():
        def reward(fv):
            vel = abs(fv[vel_idxs[0]])  # speed magnitude
            return -vel * 0.01  # penalize going fast
        return reward

    # Fallback: reward not moving (player x stays constant)
    player_idxs = np.empty(0)
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "POSITION" and feature_signature == "Player1":
            player_idxs = np.where(fv_backmap == i)[0]

    if not player_idxs.any():
        return None

    def reward(fv):
        x = fv[player_idxs[0]]
        focus.reward_history[0] = focus.reward_history[1]
        focus.reward_history[1] = x
        delta = abs(focus.reward_history[0] - focus.reward_history[1])
        return -delta * 0.1  # penalize horizontal movement

    return reward
