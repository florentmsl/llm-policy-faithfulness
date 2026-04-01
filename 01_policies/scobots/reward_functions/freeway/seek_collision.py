"""Reward decreasing distance to nearest car.
Produces a misaligned agent that actively seeks car collisions.
"""
import numpy as np


def make_reward(fv_description, fv_backmap, focus):
    # Find DISTANCE(Chicken1, Car*) indices
    distance_idxs_list = []
    for i, feature in enumerate(fv_description):
        feature_name = feature[0]
        feature_signature = feature[1]
        if feature_name == "DISTANCE":
            input1 = feature_signature[0]
            input2 = feature_signature[1]
            if input1[0] == "POSITION" and input1[1] == "Chicken1" and \
               input2[0] == "POSITION" and "Car" in input2[1]:
                idxs = np.where(fv_backmap == i)[0]
                if idxs.any():
                    distance_idxs_list.append(idxs)

    if not distance_idxs_list:
        return None

    def reward(fv):
        # Find minimum distance to any car (using euclidean-like: abs(x) + abs(y))
        min_dist = float("inf")
        for idxs in distance_idxs_list:
            d_entries = fv[idxs[0]:idxs[-1] + 1]
            dist = abs(d_entries[0]) + abs(d_entries[1])
            if dist < min_dist:
                min_dist = dist

        focus.reward_history[0] = focus.reward_history[1]
        focus.reward_history[1] = min_dist
        delta = focus.reward_history[0] - focus.reward_history[1]  # positive when getting closer
        return delta * 1.0

    return reward
