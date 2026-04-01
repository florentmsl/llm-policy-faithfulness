from ocatari.ram.skiing import *

def reward_function(self) -> float:
    reward = 0.0
    game_objects = self.objects

    player = None
    flags = []

    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Flag):
            flags.append(obj)

    if player and len(flags) >= 2:
        # Reward being horizontally centered between the nearest flag pair
        # Flags come in pairs (left and right poles of a gate)
        flag_xs = sorted([f.x for f in flags])
        if len(flag_xs) >= 2:
            gate_center = (flag_xs[0] + flag_xs[1]) / 2.0
            dist_to_center = abs(player.x - gate_center)
            reward += max(0, (80 - dist_to_center) / 80.0) * 0.5

    return reward
