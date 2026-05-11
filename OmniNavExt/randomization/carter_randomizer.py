from .robot_randomizer import RobotRandomizer

"""
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class CarterRandomizer(RobotRandomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        # Carter's AABB min and max
        self.extent = [(-1.101184962241601, -0.45593118604327454), (0.1407164487669442, 0.45593120044146757)]
