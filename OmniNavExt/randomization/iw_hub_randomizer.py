from .robot_randomizer import RobotRandomizer

"""
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class IwHubRandomizer(RobotRandomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        # iw.hub's AABB min and max
        self.extent = [(-1.0332839734863342, -0.3290388065636699), (0.3975681979007675, 0.3300932763404468)]
