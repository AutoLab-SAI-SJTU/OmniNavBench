import carb
import numpy as np
import omni.anim.navigation.core as nav
from omni.metropolis.utils.carb_util import CarbUtil
from omni.metropolis.utils.simulation_util import SimulationUtil

from .randomizer import Randomizer
from .randomizer_util import RandomizerUtil


class CameraRandomizer(Randomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)

    # get random_camera_position and rotation
    def get_random_camera_transform(self, idx, character_list=None):

        valid = False
        spawn_location = carb.Float3(0, 0, 0)
        spawn_rotation = None
        focus_character = None
        navmesh = self.inav.get_navmesh()
        if navmesh is None:
            carb.log_error("Navmesh not found when trying to get random positions")
            return

        if character_list is not None:
            # use an predefined raycast list to ensure that there is no block between character and camera
            result_list = RandomizerUtil.get_character_raycast_check(character_list)
            spawn_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id + idx)
            np.random.seed(spawn_seed)
            np.random.shuffle(result_list)
            i = 0
            # try to find a valid spawn location along those direction
            while not valid:
                # attempts over limitation
                if i == len(result_list):
                    return valid, None, spawn_location, spawn_rotation

                raycast_info = result_list[i]
                camera_distance = None
                valid_navigation_position = False
                temp_location = carb.Float3(0, 0, 0)
                calculated_location = carb.Float3(0, 0, 0)
                count = 0

                # try to find the valid position that can be projected to navmesh
                while not valid_navigation_position and count < 1000:
                    camera_distance = np.random.uniform(raycast_info.min_distance, raycast_info.max_distance)
                    calculated_location = CarbUtil.add3(
                        raycast_info.cast_position, CarbUtil.scale3(raycast_info.cast_dir, camera_distance)
                    )
                    temp_location = navmesh.query_closest_point(
                        carb.Float3(calculated_location[0], calculated_location[1], 0.0)
                    )
                    compare_location = carb.Float3(temp_location[0], temp_location[1], 0.0)
                    valid_navigation_position = SimulationUtil.is_the_same_point(
                        carb.Float3(calculated_location[0], calculated_location[1], 0.0), compare_location
                    )
                    count = count + 1

                # get spawn location
                spawn_location = carb.Float3(temp_location[0], temp_location[1], calculated_location[2])
                valid = valid_navigation_position

                # if spawn location is on navmesh, then we check whether the point is ovelapped with other points
                if valid:
                    for pos in self._existing_pos:
                        if CarbUtil.dist3(carb.Float3(pos), spawn_location) < 4:
                            valid = False
                            break

                if valid:
                    # get camera rotation: make the camera looks at target character
                    spawn_rotation = RandomizerUtil.get_camera_rotation(raycast_info.character_name, spawn_location)
                    valid = spawn_rotation is not None

                if valid:
                    # extract focus character name from raycast info.
                    focus_character = raycast_info.character_name

                i += 1

            self._existing_pos.append(spawn_location)
            return valid, focus_character, spawn_location, spawn_rotation

    def get_random_position_rotation(self, camera_count, character_list=None):
        idx_to_transform = {}

        if RandomizerUtil.do_aim_camera_to_character() and len(character_list) == 0:
            carb.log_warn("There is no character in the scene, cannot aim camera to character ..")

        if (not RandomizerUtil.do_aim_camera_to_character()) or (character_list is None) or (len(character_list) == 0):
            for i in range(camera_count):
                idx_to_transform[i] = (super().get_random_position(i), None)
            return idx_to_transform

        character_dict = {}

        for element in character_list:
            # check every element in the dict, set default to True
            character_dict[element] = True

        # group character into several cluster base on character's position and camera number
        shuffle_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id)
        grouped_characters = RandomizerUtil.group_elements(character_list, camera_count, shuffle_seed)

        # Try focusing each camera to a character in its group
        for i in range(0, camera_count):
            valid, focus_character, spawn_location, spawn_rotation = self.get_random_camera_transform(
                i, grouped_characters[i]
            )
            # if camera is focusing on character
            if valid:
                character_dict[focus_character] = False
                # record the rotation and location to the dictionary
                idx_to_transform[i] = (spawn_location, spawn_rotation)
            # Can't find a character in its group to focus on
            else:
                idx_to_transform[i] = (None, None)

        # Check every single character to see if the camera can focus on
        for i in range(0, camera_count):
            if idx_to_transform[i][0] is None or idx_to_transform[i][1] is None:
                idx_to_transform[i] = self.try_character_in_list(i, character_dict)

        return idx_to_transform

    def try_character_in_list(self, idx, character_dict):
        for character in character_dict.keys():
            if character_dict[character]:
                valid, focus_character, spawn_location, spawn_rotation = self.get_random_camera_transform(
                    idx, [character]
                )
                if valid:
                    character_dict[focus_character] = False
                    return (spawn_location, spawn_rotation)
        return (None, None)

    def get_random_camera_focallength_list(self, cam_count):
        random_focal_lengths = [None] * cam_count
        if RandomizerUtil.do_randomize_camera_info():
            spawn_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id)
            np.random.seed(spawn_seed)
            max_focal_length = RandomizerUtil.get_max_camera_focallength()
            min_focal_length = RandomizerUtil.get_min_camera_focallength()
            # if max focal length equal to the min focal length
            if max_focal_length == min_focal_length:
                # set every camera 's focal length equal to min focal length
                random_focal_lengths = [min_focal_length] * cam_count
            # if max focal length > min focal length
            elif max_focal_length > min_focal_length:
                # generate randomized value within the range
                random_numbers = np.random.uniform(min_focal_length, max_focal_length, cam_count)
                # Round to two decimal places
                random_focal_lengths = [float("{:.2f}".format(num)) for num in random_numbers]
            # else: min focal length > max focal length
            else:
                # throw error message to inform user
                carb.log_error("The max focal length should not be shorter than min focal length")
        return random_focal_lengths


class LidarCameraRandomizer(CameraRandomizer):
    # get random_camera_position and rotation
    def get_random_lidar_transform(self, character_list=None):
        return super().get_random_camera_transform(character_list)

    def get_random_position_rotation(self, idx, character_list=None):
        return super().get_random_position_rotation(idx, character_list)

    def get_random_camera_focallength_list(self, cam_count):
        return super().get_random_camera_focallength_list(cam_count)
