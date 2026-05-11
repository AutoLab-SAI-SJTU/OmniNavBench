import math

import carb
import numpy as np
import omni.usd
from omni.metropolis.utils.carb_util import CarbUtil
from omni.metropolis.utils.math_util import MathUtil
from isaacsim.core.utils.rotations import lookat_to_quatf
from OmniNavExt.envset.settings import PrimPaths
from pxr import Gf, Usd, UsdGeom

# Import omni.kit.mesh.raycast - extension should be enabled before importing
import omni.kit.mesh.raycast


class RayCast_Container:  # noqa
    def __init__(self, character_name, cast_position, cast_dir, min_distance, max_distance):
        self.character_name = character_name
        self.cast_position = cast_position
        self.cast_dir = cast_dir
        self.min_distance = min_distance
        self.max_distance = max_distance


class RandomizerUtil:
    """
    This class contains the toolbox and utility functions used across different randomizers
    """

    # Randomization Related Settings

    # the height of focus point for each character
    CHARACTER_FOCUS_HEIGHT = "/persistent/exts/isaacsim.replicator.agent/character_focus_height"
    # the minimum camrea height to generate the randomized camera
    MIN_CAMERA_HEIGHT = "/persistent/exts/isaacsim.replicator.agent/min_camera_height"
    # the maxmum camrea height to generate the randomized camera
    MAX_CAMERA_HEIGHT = "/persistent/exts/isaacsim.replicator.agent/max_camera_height"
    # the minimum look down angle to generate the randomized camera
    MAX_CAMERA_LOOK_DOWN_ANGLE = "/persistent/exts/isaacsim.replicator.agent/max_camera_look_down_angle"
    # the maxmum look down angle to generate the randomized camera
    MIN_CAMERA_LOOK_DOWN_ANGLE = "/persistent/exts/isaacsim.replicator.agent/min_camera_look_down_angle"
    # aiming camera to character
    AIM_CAMERA_TO_CHARACTER = "persistent/exts/isaacsim.replicator.agent/aim_camera_to_character"
    # min camera distance between camera and character
    MIN_CAMERA_DISTANCE = "persistent/exts/isaacsim.replicator.agent/min_camera_distance"
    # max camera distance between camera and character
    MAX_CAMERA_DISTANCE = "persistent/exts/isaacsim.replicator.agent/max_camera_distance"
    # Min Camera FocalLength when gnerate the radomized camera (Optional)
    MIN_CAMERA_FOCALLENGTH = "/persistent/exts/isaacsim.replicator.agent/min_camera_focallength"
    # Max Camera FocalLength when gnerate the radomized camera (Optional)
    MAX_CAMERA_FOCALLENGTH = "/persistent/exts/isaacsim.replicator.agent/max_camera_focallength"
    # Whether randomize camera information (focal length) when generate cameras
    RANDOMIZE_CAMERA_INFO = "/persistent/exts/isaacsim.replicator.agent/randomize_camera_info"

    """
    Randomization Curve Toolbox Functions
    """

    @staticmethod
    def bias(b, t):
        return math.pow(t, math.log(b) / math.log(0.5))

    @staticmethod
    def gain(g, t):
        if t < 0.5:
            return RandomizerUtil.bias(1.0 - g, 2.0 * t) / 2.0
        else:
            return 1.0 - RandomizerUtil.bias(1.0 - g, 2.0 - 2.0 * t) / 2.0

    """  # noqa
    Error Handling Functions
    """

    # If the number is overflown, it will be set to be it modulo the max int
    @staticmethod
    def handle_overflow(n):
        return n % (2**32 - 1)  # noqa

    """  # noqa
    Randomization Helper Functions
    """
    """  # noqa
        Agent randomization related
    """

    # Given the AABB min and max, return the 4 points on the AABB plane
    @staticmethod
    def get_2d_bounding_points(min_point, max_point, pos):
        return [
            CarbUtil.add3(carb.Float3(min_point[0], min_point[1], 0.0), pos),
            CarbUtil.add3(carb.Float3(max_point[0], min_point[1], 0.0), pos),
            CarbUtil.add3(carb.Float3(max_point[0], max_point[1], 0.0), pos),
            CarbUtil.add3(carb.Float3(min_point[0], max_point[1], 0.0), pos),
        ]

    # This function makes the randomly generated clothes color on the characters look more natural
    @staticmethod
    def soften_color(r, g, b, bias_factor, gain_factor):
        softened_r = RandomizerUtil.bias(r, bias_factor)
        softened_g = RandomizerUtil.bias(g, bias_factor)
        softened_b = RandomizerUtil.bias(b, bias_factor)
        softened_r = RandomizerUtil.gain(softened_r, gain_factor)
        softened_g = RandomizerUtil.gain(softened_g, gain_factor)
        softened_b = RandomizerUtil.gain(softened_b, gain_factor)
        return softened_r, softened_g, softened_b

    # This function decompose a distance into two random vectors
    @staticmethod
    def decompose_distance(distance):
        # This makes sure that each call has a unique result
        x = np.random.uniform(-distance, distance)  # noqa
        y = math.sqrt(distance * distance - x * x)
        dir_rand = np.random.uniform(0, 1)
        if dir_rand > 0.5:
            y = -y
        z = 0  # Assuming it's always on the ground
        return carb.Float3(x, y, z)

    """  # noqa
        Camera randomization related
    """

    # This function checks whether the hitted object is a character
    @staticmethod
    def is_character(prim_path):
        character_root_path = PrimPaths.characters_parent_path()
        if str(prim_path).startswith(character_root_path):
            return True
        return False

    @staticmethod
    def group_element_case_one(character_path_list, num_clusters, seed):
        # group characters via thier location in 3d space
        character_positions = []
        stage = omni.usd.get_context().get_stage()

        for character_path in character_path_list:  # noqa
            character_prim = stage.GetPrimAtPath(character_path)
            matrix = omni.usd.get_world_transform_matrix(character_prim)
            character_positions.append(matrix.ExtractTranslation())

        # group character base on their scatter in the 3d space.
        # use K means to seperate character into cluster
        cluster_indices = MathUtil.simple_k_means(
            all_points=character_positions, num_points_to_select=num_clusters, max_iterations=300, seed=seed
        )[0]

        # create a dictionary to record grouped characters
        clustered_characters = {i: [] for i in range(num_clusters)}

        for i, character_name in enumerate(character_path_list):
            cluster_index = cluster_indices[i]
            clustered_characters[cluster_index].append(character_name)

        return clustered_characters

    # if the number of camra is larger than the number of character
    @staticmethod
    def group_element_case_two(input_list, n):
        result = []
        for i in range(n):
            result.append([input_list[i % len(input_list)]])  # noqa
        return result

    # split characters into group:
    # result is used to assign target for each cameras
    @staticmethod
    def group_elements(input_list, n, seed):
        num_elements = len(input_list)
        if num_elements == 0:
            return [None for i in range(0, n)]

        if n <= num_elements:
            return RandomizerUtil.group_element_case_one(input_list, n, seed)
        else:
            return RandomizerUtil.group_element_case_two(input_list, n)

    # get settings
    @staticmethod
    def get_min_camera_distance():  # noqa
        min_camera_distance = carb.settings.get_settings().get(RandomizerUtil.MIN_CAMERA_DISTANCE)
        return float(min_camera_distance)

    @staticmethod
    def get_max_camera_distance():  # noqa
        max_camera_distance = carb.settings.get_settings().get(RandomizerUtil.MAX_CAMERA_DISTANCE)
        return float(max_camera_distance)

    @staticmethod
    def set_min_camera_distance(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_DISTANCE, target_value)

    @staticmethod
    def set_max_camera_distance(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_DISTANCE, target_value)

    @staticmethod
    def get_character_focus_height():  # noqa
        character_height = carb.settings.get_settings().get(RandomizerUtil.CHARACTER_FOCUS_HEIGHT)
        return float(character_height)

    @staticmethod
    def get_max_camera_height():  # noqa
        max_height = carb.settings.get_settings().get(RandomizerUtil.MAX_CAMERA_HEIGHT)
        return float(max_height)

    @staticmethod
    def set_max_camera_height(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_HEIGHT, target_value)

    @staticmethod
    def get_min_camera_height():  # noqa
        min_height = carb.settings.get_settings().get(RandomizerUtil.MIN_CAMERA_HEIGHT)
        return float(min_height)

    @staticmethod
    def set_min_camera_height(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_HEIGHT, target_value)

    @staticmethod
    def get_max_camera_look_down_angle():  # noqa
        max_angle = carb.settings.get_settings().get(RandomizerUtil.MAX_CAMERA_LOOK_DOWN_ANGLE)
        return float(max_angle)

    @staticmethod
    def get_min_camera_look_down_angle():  # noqa
        min_angle = carb.settings.get_settings().get(RandomizerUtil.MIN_CAMERA_LOOK_DOWN_ANGLE)
        return float(min_angle)

    @staticmethod
    def set_max_camera_look_down_angle(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_LOOK_DOWN_ANGLE, target_value)

    @staticmethod
    def set_min_camera_look_down_angle(target_value):
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_LOOK_DOWN_ANGLE, target_value)

    @staticmethod
    def do_aim_camera_to_character():  # noqa
        aim_camera_to_character = carb.settings.get_settings().get(RandomizerUtil.AIM_CAMERA_TO_CHARACTER)
        return bool(aim_camera_to_character)

    @staticmethod
    def set_aim_camera_to_character(usr_input):
        carb.settings.get_settings().set(RandomizerUtil.AIM_CAMERA_TO_CHARACTER, usr_input)

    @staticmethod
    def get_min_camera_focallength():  # noqa
        min_focal_length = carb.settings.get_settings().get(RandomizerUtil.MIN_CAMERA_FOCALLENGTH)
        return float(min_focal_length)

    @staticmethod
    def get_max_camera_focallength():  # noqa
        max_focal_length = carb.settings.get_settings().get(RandomizerUtil.MAX_CAMERA_FOCALLENGTH)
        return float(max_focal_length)

    @staticmethod
    def do_randomize_camera_info():  # noqa
        return carb.settings.get_settings().get(RandomizerUtil.RANDOMIZE_CAMERA_INFO)

    @staticmethod
    def set_randomize_camera_info(usr_input):
        carb.settings.get_settings().set(RandomizerUtil.RANDOMIZE_CAMERA_INFO, usr_input)

    # get the radius and center of the character by calculate the bounding box
    @staticmethod
    def get_character_radius_and_center(character_path):
        character_height = RandomizerUtil.get_character_focus_height()
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(character_path)
        # get character bounding box information
        box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bound = box_cache.ComputeWorldBound(prim)
        aligned_box = bound.ComputeAlignedBox()
        center = bound.ComputeCentroid()
        # get bounding box's Max/Min vertex to calculate bounding box's size
        bbox_max = aligned_box.GetMax()
        bbox_min = aligned_box.GetMin()
        bbox_x = (bbox_max[0] - bbox_min[0]) / 2
        bbox_y = (bbox_max[1] - bbox_min[1]) / 2
        # estimate character's radius
        radius = max(bbox_y, bbox_x)
        return radius, carb.Float3(center[0], center[1], character_height)

    @staticmethod
    def get_character_surronding_raycast(character_name, radius, center, raycast_angle_step):
        # fetch camera randomization related setting value
        max_camera_height = RandomizerUtil.get_max_camera_height()
        min_camera_height = RandomizerUtil.get_min_camera_height()
        people_focus_height = RandomizerUtil.get_character_focus_height()
        max_angle = RandomizerUtil.get_max_camera_look_down_angle()
        min_angle = RandomizerUtil.get_min_camera_look_down_angle()
        min_camera_distance = RandomizerUtil.get_min_camera_distance()
        max_camera_distance = RandomizerUtil.get_max_camera_distance()

        raycast = omni.kit.mesh.raycast.get_mesh_raycast_interface()
        # data structure used to store the raycast result.
        result = []

        # calculate angle limitation from the max min distance
        max_sin_value = (max_camera_height - people_focus_height) / min_camera_distance
        min_sin_value = (min_camera_height - people_focus_height) / max_camera_distance

        max_sin_value = min(max_sin_value, 1)

        if max_sin_value < 0 or min_sin_value < 0:
            carb.log_error(
                "Camera should not be lower than character : please reset max/min_camrea_height, "
                "people_focus_height or max/min_camera_distance"
            )

        if min_sin_value > 1:
            carb.log_error(
                "Aim Camera to Character recived Invalid input : please reset max/min_camrea_height, "
                "people_focus_height or max/min_camera_distance"
            )

        # calculate radius angle limitation
        max_angle_in_radians = math.asin(max_sin_value)
        min_angle_in_radians = math.asin(min_sin_value)

        # calculate camera look down angle limitation in degree
        max_angle_in_degrees = min(math.degrees(max_angle_in_radians), max_angle)
        min_angle_in_degrees = max(math.degrees(min_angle_in_radians), min_angle)

        difference = int(max_angle_in_degrees - min_angle_in_degrees)
        number_of_steps = int(difference / raycast_angle_step)
        # handle the edge case when max camera look down angle == min camera look down angle
        if number_of_steps == 0:
            number_of_steps = 1

        # do raycast around character,
        # Assume that we have a invisible sphere around characters, the center is calculated character focus point
        # then pick raycast point uniformly on the surface of the sphere.
        for j in range(0, number_of_steps):
            angle_v_degree = j * raycast_angle_step + min_angle_in_degrees
            angle_v = angle_v_degree * math.pi / 180.0
            # calculate the height of the surface. We would intersect this surface with the sphere to get a circle.
            z = radius * np.sin(angle_v)
            # num_points (how many raycast point we would set on the interect between the sphere and
            # the calculated surface)
            num_points = int(360 / raycast_angle_step)
            calculated_radius = radius * np.cos(angle_v)
            max_height_distance = max_camera_distance
            min_height_distance = min_camera_distance

            if int(angle_v_degree) > 0:
                max_height_distance = (max_camera_height - people_focus_height) / np.sin(angle_v) - radius
                min_height_distance = (min_camera_height - people_focus_height) / np.sin(angle_v) - radius

                if min_height_distance > max_camera_distance:
                    continue
            elif angle_v_degree < 0:
                continue

            for i in range(num_points):
                angle_h = 2 * np.pi * i / num_points
                x = calculated_radius * np.cos(angle_h)
                y = calculated_radius * np.sin(angle_h)

                # the start point of the racst. Notice that a length of character radius has been added to
                # the center of the character
                # we need to make sure the raycast submitted from character does not hit it self
                cast_point = carb.Float3(center[0] + x, center[1] + y, people_focus_height + z)
                raycast.set_bvh_refresh_rate(omni.kit.mesh.raycast.BvhRefreshRate.FAST, True)

                # calculate the direction of the raycast
                raycast_dir = CarbUtil.normalize3(
                    carb.Float3(np.cos(angle_h) * np.cos(angle_v), np.sin(angle_h) * np.cos(angle_v), np.sin(angle_v))
                )
                ray_length = max_camera_distance + 1

                # get raycast hit result
                hit_result = raycast.closestRaycast(cast_point, raycast_dir, ray_length)
                mesh_index = 0
                position = [0, 0, 0]

                # get hitted mesh index from raycast
                if hit_result:
                    mesh_index = hit_result.meshIndex
                    position = hit_result.position
                else:
                    continue

                # when raycast hit nothing: it means that there are enough space for camera.
                if mesh_index == -1:
                    min_distance = max(min_height_distance, min_camera_distance)
                    max_distance = min(max_camera_distance, max_height_distance)

                    if min_distance > max_distance:
                        continue
                    # add raycast to valid raycast list
                    result.append(
                        RayCast_Container(character_name, cast_point, raycast_dir, min_distance, max_distance)
                    )
                else:

                    # get the distance between raycast dot and the closest object on that direction
                    hit_dist = CarbUtil.dist3(cast_point, carb.Float3(position[0], position[1], position[2]))
                    camera_dist = hit_dist - 0.7
                    calulated_camera_pos = CarbUtil.add3(cast_point, CarbUtil.scale3(raycast_dir, hit_dist))

                    # check whether there are enough space for cameraa
                    if camera_dist >= min_camera_distance and calulated_camera_pos[2] >= min_camera_height:
                        min_distance = max(min_height_distance, min_camera_distance)
                        max_distance = min(max_camera_distance, camera_dist, max_height_distance)
                        if min_distance > max_distance:
                            continue
                        # add raycast to valid raycast list
                        result.append(
                            RayCast_Container(character_name, cast_point, raycast_dir, min_distance, max_distance)
                        )
        return result

    # for every character in characte_list, get the raycast hit result from different direction.
    @staticmethod
    def get_character_raycast_check(character_list, raycast_angle_step=5):
        result = []
        for character in character_list:  # noqa
            # get character's radius and focus point( where the camera should looks at)
            radius, center = RandomizerUtil.get_character_radius_and_center(character)
            # generate raycast around character. Check for valid camera position.
            raycast_list = RandomizerUtil.get_character_surronding_raycast(
                character, radius, center, raycast_angle_step
            )
            if raycast_list and len(raycast_list) > 0:
                result.extend(raycast_list)
        return result

    # check whether certain character can be seen from certain position
    @staticmethod
    def check_character_visible_in_pos(character_prim_path, spawn_location):
        _, pos = RandomizerUtil.get_character_radius_and_center(character_prim_path)
        raycast = omni.kit.mesh.raycast.get_mesh_raycast_interface()
        raycast.set_bvh_refresh_rate(omni.kit.mesh.raycast.BvhRefreshRate.FAST, True)
        pos_list = []
        # do raycast from camera position to characters's center to head.
        for i in range(0, 7):
            pos_list.append([pos[0], pos[1], pos[2] + 0.1 * i])
        for dot in pos_list:
            dist = CarbUtil.dist3(spawn_location, dot)
            raycast_dir = CarbUtil.normalize3(CarbUtil.sub3(dot, spawn_location))
            hit_result = raycast.closestRaycast(spawn_location, raycast_dir, dist)

            # get hitted prim and check whether that is a character
            hit_prim_path = raycast.get_mesh_path_from_index(hit_result.meshIndex)
            if RandomizerUtil.is_character(hit_prim_path):
                return True, pos
        return False, None

    # rotate the camera to look at the characters
    @staticmethod
    def get_camera_rotation(char_path, spawn_location):
        character_height = RandomizerUtil.get_character_focus_height()
        character_in_camera, focus_point = RandomizerUtil.check_character_visible_in_pos(char_path, spawn_location)
        if character_in_camera:
            # get the rotation as quatf
            return Gf.Quatd(
                lookat_to_quatf(
                    Gf.Vec3d(focus_point[0], focus_point[1], character_height),
                    Gf.Vec3d(spawn_location[0], spawn_location[1], spawn_location[2]),
                    Gf.Vec3d(0, 0, 1),
                )
            )
        return None

    # convert INavigation's area index to navmesh's area index, in case not all areas are used in the navmesh
    @staticmethod
    def inav_idx_to_navmesh(inav, navmesh, area_idx):
        area_count = navmesh.get_area_count()
        all_area_names = inav.get_area_names()
        area_names_to_spawn = []
        # filter all_area_names based on spawn_area_idx
        if area_idx is not None:
            if isinstance(area_idx, (list, np.ndarray)):
                for i, area_name in enumerate(all_area_names):
                    if i in area_idx:
                        area_names_to_spawn.append(area_name)
        # get area names for current navmesh
        navmesh_area_names = []
        for i in range(area_count):
            navmesh_area_names.append(navmesh.get_area_name(i))
        area_idx_converted = []
        for area_name in area_names_to_spawn:
            if area_name in navmesh_area_names:
                area_idx_converted.append(navmesh_area_names.index(area_name))
            else:
                carb.log_warn(f"Area name {area_name} is not in navmesh. It will be ignored.")
        return area_idx_converted

    @staticmethod
    def area_idx_to_probability(navmesh, area_idx):
        # determian area probabilities based on spawn_area_idx
        area_count = navmesh.get_area_count()
        area_probabilities = np.zeros(area_count, dtype=np.float32)

        # set area probabilities based on converted spawn_area_idx
        for i in range(area_count):
            if i in area_idx:
                area_probabilities[i] = 1.0

        # Check if area_probabilities is all zeros, fallback to uniform probabilities
        if np.all(area_probabilities == 0):
            carb.log_verbose("No valid spawn area provided. Setting uniform probabilities for all areas.")
            # Set uniform probabilities for all areas
            area_probabilities = np.ones(area_count, dtype=np.float32)

        return area_probabilities

    @staticmethod
    def area_name_to_index(navmesh, area_name):
        area_count = navmesh.get_area_count()
        area_idx = []

        # Convert single string to list for consistent handling
        if isinstance(area_name, str):
            area_name = [area_name]

        for i in range(area_count):
            current_area_name = navmesh.get_area_name(i)
            if current_area_name in area_name:
                area_idx.append(i)

        return area_idx
