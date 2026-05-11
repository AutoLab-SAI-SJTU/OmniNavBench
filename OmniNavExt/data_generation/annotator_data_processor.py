# container used to store skeleton information
from typing import Any

import numpy as np
from OmniNavExt.data_generation.object_info_manager import AgentInfo, ObjectInfo
from OmniNavExt.data_generation.writers.writer_utils import WriterUtils
from omni.metropolis.utils.sensor_util import SensorUtil
from omni.syntheticdata.scripts import helpers, sensors
from pxr import Gf, Semantics, Usd, UsdGeom, UsdSkel
from typing import Optional, Callable, List
from functools import partial

from OmniNavExt.envset.settings import PrimPaths, WriterSetting


class AnnotatorDataProcessor:
    @staticmethod
    def evaluate_box_within_viewport(character_box, viewport_box):
        """check whether character 2d bounding box is within image, trancated with image boundary or out of viewport"""

        def is_within_tolerance(a, b, tolerance=4):
            """helper method, check whether character is on the boundary"""
            return abs(a - b) <= tolerance

        is_within_bound, adjusted_box = AnnotatorDataProcessor.check_bounding_box_relationship(
            character_box, viewport_box, is_within_tolerance
        )
        return is_within_bound, adjusted_box

    @staticmethod
    def numpy_to_dictionary(target_numpy: np.void, target_key_list: list | None = None):
        """convert a numpy.void type to dictionary"""
        if target_key_list is None:
            return {name: target_numpy[name] for name in target_numpy.dtype.names}
        else:
            return {name: target_numpy[name] for name in target_numpy.dtype.names if str(name) in target_key_list}

    @staticmethod
    def get_scale(bbox_3d_data):
        """calculate bbox 3d's scale information"""
        x_min = bbox_3d_data["x_min"]
        x_max = bbox_3d_data["x_max"]
        y_min = bbox_3d_data["y_min"]
        y_max = bbox_3d_data["y_max"]
        z_min = bbox_3d_data["z_min"]
        z_max = bbox_3d_data["z_max"]

        scale_x = x_max - x_min
        scale_y = y_max - y_min
        scale_z = z_max - z_min

        return (scale_x, scale_y, scale_z)

    @staticmethod
    def calculate_2d_bounding_box(points):
        """
        calculate the bounding box that contains target 2d points set
        """
        # use numpy 's min/max func to calculate the boundary that contains all points
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        return x_min, y_min, x_max, y_max

    @staticmethod
    def convert_to_camera_space(world_transform, camera_view_matrix):
        return np.matmul(world_transform, camera_view_matrix)

    @staticmethod
    def check_bounding_box_relationship(inner_box, outer_box, comparison_fn):
        """Check the relationship between two bounding boxes."""

        # Adjusting to the new input order: (x_min, y_min, x_max, y_max)
        inner_x_min, inner_y_min, inner_x_max, inner_y_max = inner_box
        outer_x_min, outer_y_min, outer_x_max, outer_y_max = outer_box

        # Check if the inner box is completely out of bounds of the outer box
        if (
            comparison_fn(outer_x_min, inner_x_max)
            or comparison_fn(inner_x_min, outer_x_max)
            or comparison_fn(outer_y_min, inner_y_max)
            or comparison_fn(inner_y_min, outer_y_max)
        ):
            return WriterSetting.AgentStatus.OUTSIDE, [-1, -1, -1, -1]

        status = WriterSetting.AgentStatus.INSIDE

        # Adjust bounding box if it's truncated on the edges
        if inner_x_max > outer_x_max:
            status = WriterSetting.AgentStatus.TRUNCATED
            inner_x_max = outer_x_max

        if inner_x_min < outer_x_min:
            status = WriterSetting.AgentStatus.TRUNCATED
            inner_x_min = outer_x_min

        if inner_y_max > outer_y_max:
            status = WriterSetting.AgentStatus.TRUNCATED
            inner_y_max = outer_y_max

        if inner_y_min < outer_y_min:
            status = WriterSetting.AgentStatus.TRUNCATED
            inner_y_min = outer_y_min

        return status, (inner_x_min, inner_y_min, inner_x_max, inner_y_max)

    @staticmethod
    def extract_2d_box_information(bbox_data):
        """extract 2d box informatin as tuple"""
        return (bbox_data["x_min"], bbox_data["y_min"], bbox_data["x_max"], bbox_data["y_max"])

    @staticmethod
    def model_to_world(pt, obj_xform, camera_xform, proj_mat, screen_width, screen_height):
        """generate the 3d space point and 2d projection of each point on the cuboid vertex and the central point"""
        model_space = np.array([pt[0], pt[1], pt[2], 1])
        # calculate the world space position by cross product those two
        world_space = model_space @ obj_xform

        camera_space = world_space @ camera_xform
        ndc_space = camera_space @ proj_mat
        ndc_space /= ndc_space[3]
        x = (1 + ndc_space[0]) / 2 * screen_width
        y = (1 - ndc_space[1]) / 2 * screen_height
        return (int(x), int(y)), world_space

    @staticmethod
    def extent_dimension(extents):
        """extend the demension of input data to fit the format of helper method parameter"""
        return {
            "x_min": np.expand_dims(extents["x_min"], axis=0),
            "x_max": np.expand_dims(extents["x_max"], axis=0),
            "y_min": np.expand_dims(extents["y_min"], axis=0),
            "y_max": np.expand_dims(extents["y_max"], axis=0),
            "z_min": np.expand_dims(extents["z_min"], axis=0),
            "z_max": np.expand_dims(extents["z_max"], axis=0),
            "transform": np.expand_dims(extents["transform"], axis=0),
        }

    def validate_and_process_bbox_3d_data(object_info: ObjectInfo, camera_params: Any):
        """post process the bounding box 3d data"""
        if True:

            bbox_3d_annotator_name = "bounding_box_3d_fast"
            bbox_data = object_info.get_annotator_info(bbox_3d_annotator_name)

            if not bbox_data:
                return False
            # reformat the view params
            view_params = SensorUtil.reformat_camera_params(camera_params)
            # post process the 3d bounding box information
            corners = helpers.get_bbox_3d_corners(AnnotatorDataProcessor.extent_dimension(bbox_data))
            # if the number of vertex is not 8, pause post process
            if corners.shape[1] != 8:
                # carb.log_info("corner vertex is not 8. discard the info")
                return False

            # convert the bounding box to dictionary to include new values.
            processed_data = AnnotatorDataProcessor.numpy_to_dictionary(bbox_data)
            processed_data["scale"] = AnnotatorDataProcessor.get_scale(bbox_data)
            processed_data["vertex"] = {}
            processed_data["vertex"]["translations_3d"] = corners
            processed_data["vertex"]["translations_2d"] = SensorUtil.world_to_image_helper(
                corners.reshape(-1, 3), view_params
            )
            object_info.update_annotator_data(bbox_3d_annotator_name, processed_data)

            return True

    def evaluate_height_and_width_threshold(
        loose_bounding_box,
        tight_bounding_box,
        width_threshold: float | None = None,
        height_threshold: float | None = None,
        special_check_fn: Callable | None = None,
    ):
        """check whether the height and width threshold meet requirements"""
        character_width = tight_bounding_box[2] - tight_bounding_box[0]
        character_height = tight_bounding_box[3] - tight_bounding_box[1]
        full_body_width = loose_bounding_box[2] - loose_bounding_box[0]
        full_body_height = loose_bounding_box[3] - loose_bounding_box[1]

        width_show_ratio = character_width / (full_body_width + WriterUtils.EPS)
        height_show_ratio = character_height / (full_body_height + WriterUtils.EPS)

        # check whether character meet the unocclusion ratio threshold in width/height
        if width_threshold is not None:
            width_threshold_check = width_show_ratio > width_threshold
        else:
            width_threshold_check = True

        if height_threshold is not None:
            height_threshold_check = height_show_ratio > height_threshold
        else:
            height_threshold_check = True

        result = [width_threshold_check, height_threshold_check]

        if special_check_fn is not None:
            special_check_fn(
                loose_bounding_box=loose_bounding_box, tight_bounding_box=tight_bounding_box, result=result
            )

        return result[0], result[1]

    def shoulder_head_check(loose_bounding_box, tight_bounding_box, shoulder_height_ratio: float, result: List):
        """special check function design for character"""
        # check whether the height threshold check already passed
        if not result[1]:
            # check whether the character's upper body has been shown.
            full_body_height = abs(loose_bounding_box[3] - loose_bounding_box[1])
            shoulder_y = loose_bounding_box[1] + shoulder_height_ratio * full_body_height
            if tight_bounding_box[1] < shoulder_y:
                shoulder_height_shown_ratio = (shoulder_y - tight_bounding_box[1]) / (
                    shoulder_height_ratio * full_body_height
                )
                # check whether shoulder occlusion ratio is less than threshold
                height_threshold_check = (
                    1 - shoulder_height_shown_ratio
                ) < WriterSetting.DefaultWriterConstant.SHOULDER_OCCLUSION_THRESHOLD
                result[1] = height_threshold_check

    def validate_and_process_object(
        object_info: ObjectInfo,
        camera_params: Any,
        width_threshold: float | None = None,
        height_threshold: float | None = None,
        threshold_check_fn: Callable | None = None,
        bbox_recalculate_fn: Callable | None = None,
    ) -> bool:
        """filter character with visibility threshold"""

        # if the input setting is none, use the default settings
        if width_threshold is None:
            width_threshold = WriterSetting.DefaultWriterConstant.WIDTH_THRESHOLD
        if height_threshold is None:
            height_threshold = WriterSetting.DefaultWriterConstant.HEIGHT_THRESHOLD
        # extract target annotator from "agent" info structure
        box_3d_information = object_info.get_annotator_info("bounding_box_3d_fast")
        tight_box_information = object_info.get_annotator_info("bounding_box_2d_tight_fast")
        loose_box_information = object_info.get_annotator_info("bounding_box_2d_loose_fast")

        # check whether all necessary character data is valid
        if not (box_3d_information and tight_box_information and loose_box_information):
            return False

        tight_box = AnnotatorDataProcessor.extract_2d_box_information(tight_box_information)
        loose_box = AnnotatorDataProcessor.extract_2d_box_information(loose_box_information)
        full_body_box = loose_box

        viewport_box = (0, 0, camera_params["renderProductResolution"][0], camera_params["renderProductResolution"][1])

        # check the relationship between this character and viewport boundary.
        agent_status, adjusted_box = AnnotatorDataProcessor.evaluate_box_within_viewport(tight_box, viewport_box)

        if agent_status == WriterSetting.AgentStatus.OUTSIDE:
            # carb.log_info("character out of the boundary")
            return False

        # fail to pos process the bounding box 3d data:
        if not AnnotatorDataProcessor.validate_and_process_bbox_3d_data(object_info, camera_params):
            # carb.log_info("fail to post process the 3d bounding box information")
            return False

        # if the character is trancate
        if agent_status == WriterSetting.AgentStatus.TRUNCATED:
            if bbox_recalculate_fn is not None:
                full_body_box = bbox_recalculate_fn()
            else:
                # if no special requirement, the full bounding box would be recalculated via the 3d bounding box's 2d vertex projection.
                point_2d = box_3d_information["vertex"]["translations_2d"]
                # calculate the full body bounding box with either skeleton annotation or box_3d information
                full_body_box = AnnotatorDataProcessor.calculate_2d_bounding_box(point_2d)

        # calculate character's visible_only/full_body 2d width/height
        width_threshold_check, height_threshold_check = AnnotatorDataProcessor.evaluate_height_and_width_threshold(
            loose_bounding_box=full_body_box,
            tight_bounding_box=adjusted_box,
            width_threshold=width_threshold,
            height_threshold=height_threshold,
            special_check_fn=threshold_check_fn,
        )

        # if the unocclusion ratio threshold in height is not meet, check whether character's head and shoulder is visible.
        return (
            (height_threshold_check and width_threshold_check)
            if agent_status == WriterSetting.AgentStatus.INSIDE
            else (height_threshold_check or width_threshold_check)
        )

    def validate_and_process_character(
        agent_info: AgentInfo,
        camera_params: Any,
        width_threshold: float | None = None,
        height_threshold: float | None = None,
        shoulder_height_ratio: float | None = None,
    ):
        """check whether the character's show ratio is valid, new settings such as shoulder height ratio are added"""

        if shoulder_height_ratio is None:
            shoulder_height_ratio = WriterSetting.DefaultWriterConstant.SHOULDER_HEIGHT_RATIO

        skeleton_information = agent_info.get_annotator_info("skeleton_data")
        bbox_recalulate_fn = None
        # check whether the skeleton annotator has been activated
        if skeleton_information is not None:
            point_2d = skeleton_information["translations_2d"]
            # calculate the character full body bbox from the skeleton joints' 2d projection
            bbox_recalulate_fn = partial(AnnotatorDataProcessor.calculate_2d_bounding_box, points=point_2d)

        threshold_check_fn = partial(
            AnnotatorDataProcessor.shoulder_head_check, shoulder_height_ratio=shoulder_height_ratio
        )
        return AnnotatorDataProcessor.validate_and_process_object(
            object_info=agent_info,
            camera_params=camera_params,
            width_threshold=width_threshold,
            height_threshold=height_threshold,
            threshold_check_fn=threshold_check_fn,
            bbox_recalculate_fn=bbox_recalulate_fn,
        )
