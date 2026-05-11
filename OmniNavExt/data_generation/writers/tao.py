__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import carb
from OmniNavExt.data_generation.annotator_data_processor import AnnotatorDataProcessor
from OmniNavExt.data_generation.object_info_manager import AgentInfo, ObjectInfo
from OmniNavExt.envset.settings import WriterSetting
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, WriterRegistry
from omni.replicator.core.scripts.writers_default.tools import *

from .writer import IRABasicWriter
from .writer_utils import WriterUtils

# Procuring standard KITTI Labels for objects annotated in the KITTI-format
# The dictionary is ordered where label idx corresponds to semantic ID
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py


class TaoWriter(IRABasicWriter):
    """Writer outputting data in the Json format
        Development work to provide full support is ongoing.

    PostProcessed Annotations:
        bounding_box_2d_tight/loose, bounding_box_3d, skeleton_data

    Notes:
        Object Detection:
            Filter character base on character's occlusion (occlusion is estimated from the area ratio of tight / loose bounding boxes)
            Partial Support: occluded
    """

    def __init__(
        self,
        bbox: bool = True,  # output character's skeleton information and bounding box informaiton
        shoulder_height_ratio: float = 0.6,
        valid_width_unoccluded_threshold: float = 0.6,  # utilized to check character's visibility in height
        valid_height_unoccluded_threshold: float = 0.25,  # utilized to check character's visibliity in width
        *args,  # Additional positional arguments for the parent class
        **kwargs,  # Additional keyword arguments for the parent class
    ):
        """Create a KITTI Writer"""

        # Call the parent constructor to handle parent class parameters
        filtered_argus = kwargs

        # if user want to output object's bounding box information
        self.bbox = bbox
        if self.bbox:
            filtered_argus["object_info_bounding_box_2d_tight"] = True
            filtered_argus["object_info_bounding_box_2d_loose"] = True
            filtered_argus["object_info_bounding_box_3d"] = True

        super().__init__(*args, **kwargs)
        self.valid_width_unoccluded_threshold = valid_width_unoccluded_threshold
        self.valid_height_unoccluded_threshold = valid_height_unoccluded_threshold
        self.shoulder_height_ratio = shoulder_height_ratio
        self.format = "JSON"
        # skip first several frame
        self.skip_frames = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/skip_starting_frames"
        )
        # frame interval of writing new annotators.
        self.writer_interval = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/frame_write_interval"
        )
        # Track numbering of the next frame to be written.
        self._frame_counter = 0

    @classmethod
    def params_values(cls) -> dict:
        params_values_dict = {}
        # Gather self parameters
        params_values_dict.update(WriterUtils.inspect_writer_init(TaoWriter))
        # Gather parent parameters
        params_values_dict.update(IRABasicWriter.params_values())
        # Temp: hide some params from parent class until writer refactor 2 is done
        hidden_param_list = [
            "object_info_bounding_box_2d_tight",
            "object_info_bounding_box_2d_loose",
            "object_info_bounding_box_3d",
            # "agent_info_skeleton_data",
        ]
        for p in hidden_param_list:
            params_values_dict.pop(p)
        return params_values_dict

    @classmethod
    def tooltip(cls):
        return f"""
            TaoWriter
            - Generates 3d bbox, 2d bbox, sematic segmentation and rgb from cameras.
            - Follows TAO labeeling standards when generating 2d and 3d bboxes.
        """

    def postprocess_object_detection_annotator(self, object_info, camera_params):
        """post process the annotator_data to refine the format"""

        # check whether the object information is an agent info
        if isinstance(object_info, AgentInfo):

            x_min, y_min, x_max, y_max = (
                0,
                0,
                camera_params["renderProductResolution"][0],
                camera_params["renderProductResolution"][1],
            )

            # refine the format of skeleton annotator, remove redundant info:
            skeleton_annotator = object_info.get_annotator_info("skeleton_data")
            # if target object has skeleton annotator
            if skeleton_annotator:
                target_key_name = ["global_translations", "translations_2d", "skeleton_joints"]
                WriterUtils.filter_dict_by_keys(skeleton_annotator, target_key_name)
                joint_translation_2d = skeleton_annotator["translations_2d"]
                skeleton_num = len(joint_translation_2d)
                # add a list to indicate whether the joint is in the viewport
                skeleton_annotator["in_view"] = [True] * skeleton_num
                for i in range(skeleton_num):
                    x, y = joint_translation_2d[i][0], joint_translation_2d[i][1]
                    if not x_min <= x <= x_max and y_min <= y <= y_max:
                        skeleton_annotator["in_view"][i] = False
                        # if the joint is not in the 2d viewport, set the value to None
                        joint_translation_2d[i] = [None, None]

        elif isinstance(object_info, ObjectInfo):
            pass

    def object_detection_enabled(self):
        """check whether user want to output object detection data"""
        return self.bbox

    def is_valid_info(self, object_info, camera_params):
        """check whether object info is valid"""
        # check whether object is an agent info
        if isinstance(object_info, AgentInfo):
            # check whether character is valid, and preprocess the boundary information
            return AnnotatorDataProcessor.validate_and_process_character(
                agent_info=object_info,
                camera_params=camera_params,
                width_threshold=self.valid_width_unoccluded_threshold,
                height_threshold=self.valid_height_unoccluded_threshold,
                shoulder_height_ratio=self.shoulder_height_ratio,
            )
        # if the object is an object info
        elif isinstance(object_info, ObjectInfo):
            # check whether the object's 3d bounding box is validate
            return AnnotatorDataProcessor.validate_and_process_object(
                object_info=object_info,
                camera_params=camera_params,
                width_threshold=self.valid_width_unoccluded_threshold,
                height_threshold=self.valid_height_unoccluded_threshold,
            )
        return False


WriterRegistry.register(TaoWriter)
