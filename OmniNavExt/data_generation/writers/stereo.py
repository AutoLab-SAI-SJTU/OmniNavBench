__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import io
import os
import sys
from typing import Any

import carb
import numpy as np
import omni.usd
from OmniNavExt.data_generation.annotator_data_processor import AnnotatorDataProcessor
from OmniNavExt.data_generation.object_info_manager import AgentInfo, ObjectInfo
from OmniNavExt.envset.settings import PrimPaths, WriterSetting
from OmniNavExt.envset.stage_util import StereoCamUtil
from omni.metropolis.utils.sensor_util import SensorUtil
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts import functional as F

from .writer import IRABasicWriter
from .writer_utils import WriterUtils


class Depth_File_Type:
    IMAGE = ("PNG",)
    PFM = "PFM"


__version__ = "0.0.2"


class StereoWriter(IRABasicWriter):
    """Writer outputting data in json format
        Development work to provide full support is ongoing.

    PostProcessed Annotations:
        Object Detection (2d,3d bounding box information)
        Depth (allow pmf format storage)

    Notes:
        Object Detection
        Bounding boxes with a height smaller than 25 pixels are discarded

        Supported: bounding box extents, semantic labels
        Partial Support: occluded (occlusion is estimated from the area ratio of tight / loose bounding boxes)
        Unsupported: alpha, dimensions, location, rotation_y, truncated (all set to default values of 0.0)
    """

    def __init__(
        self,
        bbox: bool = True,
        customized_distance_to_image_plane: bool = True,
        customized_camera_params=True,
        valid_width_unoccluded_threshold: float = 0.6,  # utilized to check character's visibility in height
        valid_height_unoccluded_threshold: float = 0.6,  # utilized to check character's visibliity in width
        shoulder_height_ratio: float = 0.25,
        depth_format: str = "PNG",
        *args,  # Additional positional arguments for the parent class
        **kwargs,  # Additional keyword arguments for the parent class
    ):
        """Create a JSON Writer

        Args:
            output_dir: Output directory to which the Annotator Data would be write.
            depth_format: The format of depth data
            bbox: Output object detection data
        """
        # Call the parent constructor to handle parent class parameters
        filtered_argus = kwargs

        self.depth_format = depth_format
        if self.depth_format == "PNG":
            # convert the depth output format from default npy to png
            filtered_argus["colorize_depth"] = True

        # if user want to output object's bounding box information
        self.bbox = bbox
        if self.bbox:
            filtered_argus["object_info_bounding_box_2d_tight"] = True
            filtered_argus["object_info_bounding_box_2d_loose"] = True
            filtered_argus["object_info_bounding_box_3d"] = True
        # if user want to output
        filtered_argus["customized_distance_to_image_plane"] = customized_distance_to_image_plane
        filtered_argus["customized_camera_params"] = customized_camera_params
        filtered_argus["output_objects"] = True

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
        params_values_dict.update(WriterUtils.inspect_writer_init(StereoWriter))
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

    @staticmethod
    def tooltip():
        return f"""
            StereoWriter
            - Generate Paired Stereo Camera Data:
            - Generates 3d bbox, 2d bbox and rgb from cameras.
        """

    def _write_all_sensor_datas(self, data: dict):
        """write all sensor datas"""
        render_product = dict(data["renderProducts"])
        self.object_info_manager.refresh_info(render_product)
        for key, annotator_dict in render_product.items():
            camera_path = annotator_dict["camera"]
            camera_type = self.check_sensor_type(camera_path)
            if camera_type == WriterSetting.SensorType.Camera:
                super()._write_sensor_data(annotator_dict=annotator_dict)

        pass

    def customized_post_process(self, annotator_dict: dict, sub_dir: str):
        """customized postprocess for target annotators"""
        if "distance_to_image_plane" in annotator_dict:
            self._write_distance_to_image_plane(annotator_dict, sub_dir)
        if "camera_params" in annotator_dict:
            self._write_camera_params(annotator_dict, sub_dir)
        pass

    def check_sensor_type(self, target_camera_path: str):
        """check sensor type from prim path"""
        camera_parent_path = PrimPaths.cameras_parent_path()
        lidar_parent_path = PrimPaths.lidar_cameras_parent_path()
        if target_camera_path.startswith(camera_parent_path):
            return WriterSetting.SensorType.Camera
        if target_camera_path.startswith(lidar_parent_path):
            return WriterSetting.SensorType.Lidar

        return WriterSetting.SensorType.Unknown

    def write_pfm(
        self, sub_dir: str, data: np.ndarray, scale: float = 1, file_identifier=b"Pf", dtype="float32"
    ) -> None:
        """
        Write float data to PFM file.

        Args:
            path: Write path URI
            data: Data to write as a Numpy array.
            backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
            scale: Scale factor to write in the PFM header. Can be used to handle endianess.
        """
        # file = os.path.join(self.output_dir, path)
        # image = data
        data = np.flipud(np.nan_to_num(data, nan=0.0, posinf=0.0))
        height, width = np.shape(data)[:2]
        values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
        endianess = data.dtype.byteorder
        if endianess == "<" or (endianess == "=" and sys.byteorder == "little"):
            scale *= -1

        buf = io.BytesIO()
        buf.write((file_identifier))
        buf.write(("\n%d %d\n" % (width, height)).encode())
        buf.write(("%d\n" % scale).encode())
        buf.write(values)
        file_path = os.path.join(
            sub_dir, "depth", f"distance_to_image_plane_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.pfm"
        )
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_distance_to_image_plane(self, annotator_dict: dict, sub_dir: str):
        # data, sub_dir: str, annotator: str
        annotator_name = "distance_to_image_plane"
        depth_annotator = annotator_dict[annotator_name]

        if self.depth_format == "PNG" or self.depth_format == "NPY":
            output_path = os.path.join(sub_dir, annotator_name) + os.path.sep
            # use the default write distance with default method
            super()._write_distance_to_image_plane(depth_annotator, output_path)
            return
        else:
            # define the method to save the file as a PFM File
            self.write_pfm(sub_dir=sub_dir, data=depth_annotator)

    def get_stereo_camera_data(self, target_camera_path):
        paired_camera_path = StereoCamUtil.get_paired_stereo_camera_path(target_camera_path)
        if not self.object_info_manager.is_camera_exist(paired_camera_path):
            carb.log_info("camera path " + str(target_camera_path) + "has no paired camera")
            return None, None
        target_camera_position = self.get_camera_position(target_camera_path)
        paired_camera_position = self.get_camera_position(paired_camera_path)

        if target_camera_position is None or paired_camera_position is None:
            carb.log_warn(
                "target_camera {target_camera_path} position: {target_camera_position} paired camera {paired_camera_path} paired camera position {paired_camera_position}".format(
                    target_camera_path=target_camera_path,
                    target_camera_position=str(target_camera_position),
                    paired_camera_path=paired_camera_path,
                    paired_camera_position=str(paired_camera_position),
                )
            )
            return None, None
        stereo_baseline = WriterUtils.calculate_distance(target_camera_position, paired_camera_position)
        camera_height = target_camera_position[2]
        return stereo_baseline, camera_height

    def get_camera_position(self, target_camera_path):
        """get camera position in the stage"""
        camera_view_info = self.object_info_manager.get_camera_view_info(target_camera_path)
        if camera_view_info is None:
            carb.log_warn(
                "{camera_path} do not have matching camera view information".format(camera_path=target_camera_path)
            )
            carb.log_info("current available camera" + str(self.object_info_manager.path_to_camera_view_info.keys()))
            return None
        camera_params_data = camera_view_info.get_camera_params()
        camera_transform = np.linalg.inv(np.reshape(camera_params_data["cameraViewTransform"], (4, 4)))
        camera_position = [camera_transform[3][0], camera_transform[3][1], camera_transform[3][2]]
        return camera_position

    def _write_camera_params(self, annotator_dict, sub_dir):
        """output camera information"""
        # output camera information to target path
        camera_info = annotator_dict["camera_params"]
        rp_width, rp_height = camera_info["renderProductResolution"][0], camera_info["renderProductResolution"][1]
        camera_view = np.reshape(camera_info["cameraViewTransform"], (4, 4))
        camera_intrinsics = SensorUtil.get_camera_intrinsic_dict(camera_params=camera_info)

        label_set = {}
        split_list = sub_dir.split("_")
        if split_list[-1] == "Camera":
            camera_id = "_"
        else:
            camera_id = split_list[-1]

        # output camera's projection matrix
        label_set["camera_projection"] = np.reshape(camera_info["cameraProjection"], (4, 4)).flatten().tolist()
        # output resolution information in target format
        label_set["rp_height"] = rp_height
        label_set["rp_width"] = rp_width
        # output camera's intrinsic information
        intrinsic = [
            camera_intrinsics["fx"],
            camera_intrinsics["fy"],
            camera_intrinsics["cx"],
            camera_intrinsics["cy"],
        ]
        # get intrinsic matrix with target format
        label_set["fx_fy_cx_cy"] = intrinsic
        label_set["focallength"] = camera_info["cameraFocalLength"]
        label_set["camera_transform"] = np.linalg.inv(np.reshape(camera_info["cameraViewTransform"], (4, 4)))
        camera_path = annotator_dict["camera"]
        stereo_baseline, camera_height = self.get_stereo_camera_data(camera_path)
        label_set["stereo_baseline"] = stereo_baseline
        label_set["camera_height"] = camera_height

        # get distance between two different camera#
        # get distance between two different camera#
        calibration_filepath = os.path.join(
            sub_dir, "calibration", f"camera_params_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        )
        self._backend.schedule(
            F.write_json, data=label_set, path=calibration_filepath, indent=4, default=WriterUtils.numpy_encoder
        )

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
            # check whether object's 3d bounding box info is validate
            return AnnotatorDataProcessor.validate_and_process_object(
                object_info=object_info,
                camera_params=camera_params,
                width_threshold=self.valid_width_unoccluded_threshold,
                height_threshold=self.valid_height_unoccluded_threshold,
            )

        return False

    def object_detection_enabled(self):
        """check whether user want to output object detection data"""
        return self.bbox

    def postprocess_object_detection_annotator(self, object_info, camera_params):
        """post process the annotator_data to refine the format"""
        # check whether object is an ObjectInfo
        if isinstance(object_info, ObjectInfo):
            bbox_3d_annotator = object_info.get_annotator_info("bounding_box_3d_fast")
            if bbox_3d_annotator:
                # convert the world transform to object's view space transforma
                bbox_3d_transform = bbox_3d_annotator["transform"]
                camera_view_matrix = np.reshape(camera_params["cameraViewTransform"], (4, 4))
                bbox_3d_annotator["camera_view_transform"] = AnnotatorDataProcessor.convert_to_camera_space(
                    world_transform=bbox_3d_transform, camera_view_matrix=camera_view_matrix
                )

        # check whether the object is an agent info
        if isinstance(object_info, AgentInfo):
            # extract current view port boundary in 2d image
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


WriterRegistry.register(StereoWriter)
