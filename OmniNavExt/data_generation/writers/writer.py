# from omni.replicator.core import Writer as ReplicatorWriter
import inspect
import json
import os

import carb
import omni.timeline
from OmniNavExt.data_generation.object_info_manager import AgentInfo, ObjectInfo, ObjectInfoManager
from OmniNavExt.envset.settings import WriterSetting
from omni.replicator.core import WriterRegistry
from omni.replicator.core.scripts import functional as F
from omni.replicator.core.scripts.writers_default.basicwriter import BasicWriter

from .writer_utils import WriterUtils


class IRABasicWriter(BasicWriter):
    """
    Base Writer class built on top of Replicator Writer to provide additional info needed in this extension.
    Inherient from this class if you want to add your writer as a built-in IRA writer.
    """

    @classmethod
    def params_values(cls) -> dict:
        """
        Parameters to expose in this extension (in config file format, UI) and their default values.
        It follows the signature of the class init function.
        """
        param_value_dict = {}
        # Gather child class parameters
        if cls != IRABasicWriter:
            param_value_dict.update(WriterUtils.inspect_writer_init(cls))
        # Gather self parameters
        param_value_dict.update(WriterUtils.inspect_writer_init(IRABasicWriter))
        # Hide some values that's seldom used and hard to display
        hidden_param_list = ["renderproduct_idxs", "target_object_labels", "output_objects"]
        for p in hidden_param_list:
            param_value_dict.pop(p)
        # Gather BasicWriter parameters
        param_value_dict.update(IRABasicWriter.basic_params_values())
        return param_value_dict

    @classmethod
    def basic_params_values(cls) -> dict:
        param_value_dict = {}
        # Since we don't own BasicWriter, we define its parm value we care here.
        # This way we can control its order and default values
        param_value_dict["output_dir"] = WriterSetting.get_writer_default_output_path()
        param_value_dict["semantic_filter_predicate"] = "class:character|robot;id:*"
        # param_value_dict["semantic_types"] = None # Not recommended to use it anymore
        param_value_dict["image_output_format"] = "png"
        param_value_dict["rgb"] = True
        # Override: disable camera_params output.
        # param_value_dict["camera_params"] = True
        param_value_dict["camera_params"] = False
        # param_value_dict["bounding_box_2d_tight"] = False # Temporary disable them since it is overwritten by "object_info_bounding_box_2d_tight"
        # param_value_dict["bounding_box_2d_loose"] = False
        # param_value_dict["bounding_box_3d"] = False
        param_value_dict["semantic_segmentation"] = False
        param_value_dict["instance_id_segmentation"] = False
        param_value_dict["instance_segmentation"] = False
        param_value_dict["distance_to_camera"] = False
        param_value_dict["distance_to_image_plane"] = False
        param_value_dict["colorize_depth"] = False
        param_value_dict["colorize_semantic_segmentation"] = True
        param_value_dict["colorize_instance_id_segmentation"] = True
        param_value_dict["colorize_instance_segmentation"] = True
        param_value_dict["occlusion"] = False
        param_value_dict["normals"] = False
        param_value_dict["motion_vectors"] = False
        param_value_dict["pointcloud"] = False
        param_value_dict["pointcloud_include_unlabelled"] = False
        # param_value_dict["skeleton_data"] = False         # Temporary disable, overwritten by "agent_info_skeleton_data"
        param_value_dict["use_common_output_dir"] = False

        return param_value_dict

    @classmethod
    def allow_basic_writer_params(cls) -> bool:
        """
        Mark if this writer is competible with BasicWriter parameters
        """
        return True

    @classmethod
    def tooltip(cls) -> str:
        """
        The tooltip to display in UI.
        """
        return ""

    @classmethod
    def extract_object_detection_annotator(cls, param_name: str):
        """extract annotator and seperate prefix"""
        # compose annotator prefix list for all annotators that need to be handled by write_object_detection
        annotator_classification_prefix = [
            WriterSetting.AnnotatorPrefix.ObjectDetection.AGENT_SPECIFIC,
            WriterSetting.AnnotatorPrefix.ObjectDetection.GENERIC,
            WriterSetting.AnnotatorPrefix.Others.CUSTOMIZED,
        ]
        classification, annotator_name = WriterUtils.extract_prefix_and_remainder(
            param_name, annotator_classification_prefix
        )
        return classification, annotator_name

    @classmethod
    def get_params_as_dict(cls, params):
        """Remove 'self' from the dictionary"""
        params.pop("self", None)
        return params

    @classmethod
    def get_all_write_functions(cls):
        """match all '_write" functions' name with target annotators"""
        write_function_prefix = ["_write"]
        annotator_to_function_dict = {}
        for function_name, obj in inspect.getmembers(cls):
            prefix, annotator = WriterUtils.extract_prefix_and_remainder(function_name, write_function_prefix)
            if prefix is not None and annotator:
                annotator_to_function_dict[annotator] = str(function_name)
        return annotator_to_function_dict

    def __init__(
        self,
        output_objects: bool = True,
        # Sample Customized Annotator List
        object_info_bounding_box_2d_tight: bool = False,
        object_info_bounding_box_2d_loose: bool = False,
        object_info_bounding_box_3d: bool = False,
        agent_info_skeleton_data: bool = False,
        renderproduct_idxs: list[tuple] = None,
        target_object_labels: list[str] = None,
        *args,
        **kwargs,
    ):
        # record all user input
        self.all_input_parameters = locals()
        self.object_detection_annotator_classification = {
            WriterSetting.AnnotatorPrefix.ObjectDetection.AGENT_SPECIFIC: [],
            WriterSetting.AnnotatorPrefix.ObjectDetection.GENERIC: [],
        }
        self.customized_annotator_list = []
        filtered_kwargs = self.initialize_writer(**kwargs)
        carb.log_info("Filtered Parameters: {filtered_kwargs}".format(filtered_kwargs=filtered_kwargs))
        # Call BasicWriter's constructor
        super().__init__(*args, **filtered_kwargs)
        # Whether output objects information
        self.output_objects = True  # Temp: always turn on output_objects
        self.data_structure = "renderProduct"
        self._render_product_idxs = renderproduct_idxs
        # Object info manager would be the object that manages all the data format
        # Temporary turn off target to objects
        self.target_object_labels = None
        self.object_info_manager = ObjectInfoManager(
            output_objects=self.output_objects,
            annotator_classification=self.object_detection_annotator_classification,
            target_object_labels=self.target_object_labels,
        )
        # The default output format of this writer is JSON
        self.output_format = "JSON"
        self.annotator_to_function_dict = self.get_all_write_functions()

        self._frame_counter = 0  # PLACEHOLDER ::frame counter to count current frame number
        self.skip_frames = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/skip_starting_frames"
        )
        self.writer_interval = 1  # PLACEHOLDER :: write interval

    def initialize_writer(self, **kwargs):
        """
        Initialize the writer by filtering parameters, classifying annotators, and
        preparing arguments for the superclass constructor.
        """
        # filter all the input, extract basicwriter's input
        b_init_params = inspect.signature(BasicWriter.__init__).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in b_init_params}
        # Hanlde customized annotator input defined by users.
        params_dict = self.get_params_as_dict(self.all_input_parameters)
        # union all parameters in a dict
        combined_params = {**params_dict, **kwargs}
        for params_name, param_value in combined_params.items():
            classification, annotator = self.extract_object_detection_annotator(params_name)
            if classification:
                if annotator not in b_init_params:
                    carb.log_info(
                        "Warning as message :: Object Detection Annotator [{annotator_name}] is not in default annotator list. ".format(
                            annotator_name=annotator
                        )
                    )
                    continue
                carb.log_info(
                    f"Object Detection Annotator: annotator name: {annotator}, classification: {classification}, value: {param_value}"
                )
                filtered_kwargs[annotator] = param_value
                # if classification belong to the object detection annotator classification
                if classification in self.object_detection_annotator_classification:
                    self.object_detection_annotator_classification[classification].append(annotator)
                # if the classification belongs to our customize annotator categories.
                if classification is not None:
                    self.customized_annotator_list.append(annotator)

        return filtered_kwargs

    def get_customized_annotators(self):
        """get a set of detection need to be handled by object detection method"""
        result = set(self.customized_annotator_list)
        return result

    def write(self, data):
        """override the writer function from basic writer"""
        timeline = omni.timeline.get_timeline_interface()
        if self.skip_frames > 0:
            self.skip_frames -= 1
            return
        # check current frame_counter with the writer interval.
        if self._frame_counter % self.writer_interval != 0:
            self._frame_counter += 1
        else:
            self._write_all_sensor_datas(data)
            self._frame_id += 1
            self._frame_counter += 1

    def _write_all_sensor_datas(self, data: dict):
        """
        For annotators included in object detection list:  output data in custom manner.
        For other annotators : try to call pre-defined output function.
        """
        # this time output the key list only:
        # output the key of the data renderproduct

        render_product = dict(data["renderProducts"])
        self.object_info_manager.refresh_info(render_product)

        for key, annotator_dict in render_product.items():
            # get current camera_path
            self._write_sensor_data(annotator_dict=annotator_dict)

    def _write_sensor_data(self, annotator_dict: dict):
        """write annotator data related to single sensor"""
        camera_path = annotator_dict["camera"]
        # get current activated annotator list:
        all_annotators = annotator_dict.keys()
        # get all customized annotator list:
        customized_annotator_set = self.get_customized_annotators()
        # get default annotators:
        default_annotators = [
            item
            for item in all_annotators
            if not any(str(item).startswith(prefix) for prefix in customized_annotator_set)
        ]

        camera_id = str(camera_path).replace("/", "_")

        for annotator_name in default_annotators:

            output_path = os.path.join(camera_id, annotator_name) + os.path.sep
            annotator_data = annotator_dict[annotator_name]

            for annotator_type, output_function_name in self.annotator_to_function_dict.items():
                if str(annotator_name).startswith(annotator_type):
                    # check whether the output function has been implemented
                    if hasattr(self, output_function_name):
                        default_output_function = getattr(self, output_function_name)
                        default_output_function(annotator_data, output_path)

                    break
        # whether user want to write object detection
        if self.object_detection_enabled():
            # collected all information included by this camera into a json file.
            self.write_object_detection(camera_path=camera_path, sub_dir=camera_id)
        self.customized_post_process(annotator_dict=annotator_dict, sub_dir=camera_id)

    def object_detection_enabled(self):
        """place holder, whether object detection data would output"""
        return True

    def customized_post_process(self, annotator_dict, sub_dir: str):
        """place holder to handle customized post process data"""
        pass

    def is_valid_info(self, object_info, camera_params):
        """check whether the target object is valid"""
        return True

    def postprocess_object_detection_annotator(self, object_info, camera_params):
        """place holder to post process the objecct detection data"""
        pass

    def write_object_detection(self, camera_path: str, sub_dir: str):
        """Add filters to check agent info's visibility"""
        info_dict = {"agents": {}, "objects": {}}
        # fetch object detection info base on
        object_detection_info = self.object_info_manager.get_camera_view_info(camera_path)

        if object_detection_info:
            # Collect agent information if available:
            agent_info_dict = object_detection_info.get_agent_info_dict()
            # collect camera information
            camera_params = object_detection_info.get_camera_params()
            if agent_info_dict:
                for prim_path, agent_info in agent_info_dict.items():
                    # check whether agent is valid
                    if not self.is_valid_info(agent_info, camera_params):
                        continue
                    self.postprocess_object_detection_annotator(object_info=agent_info, camera_params=camera_params)
                    info_dict["agents"][prim_path] = agent_info.get_all_info()

            if self.output_objects:
                object_info_dict = object_detection_info.get_object_info_dict()
                for prim_path, object_info in object_info_dict.items():
                    # check whether object is valid
                    if not self.is_valid_info(object_info, camera_params):
                        continue
                    self.postprocess_object_detection_annotator(object_info=object_info, camera_params=camera_params)
                    info_dict["objects"][prim_path] = object_info.get_all_info()

        # Construct the output file path
        output_filepath = os.path.join(
            sub_dir,
            "object_detection",
            f"object_detection_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json",
        )

        # Schedule the backend write operation, helper method "numpy encoder" is imported to let
        self._backend.schedule(
            F.write_json, data=info_dict, path=output_filepath, indent=4, default=WriterUtils.numpy_encoder
        )


WriterRegistry.register(IRABasicWriter)
