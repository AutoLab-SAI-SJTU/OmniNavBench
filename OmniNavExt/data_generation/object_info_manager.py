from typing import Any

import carb
import numpy as np
from OmniNavExt.envset.agent_manager import AgentManager

from OmniNavExt.envset.settings import WriterSetting
from .writers.writer_utils import WriterUtils


class ObjectInfo:
    """Store raw information related to target object"""

    def __init__(self, label: str = "", prim_path: str = "", annotators: dict[str, Any] | None = None):
        self.label: str = label  # object's semantic label
        self.prim_path: str = prim_path  # object's prim path
        if annotators is None:
            self.annotator_data = {}
        else:
            self.annotator_data = annotators

    def get_label(self) -> str:
        return self.label

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_all_annotator(self) -> dict[str:Any]:
        return self.annotator_data

    def update_annotator_data(self, annotator_name, data):
        """store annotator info base on annotator's name"""
        self.annotator_data[annotator_name] = data

    def get_all_info(self) -> dict[str, Any]:
        """helper method to get object information, return data into a dictionary"""
        return {"label": self.label, "annotators": self.annotator_data}

    def get_annotator_info(self, annotator_name: str):
        if annotator_name not in self.annotator_data:
            carb.log_info(
                " Warning as message:: Object Detection class: {class_name} does not handle: {annotator_name} annotator".format(
                    class_name=type(self).__name__, annotator_name=annotator_name
                )
            )
            return None
        else:
            return self.annotator_data[annotator_name]


class AgentInfo(ObjectInfo):
    """Store raw information related to target agent"""

    def __init__(self, label: str = "", prim_path: str = "", annotators: dict[str, Any] | None = None):
        super().__init__(label, prim_path, annotators)
        self.agent_attributes: dict[str, Any] = {}
        if annotators is None:
            self.annotator_data = {}
        else:
            self.annotator_data = annotators

    def get_all_info(self):
        info = super().get_all_info()
        return info


class ObjectDetectionInfo:
    """Store all Raw information within the target viewport"""

    def __init__(
        self,
        path: str = "",
        output_objects: bool = False,
        annotator_classification: dict[str, list[str]] = {},
        target_object_labels: list[str] = None,
    ):
        self.camera_prim_path: str = path
        self.camera_params: dict = {}
        # match prim data with prim path
        self.object_info_dict: dict[str, ObjectInfo] = {}
        # match agent data with agent path
        self.agent_info_dict: dict[str, AgentInfo] = {}
        # get access to agent manager
        self.agent_manager = AgentManager.get_instance()
        self.output_objects = output_objects
        self.annotator_classification = annotator_classification
        self.target_object_labels = target_object_labels
        carb.log_info("Current Objection Classification : " + str(self.annotator_classification))

    def get_object_info_dict(self) -> dict[str, ObjectInfo]:
        """return a dictionary that store all objects' annotator data"""
        return self.object_info_dict

    def get_agent_info_dict(self) -> dict[str, AgentInfo]:
        """return a dictionary that store all agents' annotator data"""
        return self.agent_info_dict

    def get_camera_params(self):
        """get camera params"""
        return self.camera_params

    def get_camera_position(self):
        """get camera pos"""
        camera_transform = np.linalg.inv(self.camera_params["cameraViewTransform"])
        camera_position = [camera_transform[3, 0], camera_transform[3, 1], camera_transform[3, 2]]
        return camera_position

    def clear_data(self):
        """clear all data captured by this camera view"""
        self.object_info_dict.clear()
        self.agent_info_dict.clear()
        self.camera_params.clear()

    def refresh_info(self, annotator_data_dict: dict[str, Any]):
        """refresh info stored in the dictionary"""
        # clean all object info dict
        self.clear_data()
        # refresh camera information
        self.refresh_camera_info(annotator_data_dict)
        # refresh object information
        self.refresh_object_info(annotator_data_dict)
        # refresh agent information
        self.refresh_agent_info(annotator_data_dict)

    def refresh_camera_info(self, annotator_data_dict: dict[str, Any]):
        """refresh camera info"""
        # extract data related to camera information:
        if not self.camera_prim_path:
            self.camera_prim_path = annotator_data_dict["camera"]
        self.camera_params = annotator_data_dict["camera_params"]

    def refresh_agent_info(self, annotator_data_dict):
        """refresh agent related information"""
        # Fetch all activated agent specific annotators
        annotator_data_key = annotator_data_dict.keys()
        agent_annotator_list = self.annotator_classification[
            WriterSetting.AnnotatorPrefix.ObjectDetection.AGENT_SPECIFIC
        ]
        agent_specific_annotators = WriterUtils.select_elements_based_on_prefix(
            annotator_data_key, agent_annotator_list
        )

        agent_specific_annotators_dict = {key: annotator_data_dict.get(key, None) for key in agent_specific_annotators}
        for annotator_name, annotator in agent_specific_annotators_dict.items():
            # if the annotator is skeleton data
            if annotator_name == "skeleton_data":
                # match the skeleton path from the annotator
                num_skeletons = annotator["numSkeletons"]
                skeleton_data_dict = WriterUtils.post_process_skeleton_datas(annotator)
                # iterate through the skeleton index
                for index in range(num_skeletons):
                    # compose the skeleton key utilized in the skeleton dict.
                    skeleton_id_str = "skeleton_{id}".format(id=index)
                    # get skeleton data matching with this character
                    skeleton_data = skeleton_data_dict[skeleton_id_str]
                    # get the skeleton path:
                    skeleton_path = skeleton_data["skel_path"]
                    # extract character prim path and character label
                    character_data = self.agent_manager.get_agent_data_by_skelpath(str(skeleton_path))
                    if not character_data:
                        return None

                    character_prim_path = character_data.prim_path
                    character_label = character_data.label
                    # if the agent info has not yet been recorded
                    if character_prim_path not in self.agent_info_dict.keys():
                        # initialize a new Agent Info to record the data
                        self.agent_info_dict[str(character_prim_path)] = AgentInfo(
                            label=character_label, prim_path=character_prim_path
                        )
                    # update the skeleton data to the agent info structure
                    self.agent_info_dict[str(character_prim_path)].update_annotator_data(annotator_name, skeleton_data)

    def refresh_object_info(self, annotator_data_dict: dict[str, Any]):
        """Take the latest annotator stream from replicator and refresh the object info dict."""

        # Fetch all activated generic annotators
        annotator_data_key = annotator_data_dict.keys()
        object_annotator_list = self.annotator_classification[WriterSetting.AnnotatorPrefix.ObjectDetection.GENERIC]
        generic_annotators = WriterUtils.select_elements_based_on_prefix(annotator_data_key, object_annotator_list)
        # Build the object annotator dictionary
        generic_annotators_dict = {key: annotator_data_dict[key] for key in generic_annotators}
        # Update the object info structure with the concatenated annotator data
        for annotator_name, annotator in generic_annotators_dict.items():
            object_prim_paths = annotator["primPaths"]
            id_to_label = annotator["idToLabels"]
            id_to_data = annotator["data"]

            for idx, prim_path in enumerate(object_prim_paths):
                semantic_id = id_to_data[idx]["semanticId"]
                label = id_to_label[semantic_id]
                object_data = id_to_data[idx]

                # Determine the appropriate info dictionary (agent or object)
                if self.agent_manager.is_agent_semantic_prim_path(prim_path):
                    agent_data = self.agent_manager.get_agent_data_by_label_path(label_path=prim_path)
                    prim_path = agent_data.prim_path
                    asset_url = agent_data.asset_url
                    info_dict = self.agent_info_dict
                    if prim_path not in info_dict:
                        info_dict[prim_path] = AgentInfo(label=label, prim_path=prim_path)
                        info_dict[prim_path].update_annotator_data("asset_url", asset_url)
                else:
                    if not self.output_objects:
                        continue
                    # if user set specific object list
                    if self.target_object_labels:
                        # check whether object type is in the list
                        if not label in self.target_object_labels:
                            continue

                    info_dict = self.object_info_dict
                    if prim_path not in info_dict:
                        info_dict[prim_path] = ObjectInfo(label=label, prim_path=prim_path)

                # Update the info dictionary with the annotator data
                info_dict[prim_path].update_annotator_data(annotator_name, object_data)


class ObjectInfoManager:
    """Info Manager that take annotator data and reconstrcut"""

    def __init__(
        self,
        output_objects: bool = False,
        annotator_classification: dict[str, list[str]] = {},
        target_object_labels: list[str] = None,
    ):
        # whether user request output object information
        self.output_objects = output_objects
        # structure that match camera view information to camera path
        self.path_to_camera_view_info: dict[str, ObjectDetectionInfo] = {}
        # self.annotators = annotators
        self.agent_manager = AgentManager.get_instance()
        # let the agent manager fetch all character data in the stage.
        self.agent_manager.extract_agent_data()
        self.annotator_classification = annotator_classification
        self.target_object_labels = target_object_labels

    def is_camera_exist(self, target_camera_path):
        return target_camera_path in self.path_to_camera_view_info

    def refresh_info(self, data: dict):
        """refresh each view port info via current annotator information"""
        # test what is the key of each data:
        for key, value in data.items():
            camera_path = value["camera"]
            if camera_path not in self.path_to_camera_view_info:
                self.path_to_camera_view_info[camera_path] = ObjectDetectionInfo(
                    path=camera_path,
                    output_objects=self.output_objects,
                    annotator_classification=self.annotator_classification,
                    target_object_labels=self.target_object_labels,
                )
            self.path_to_camera_view_info[camera_path].refresh_info(value)

    def get_camera_view_info(self, camera_path: str) -> ObjectDetectionInfo | None:
        """get camera view info via camrea path"""
        camera_view_info = self.path_to_camera_view_info.get(camera_path, None)
        return camera_view_info
