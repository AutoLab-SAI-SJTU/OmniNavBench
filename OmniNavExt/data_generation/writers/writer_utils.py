import carb
import numpy as np
import inspect
from omni.replicator.core.scripts.utils import skeleton_data_utils


class WriterUtils:

    EPS = 1e-5

    def extract_prefix_and_remainder(string: str, prefix_list: list[str]):
        """seperate prefix from the string"""
        for prefix in prefix_list:
            if string.startswith(prefix):
                remainder = string[len(prefix) :]
                return prefix, remainder.strip("_ ").strip()
        return None, string

    def select_elements_based_on_prefix(list_a: list[str], list_b: list[str]):
        """for arbitary element in list a if it start with arbitary element in list b, select it"""
        selected_elements = []
        for element in list_a:
            for prefix in list_b:
                if element.startswith(prefix):
                    selected_elements.append(element)
        return selected_elements

    def numpy_encoder(obj):
        """help the writer to output data in correct format"""
        if isinstance(obj, np.void):
            return WriterUtils.convert_to_serialized_dict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()

        return obj

    def convert_to_serialized_dict(object_data):
        """
        helper method, convert the void type output to python format
        """
        temp_dict = {}
        for field_name in object_data.dtype.names:
            field_value = object_data[field_name]

            # recursively convert the numpy format value to default python format
            temp_dict[field_name] = WriterUtils.numpy_encoder(field_value)

        return temp_dict

    def post_process_skeleton_datas(anno_rp_data) -> dict:
        """post process the skeleton data, convert into skeleton dict"""

        skeleton_dict = {}
        skel_name = anno_rp_data["skelName"]
        skel_path = anno_rp_data["skelPath"]
        asset_path = anno_rp_data["assetPath"]
        animation_variant = anno_rp_data["animationVariant"]
        skeleton_parents = skeleton_data_utils.get_skeleton_parents(
            anno_rp_data["numSkeletons"], anno_rp_data["skeletonParents"], anno_rp_data["skeletonParentsSizes"]
        )
        rest_global_translations = skeleton_data_utils.get_rest_global_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restGlobalTranslations"],
            anno_rp_data["restGlobalTranslationsSizes"],
        )
        rest_local_translations = skeleton_data_utils.get_rest_local_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restLocalTranslations"],
            anno_rp_data["restLocalTranslationsSizes"],
        )
        rest_local_rotations = skeleton_data_utils.get_rest_local_rotations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restLocalRotations"],
            anno_rp_data["restLocalRotationsSizes"],
        )
        global_translations = skeleton_data_utils.get_global_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["globalTranslations"],
            anno_rp_data["globalTranslationsSizes"],
        )
        local_rotations = skeleton_data_utils.get_local_rotations(
            anno_rp_data["numSkeletons"], anno_rp_data["localRotations"], anno_rp_data["localRotationsSizes"]
        )
        translations_2d = skeleton_data_utils.get_translations_2d(
            anno_rp_data["numSkeletons"], anno_rp_data["translations2d"], anno_rp_data["translations2dSizes"]
        )
        skeleton_joints = skeleton_data_utils.get_skeleton_joints(anno_rp_data["skeletonJoints"])
        joint_occlusions = skeleton_data_utils.get_joint_occlusions(
            anno_rp_data["numSkeletons"], anno_rp_data["jointOcclusions"], anno_rp_data["jointOcclusionsSizes"]
        )
        occlusion_types = skeleton_data_utils.get_occlusion_types(
            anno_rp_data["numSkeletons"], anno_rp_data["occlusionTypes"], anno_rp_data["occlusionTypesSizes"]
        )
        in_view = anno_rp_data["inView"]

        for skel_num in range(anno_rp_data["numSkeletons"]):
            skeleton_dict[f"skeleton_{skel_num}"] = {}
            skeleton_dict[f"skeleton_{skel_num}"]["skel_name"] = skel_name[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skel_path"] = skel_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["asset_path"] = asset_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["animation_variant"] = animation_variant[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_parents"] = (
                skeleton_parents[skel_num].tolist() if skeleton_parents else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_global_translations"] = (
                rest_global_translations[skel_num].tolist() if rest_global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_translations"] = (
                rest_local_translations[skel_num].tolist() if rest_local_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_rotations"] = (
                rest_local_rotations[skel_num].tolist() if rest_local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["global_translations"] = (
                global_translations[skel_num].tolist() if global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["local_rotations"] = (
                local_rotations[skel_num].tolist() if local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["translations_2d"] = (
                translations_2d[skel_num].tolist() if translations_2d else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_joints"] = (
                skeleton_joints[skel_num] if skeleton_joints else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["joint_occlusions"] = (
                joint_occlusions[skel_num].tolist() if joint_occlusions else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["occlusion_types"] = (
                occlusion_types[skel_num] if occlusion_types else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["in_view"] = bool(in_view[skel_num]) if in_view.any() else False

        return skeleton_dict

    def filter_dict_by_keys(data_dict, keys_to_keep):
        """
        only keepe the key-value pairs that are recoreded in keys_to_keep 。
        """
        keys_to_keep = set(keys_to_keep)

        for key in list(data_dict.keys()):
            if key not in keys_to_keep:
                data_dict.pop(key)

        return data_dict

    def calculate_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points in 3D space.

        point1: List or array of the form [x, y, z]
        point2: List or array of the form [x1, y1, z1]
        """
        # Convert the points to numpy arrays if they aren't already
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Calculate the Euclidean distance
        distance = np.sqrt(np.sum((point2 - point1) ** 2))

        return distance

    def inspect_writer_init(writer_class):
        result = {}
        for name, param in inspect.signature(writer_class.__init__).parameters.items():
            # Skip ambigious params
            if name == "self" or name == "args" or name == "kwargs":
                continue
            # Only gather params that can be displayed
            # if param.annotation == bool or param.annotation == int or param.annotation == float or param.annotation == str:
            #     result[name] = param.default
            result[name] = param.default
        return result
