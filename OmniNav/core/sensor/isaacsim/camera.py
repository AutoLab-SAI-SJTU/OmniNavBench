from typing import Optional, Tuple

import numpy as np
import omni.replicator.core as rep
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.usd import get_prim_at_path
from pxr import UsdGeom, Gf

from OmniNav.core.sensor.camera import ICamera


class IsaacsimCamera(ICamera):
    """
    IsaacSim's implementation on `ICamera` class.

    Args:
        name (str): The unique identifier for the camera.
        prim_path (Optional[str]): The primary path associated with the camera.
        rgba (Optional[bool], default=False): Whether to get rgba form the camera or not.
        distance_to_image_plane (Optional[bool], default=False): Whether to get distance_to_image_plane form the camera or not.
        bounding_box_2d_tight (Optional[bool], default=False): Whether to get bounding_box_2d_tight form the camera or not.
        camera_params (Optional[bool], default=False): Whether to get camera_params form the camera or not.
        resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ). Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim (with respect to its parent prim). shape is (3, ). Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim (depends if translation or position is specified). quaternion is scalar-first (w, x, y, z). shape is (4, ). Defaults to None, which means left unchanged.
    """

    def __init__(
        self,
        name: str = 'camera',
        prim_path: Optional[str] = None,
        rgba: Optional[bool] = True,
        distance_to_image_plane: Optional[bool] = False,
        distance_to_camera: Optional[bool] = False,
        bounding_box_2d_tight: Optional[bool] = False,
        bounding_box_2d_loose: Optional[bool] = False,
        bounding_box_3d: Optional[bool] = False,
        semantic_segmentation: Optional[bool] = False,
        instance_segmentation: Optional[bool] = False,
        instance_id_segmentation: Optional[bool] = False,
        pointcloud: Optional[bool] = False,
        camera_params: Optional[bool] = False,
        resolution: Optional[Tuple[int, int]] = None,
        position: Optional[Tuple[float, float, float]] = None,
        translation: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.name = name
        self.rgba = rgba
        self.distance_to_image_plane = distance_to_image_plane
        self.distance_to_camera = distance_to_camera
        self.bounding_box_2d_tight = bounding_box_2d_tight
        self.bounding_box_2d_loose = bounding_box_2d_loose
        self.bounding_box_3d = bounding_box_3d
        self.semantic_segmentation = semantic_segmentation
        self.instance_segmentation = instance_segmentation
        self.instance_id_segmentation = instance_id_segmentation
        self.pointcloud = pointcloud
        self.camera_params = camera_params
        self.rp = None
        self.rp_annotators = {}
        self.rp = rep.create.render_product(prim_path, resolution)
        self.prim = XFormPrim(prim_path)
        super().__init__()
        if position is not None:
            self.prim.set_world_pose(position, orientation)
        if translation is not None:
            self.prim.set_local_pose(translation, orientation)
        self.init_rp_annotators()

    def init_rp_annotators(self):
        if self.rgba:
            self.rp_annotators['rgba'] = rep.AnnotatorRegistry.get_annotator('LdrColor')
            self.rp_annotators['rgba'].attach(self.rp)
        if self.distance_to_image_plane:
            self.rp_annotators['distance_to_image_plane'] = rep.AnnotatorRegistry.get_annotator(
                'distance_to_image_plane'
            )
            self.rp_annotators['distance_to_image_plane'].attach(self.rp)
        if self.distance_to_camera:
            self.rp_annotators['distance_to_camera'] = rep.AnnotatorRegistry.get_annotator('distance_to_camera')
            self.rp_annotators['distance_to_camera'].attach(self.rp)
        if self.bounding_box_2d_tight:
            self.rp_annotators['bounding_box_2d_tight'] = rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight')
            self.rp_annotators['bounding_box_2d_tight'].attach(self.rp)
        if self.bounding_box_2d_loose:
            self.rp_annotators['bounding_box_2d_loose'] = rep.AnnotatorRegistry.get_annotator('bounding_box_2d_loose')
            self.rp_annotators['bounding_box_2d_loose'].attach(self.rp)
        if self.bounding_box_3d:
            self.rp_annotators['bounding_box_3d'] = rep.AnnotatorRegistry.get_annotator('bounding_box_3d')
            self.rp_annotators['bounding_box_3d'].attach(self.rp)
        if self.semantic_segmentation:
            self.rp_annotators['semantic_segmentation'] = rep.AnnotatorRegistry.get_annotator('semantic_segmentation')
            self.rp_annotators['semantic_segmentation'].attach(self.rp)
        if self.instance_segmentation:
            self.rp_annotators['instance_segmentation'] = rep.AnnotatorRegistry.get_annotator('instance_segmentation')
            self.rp_annotators['instance_segmentation'].attach(self.rp)
        if self.instance_id_segmentation:
            self.rp_annotators['instance_id_segmentation'] = rep.AnnotatorRegistry.get_annotator(
                'instance_id_segmentation'
            )
            self.rp_annotators['instance_id_segmentation'].attach(self.rp)
        if self.pointcloud:
            self.rp_annotators['pointcloud'] = rep.AnnotatorRegistry.get_annotator('pointcloud')
            self.rp_annotators['pointcloud'].attach(self.rp)
        if self.camera_params:
            self.rp_annotators['camera_params'] = rep.AnnotatorRegistry.get_annotator('camera_params')
            self.rp_annotators['camera_params'].attach(self.rp)

    def get_rgba(self) -> np.ndarray:
        """See `ICamera.get_rgba` for documentation."""
        if self.rgba:
            return self.rp_annotators['rgba'].get_data()
        return None

    def get_distance_to_image_plane(self) -> np.ndarray:
        """See `ICamera.get_distance_to_image_plane` for documentation."""
        if self.distance_to_image_plane:
            return self.rp_annotators['distance_to_image_plane'].get_data()
        return None

    def get_distance_to_camera(self) -> np.ndarray:
        if self.distance_to_camera:
            return self.rp_annotators['distance_to_camera'].get_data()
        return None

    def get_bounding_box_2d_tight(self) -> np.ndarray:
        """See `ICamera.get_bounding_box_2d_tight` for documentation."""
        if self.bounding_box_2d_tight:
            return self.rp_annotators['bounding_box_2d_tight'].get_data()
        return None

    def get_bounding_box_2d_loose(self) -> np.ndarray:
        if self.bounding_box_2d_loose:
            return self.rp_annotators['bounding_box_2d_loose'].get_data()
        return None

    def get_bounding_box_3d(self) -> np.ndarray:
        if self.bounding_box_3d:
            return self.rp_annotators['bounding_box_3d'].get_data()
        return None

    def get_semantic_segmentation(self) -> np.ndarray:
        if self.semantic_segmentation:
            return self.rp_annotators['semantic_segmentation'].get_data()
        return None

    def get_instance_segmentation(self) -> np.ndarray:
        if self.instance_segmentation:
            return self.rp_annotators['instance_segmentation'].get_data()
        return None

    def get_instance_id_segmentation(self) -> np.ndarray:
        if self.instance_id_segmentation:
            return self.rp_annotators['instance_id_segmentation'].get_data()
        return None

    def get_pointcloud(self) -> np.ndarray:
        if self.pointcloud:
            return self.rp_annotators['pointcloud'].get_data()
        return None

    def get_camera_params(self) -> np.ndarray:
        """See `ICamera.get_camera_params` for documentation."""
        if self.camera_params:
            return self.rp_annotators['camera_params'].get_data()
        return None

    def get_clipping_range(self) -> Optional[Tuple[float, float]]:
        """Return (near, far) clipping range if the prim has it."""
        prim = get_prim_at_path(self.prim.prim_path)
        if prim:
            camera = UsdGeom.Camera(prim)
            attr = camera.GetClippingRangeAttr()
            if attr.IsValid():
                value = attr.Get()
                if value is not None:
                    return float(value[0]), float(value[1])
        return None

    def set_clipping_range(self, near: float, far: float) -> None:
        """Set camera clipping range."""
        prim = get_prim_at_path(self.prim.prim_path)
        if prim:
            camera = UsdGeom.Camera(prim)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))

    def cleanup(self) -> None:
        if self.rp is not None:
            for anno in self.rp_annotators.values():
                anno.detach(self.rp)
            del self.rp_annotators
            self.rp_annotators = {}
            self.rp = None

    def unwrap(self):
        return self.prim
