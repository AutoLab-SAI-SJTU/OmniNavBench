# from __future__ import annotations
__copyright__ = "Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import re
import socket
import subprocess as sp
import numpy as np
import sys
import functools
from shutil import which
from typing import Callable, Dict, Iterable, List, Tuple, Union, Any

import carb  # carb logging

# reload(sys.modules["omni.replicator.core"])
import omni.replicator.core as rep
import omni.timeline
import pynvml as pynvml
from OmniNavExt.data_generation.writers.writer import IRABasicWriter
from omni.metropolis.utils.usd_util import USDUtil
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, WriterRegistry
from omni.replicator.core.scripts.utils.utils import ReplicatorItem
from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture
from omni.replicator.core.scripts.writers_default.tools import colorize_distance, colorize_normals


__version__ = "0.0.1"

# [Change Log]
# 09/15/2023    Verified on omni.replicator.core-1.10.4+kit105.1.

# print(rep.__file__)
# # /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/__init__.py
# NvEnc only has a 48bit per pixel (unpacked) format. 64bit, 128bit per pixel formats are not supported by NVENC.
# https://forums.developer.nvidia.com/t/dxgi-nvenc-yuv444-10bit-format-compatability/189782
# NVENC can perform end-to-end encoding for H.264, HEVC 8-bit, HEVC 10-bit, AV1 8-bit and AV1 10-bit. Pixel channel
# datatypes of np.float16 or np.float32 cannot be supported. Fallback to libx264 (software) codec on grayf32le/be,
# rgba64le/be, gbrapf32le/be pixel formats.
# https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-application-note/index.html#nvenc-capabilities
# _annotator_type_uint8_channel_1 = [
#     'pointcloud_pointSemantic',
# ]
#
# https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
_annotator_type_uint8_channel_4 = [
    "LdrColor",
    "rgb",
    "semantic_segmentation",  # colorize = True
    "semantic_segmentation_fast",  # colorize = True
    "instance_id_segmentation",  # colorize = True
    "instance_id_segmentation_fast",
    "instance_segmentation",  # colorize = True
    "instance_segmentation_fast",
    "DiffuseAlbedo",  # ??? pixel format is wrong ???
    "Roughness",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    # 'pointcloud_pointRgb',
]

_annotator_type_float16_channel_1 = ["EmissionAndForegroundMask"]

_annotator_type_float32_channel_1 = [
    "DepthLinearized",
    # ??? pixel format is wrong ???
]

_annotator_type_float16_channel_4 = [
    "HdrColor",
    # 'AmbientOcclusion', # ??? data['AmbientOcclusion'] is empty ???
    # 'SpecularAlbedo', # ??? data['AmbientOcclusion'] is empty ???
    # 'DirectDiffuse', # ??? data['AmbientOcclusion'] is empty ???
    # 'DirectSpecular', # ??? data['AmbientOcclusion'] is empty ???
    # 'IndirectDiffuse', # ??? data['AmbientOcclusion'] is empty ???
    # # Path Tracing AOVs
    # 'PtDirectIllumation',
    # 'PtGlobalIllumination',
    # 'PtReflections',
    # 'PtRefractions',
    # 'PtSelfIllumination',
    # 'PtBackground',
    # 'PtWorldNormal',
    # 'PtWorldPos',
    # 'PtZDepth',
    # 'PtVolumes',
    # 'PtDiffuseFilter',
    # 'PtReflectionFilter',
    # 'PtRefractionFilter',
    # 'PtMultiMatte0',
    # 'PtMultiMatte1',
    # 'PtMultiMatte2',
    # 'PtMultiMatte3',
    # 'PtMultiMatte4',
    # 'PtMultiMatte5',
    # 'PtMultiMatte6',
    # 'PtMultiMatte7',
]

# _annotator_type_float32_channel_3 = [
#     'pointcloud', # data['pointcloud']['data']
# ]

# !!!Not able to play grbapf32le/be - Need help on ffmpeg encoding.!!!
# https://stackoverflow.com/questions/71725213/ffmpeg-cant-recognize-3-channels-with-each-32-bit
# _annotator_type_float32_channel_4 = [
#     'normals',
#     'motion_vectors',
#     'cross_correspondence',
#     'SmoothNormal', # ??? data['AmbientOcclusion'] is empty ???
#     'BumpNormal', # ??? data['AmbientOcclusion'] is empty ???
#     'Motion2d', # ??? data['AmbientOcclusion'] is empty ???
#     'Reflections', # ??? data['AmbientOcclusion'] is empty ???
#     # 'pointcloud_pointNormals',
# ]

# FIXME::
# currently the visualization of channel 1 image streaming has potential issue.
# for the visualization, converting progress to make it 4 channel is applied.


_nvenc_annotators = _annotator_type_uint8_channel_4
# 'pointcloud', # Empty from camera. Is this a bug???

_software_annotators = (
    _annotator_type_float16_channel_1 + _annotator_type_float32_channel_1 + _annotator_type_float16_channel_4
)

_supported_annotators = _nvenc_annotators + _software_annotators

# Default GPU device where NVENC operates on
_default_device = 0
# Default annotator of each render product to capture and stream
_default_annotator = "LdrColor"


class RTSPCamera:
    """The class records a render products (HydraTexture) by its prim path.
    The class also records the ffmpeg subprocess command which is customized
    by the render product's camera parameters, e.g. fps, width, height, annotator.
    The published RTSP URL of each RTSPCamera instance is constructed by appending
    the render product's camera prim path and the annotator name to the base output directory.

    Notes:
        The supported annotators are:
            'LdrColor' / 'rgb',
            'semantic_segmentation',
            'instance_id_segmentation',
            'instance_segmentation',
            'DiffuseAlbedo',
            'Roughness',
            'EmissionAndForegroundMask'
            'distance_to_camera',
            'distance_to_image_plane',
            'DepthLinearized',
            'HdrColor'
        Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
        for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
        'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
        are software encoded by CPU. The default video stream format is HEVC.
    """

    _default_device = _default_device
    _default_fps = 30  # RTSP stream frame rate. Should be consistent with render product frame rate.
    _default_bitrate = 5000  # encoding bitrate in kb.
    _default_annotator = _default_annotator
    _default_resolution = (1920, 1080)

    _name = "RTSPCamera"
    _version = __version__

    def __init__(
        self,
        rtsp_stream_url: str, # for rtsp, it is a necessary input
        device: int = _default_device,  # GPU device for encoding acceleration
        fps: int = _default_fps,  # Streaming FPS
        resolution: Tuple[int, int] = _default_resolution,
        bitrate: int = _default_bitrate,  # Encoding bit rate
        annotator: str = _default_annotator,  # Only support ONE annotator
        # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
        prim_path: str = None,  # Camera prim path (how to do "from .utils.viewport_manager import HydraTexture")
    ):
        """Create an RTSP camera instance

        Args:
            device:     GPU device id to be used for hardware accelerated encoding.
            fps:        The encoded video frame rate.
            resolution:      The encoded video frame resolution in pixels.
            bitrate:    The maximal value of the encoded video bitrate in kb/s.
            annotator: The annotator data to be streamed. The supported annotators are:
                'LdrColor' / 'rgb',
                'semantic_segmentation',
                'instance_id_segmentation',
                'instance_segmentation',
                'DiffuseAlbedo',
                'Roughness',
                'EmissionAndForegroundMask'
                'distance_to_camera',
                'distance_to_image_plane',
                'DepthLinearized',
                'HdrColor'
                Please refer to
                https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
            rtsp_stream_url: The base RTSP URL defined by server, port, and topic. Each
                        render product has the unique RTSP URL which is constructed by
                        appending the render product name by the end of rtsp_stream_url.
            prim_path:  The full camera prim path
        """
        # extract the width and height from the resolution


        if (not prim_path) or (not rtsp_stream_url):
            raise ValueError(f"{prim_path} and {rtsp_stream_url} need to be set")


        self.initialize(
            device=device,
            fps=fps,
            resolution=resolution,
            bitrate=bitrate,
            annotator=annotator,
            rtsp_stream_url=rtsp_stream_url,
            prim_path=prim_path,
        )

    def initialize(
        self,
        device: int = _default_device,  # GPU device for encoding acceleration
        fps: int = _default_fps,  # Streaming FPS
        resolution: Tuple[int, int] = _default_resolution,
        bitrate: int = _default_bitrate,  # x264enc bit rate
        annotator: str = _default_annotator,  # Only support ONE annotator
        rtsp_stream_url: str = None,  # Base RTSP URL
        prim_path: str = None,  # RTSP URL
    ):
        """Initialize the writer.

        Args:
            device:     GPU device id to be used for hardware accelerated encoding.
            fps:        The encoded video frame rate.
            resolution: The encoded video resolution in pixels.
            bitrate:    The maximal value of the encoded video bitrate in kb/s.
            annotator: The annotator data to be streamed. The supported annotators are:
                'LdrColor' / 'rgb',
                'semantic_segmentation',
                'instance_id_segmentation',
                'instance_segmentation',
                'DiffuseAlbedo',
                'Roughness',
                'EmissionAndForegroundMask'
                'distance_to_camera',
                'distance_to_image_plane',
                'DepthLinearized',
                'HdrColor'
                Please refer to
                https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
            rtsp_stream_uri: The base RTSP URL defined by server, port, and topic. Each
                        render product has the unique RTSP URL which is constructed by
                        appending the render product name by the end of rtsp_stream_uri.
            prim_path:  The full camera prim path
        Returns:
            None
        """

        self.device = device
        self.fps = fps
        self.resolution = resolution
        self.bitrate = bitrate
        self.annotator = annotator
        self.rtsp_stream_url = rtsp_stream_url
        self.prim_path = prim_path
        self._verify_init_arguments()

        # TODO: Support more annotators described in https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
        # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html?highlight=hdrcolor#annotator-output
        # https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py
        # Annotators encoded by NVENC.
        if self.annotator in _annotator_type_uint8_channel_4:
            self.pixel_fmt = "rgba"
        elif self.annotator in _annotator_type_float16_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "gray16le"
            else:
                self.pixel_fmt = "gray16be"
        elif self.annotator in _annotator_type_float32_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "grayf32le"
            else:
                self.pixel_fmt = "grayf32be"
        elif self.annotator in _annotator_type_float16_channel_4:
            if sys.byteorder == "little":
                self.pixel_fmt = "rgba64le"
            else:
                self.pixel_fmt = "rgba64be"
        # elif self.annotator in ['pointcloud']: # (N,3) - np.float32
        #     # pc_data = pointcloud_anno.get_data()
        #     # print(pc_data)
        #     # # {
        #     # #     'data': array([...], shape=(<num_points>, 3), dtype=float32),
        #     # #     'info': {
        #     # #         'pointNormals': [ 0.000e+00 1.00e+00 -1.5259022e-05 ... 0.00e+00 -1.5259022e-05 1.00e+00], shape=(<num_points> * 4), dtype=float32),
        #     # #         'pointRgb': [241 240 241 ... 11  12 255], shape=(<num_points> * 4), dtype=uint8),
        #     # #         'pointSemantic': [2 2 2 ... 2 2 2], shape=(<num_points>), dtype=uint8),
        #     # #
        #     # #     }
        #     # # }
        #     # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html?highlight=hdrcolor#point-cloud
        #     # Due to the 48bit per pixel limitation in NVENC, 'data' and 'info'.'pointNormals' are not supported.
        #     # By default, render 'info'.'pointRgb' - (N, 4) - np.uint8
        #     self.pixel_fmt = 'rgba'
        # Annotators encoded by software due to NVENC 48bit/pixel limit
        # elif self.annotator in _annotator_type_float32_channel_4:
        #     # output_data_type=np.float32, output_channels=4
        #     # https://stackoverflow.com/questions/1346034/whats-the-most-pythonic-way-of-determining-endianness
        #     if sys.byteorder == 'little':
        #         self.pixel_fmt = 'gbrapf32le'
        #     else:
        #         self.pixel_fmt = 'gbrapf32be'
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")

        # Replace special character/separator with '_'
        prim_path = self.prim_path.replace("/", "_")
        prim_path = prim_path.replace(" ", "_")
        # generate a distinguishable url path for each annotator
        full_rtsp_stream_url_path = self.rtsp_stream_url + prim_path + "_" + self.annotator

        # --- Verify ffmpeg installed ---
        if not which("ffmpeg"):
            raise ValueError(f"ffmpeg cannot be found in the system.")
        # --- Verify ffmpeg installed ---

        # Video writer
        # https://video.stackexchange.com/questions/12905/repeat-loop-input-video-with-ffmpeg
        # To enable GPU acceleration, please uncommented these two lines. One for the input. One for the output.
        # '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-hwaccel_device', str(device),
        # '-c:v', 'h264_nvenc', '-c:a', 'copy', # encodes all video streams with h264_nvenc.
        self.command = ["ffmpeg"]

        if self.annotator in _nvenc_annotators:
            self.command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-hwaccel_device", str(self.device)]
            vcodec = "hevc_nvenc"  # 'h264_nvenc'
        elif self.annotator in _software_annotators:  # Software encoding
            vcodec = "libx265"  # 'libx264'
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")

        # https://stackoverflow.com/questions/71725213/ffmpeg-cant-recognize-3-channels-with-each-32-bit

        self.command += [
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            self.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{self.resolution[0]}x{self.resolution[1]}",
            # '-r', str(fps),
            # '-stream_loop', '-1', # Loop infinite times.
            "-i",
            "-",
            "-c:a",
            "copy",
            "-c:v",
            vcodec,
            "-preset",
            "ll",
            # '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            # '-vbr', '5', # Variable Bit Rate (VBR). Valid values are 1 to 5. 5 has the best quality.
            # '-b:v', f'{bitrate}k',
            "-maxrate:v",
            f"{self.bitrate}k",
            "-bufsize:v",
            "64M",  # Buffering is probably required
            # passthrough (0) - Each frame is passed with its timestamp from the demuxer to the muxer.
            # -vsync 0 cannot be applied together with -r/-fpsmax.
            # cfr (1) - Frames will be duplicated and dropped to achieve exactly the requested constant frame rate.
            # vfr (2) - Frames are passed through with their timestamp or dropped so as to prevent 2 frames from having the same timestamp.
            # '-vsync', 'passthrough',
            "-vsync",
            "cfr",
            "-r",
            str(self.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            full_rtsp_stream_url_path,
        ]
        # switch from the info to warn to make the annotator more obvious
        carb.log_warn(
            f'"{self.annotator}" of "{self.prim_path}" will be published to "{full_rtsp_stream_url_path}" encoded by "{vcodec}".'
        )

        self.pipe = None

    def _verify_init_arguments(self):
        if self.device <= 0:
            self.device = self._default_device

        if self.fps <= 0:
            self.fps = self._default_fps

        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            self.resolution = self._default_resolution

        if self.bitrate <= 0:
            self.bitrate = self._default_bitrate

        if self.annotator not in _supported_annotators:
            raise ValueError(f"{self.annotator} is not part of the supported annotators, {_supported_annotators}.")

        if not self.rtsp_stream_url:
            raise ValueError(f'{self._name} member "rtsp_stream_url" cannot be empty.')

        if not self.prim_path:
            raise ValueError(f'{self._name} member "prim_path" cannot be empty.')

    def get_resolution(self) -> Tuple[int, int]:
        return self.resolution


class RTSPWriter(IRABasicWriter):
    """Publish annotations of attached render products to an RTSP server.

    The Writer tracks a dictionary of render products (HydraTexture) by the combo of the
    annotator name and the render product's prim path. Each render product is recorded as
    an instance of RTSPCamera. The published RTSP URL of each RTSPCamera instance is
    constructed by appending the render product's camera prim path and the annotator name
    to the base output directory.

    The supported annotators are:
        'LdrColor' / 'rgb',
        'semantic_segmentation',
        'instance_id_segmentation',
        'instance_segmentation',
        'DiffuseAlbedo',
        'Roughness',
        'EmissionAndForegroundMask'
        'distance_to_camera',
        'distance_to_image_plane',
        'DepthLinearized',
        'HdrColor'
    Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
    for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
    'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
    are software encoded by CPU. The video stream format is HEVC.
    """

    # Default RTSP server, port, and topic.
    _default_device = _default_device
    _default_annotator = _default_annotator
    _default_server = "localhost"
    _default_port = 8554
    _default_topic = "RTSPWriter"
    _default_rtsp_stream_url = f"rtsp://{_default_server}:{_default_port}/{_default_topic}"

    _push_stream_fn_suffix = "push_stream_frame"

    _name = "RTSPWriter"
    _version = __version__

    @classmethod
    def params_values(cls) -> dict:
        params_values_dict = {}
        # Set the default value
        params_values_dict["output_dir"] = "" # set the parameter to an empty string
        # Gather self parameters
        params_values_dict["rtsp_stream_url"] = cls._default_rtsp_stream_url
        # Gather parent parameters
        params_values_dict["rtsp_rgb"] = True
        params_values_dict["rtsp_semantic_segmentation"] = False
        params_values_dict["rtsp_instance_id_segmentation"] = False
        params_values_dict["rtsp_instance_segmentation"] = False
        params_values_dict["rtsp_normals"] = False
        params_values_dict["rtsp_distance_to_image_plane"] = False
        params_values_dict["rtsp_distance_to_camera"] = False
        return params_values_dict

    @classmethod
    def postprocess_params(cls, filtered_argus: Dict, extra_parameters: Dict):
        if extra_parameters.get("rtsp_rgb", False):
            filtered_argus["rgb"] = True

        if extra_parameters.get("rtsp_semantic_segmentation", False):
            filtered_argus["semantic_segmentation"] = True
            filtered_argus["colorize_semantic_segmentation"] = True

        if extra_parameters.get("rtsp_instance_id_segmentation"):
            filtered_argus["instance_id_segmentation"] = True
            filtered_argus["colorize_instance_id_segmentation"] = True

        if extra_parameters.get("rtsp_instance_segmentation"):
            filtered_argus["instance_segmentation"] = True
            filtered_argus["colorize_instance_segmentation"] = True

        if extra_parameters.get("rtsp_distance_to_image_plane"):
            filtered_argus["distance_to_image_plane"] = True
            # NOTE::colorize the depth is not necessary in following test
            # this parameter is added to indicate that the depth information has been converted to image format
            filtered_argus["colorize_depth"] = True
        if extra_parameters.get("rtsp_distance_to_camera"):
            filtered_argus["distance_to_camera"] = True
            filtered_argus["colorize_depth"] = True

        if extra_parameters.get("rtsp_normals"):
            filtered_argus["normals"] = True

    def __init__(
        self,
        rtsp_rgb: bool = True,
        rtsp_semantic_segmentation: bool = False,
        rtsp_instance_id_segmentation: bool = False,
        rtsp_instance_segmentation: bool = False,
        rtsp_normals: bool = False,
        rtsp_distance_to_image_plane: bool = False,
        rtsp_distance_to_camera: bool = False,
        rtsp_stream_url:str = None,
        device: int = _default_device,  # GPU device where NVENC operates on.
        *args,
        **kwargs,
    ):
        """Initialize the writer.

        Args:
            rtsp_stream_url: The base RTSP URL defined by server, port, and topic. Each
                render product has the unique RTSP URL which is constructed by
                appending the render product name by the end of rtsp_stream_url.
        """

        # add prefix toward the annotator, distinguish them from the original annotator. since we are process them with a different way
        # fetch all extension parameter (aka the parameter with "rtsp_" prefix )
        extra_parameters = {k: v for k, v in locals().items() if k not in ["self", "args", "kwargs"]}
        self.postprocess_params(filtered_argus=kwargs, extra_parameters=extra_parameters)
        # FIXME::
        # we need a more elegant way to handle the output_dir input
        kwargs["output_dir"] = "placeholder"  # set a placeholder to avoid throw error message
        # the construction of the obj_info_manager should be selectable
        # Intentionally skip IRABasicWriter.__init__ to avoid XXX side-effects
        super(IRABasicWriter, self).__init__(*args, **kwargs)  # noqa: pylint=E1003
        self.data_structure = "renderProduct"
        # self.skip_frames = carb.settings.get_settings().get(
        #     "/persistent/exts/isaacsim.replicator.agent/skip_starting_frames"
        # )
        self.rtsp_stream_url = rtsp_stream_url
        self.setup_stream(device=device)

    # FIXME:: change the name confliction.
    # Be default writer would use initlaize to input parameter
    # which would trigger init. Therefore, initalize and init cannot include each other
    def setup_stream(
        self,
        device: int = _default_device,  # GPU device where NVENC operates on.
    ):
        """Initialize the writer.

        Args:
            rtsp_stream_url: The base RTSP URL defined by server, port, and topic. Each
                render product has the unique RTSP URL which is constructed by
                appending the render product name by the end of rtsp_stream_url.

        Returns:
            None
        """
        self.device = device
        if self.device < 0:
            self.device = _default_device

        # if annotator == "rgb":
        #     annotator = "LdrColor"
        # if annotator not in _supported_annotators:
        #     raise ValueError(f"{annotator} is not part of the supported annotators, {_supported_annotators}.")

        # self.annotators = []
        # if annotator in ["semantic_segmentation", "instance_id_segmentation", "instance_segmentation"]:
        #     # https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/writers_default/basicwriter.py
        #     # If ``True``, semantic segmentation is converted to an image where semantic IDs are mapped to colors
        #     # and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
        #     # Defaults to ``True``.
        #     self.annotators.append(AnnotatorRegistry.get_annotator(annotator, init_params={"colorize": True}))
        # else:
        #     self.annotators.append(AnnotatorRegistry.get_annotator(annotator))
        # print(f'DEBUG: dir(self.annotators[0]) = {dir(self.annotators[0])}')
        # print(f'DEBUG: self.annotators[0].get_name() = {self.annotators[0].get_name()}')
        # --- Verify RTSP URL ---
        match = re.search(r"^rtsp\://(.+)\:([0-9]+)/(.+)$", self.rtsp_stream_url)
        if not match:
            raise ValueError(
                f'{self.rtsp_stream_url} is not a valid RTSP stream URL. The format is "rtsp://<hostname>:<port>/<topic>".'
            )

        hostname = match.group(1)
        port = match.group(2)
        # topic = match.group(3)

        # Verify RTSP server is live
        sock = socket.socket()
        try:
            sock.connect((hostname, int(port)))
            # originally, it was
            # except Exception, e:
            # but this syntax is not supported anymore.
        except Exception as e:
            raise ValueError(f"RTSP server at {hostname}:{port} is not accessible with exception {e}.")
        finally:
            sock.close()
        # --- Verify RTSP URL ---
        self.backend = None  # BackendDispatch({"paths": {"out_dir": "/tmp"}}) # None
        self._backend = self.backend  # Kept for backwards compatibility

        self.cameras: Dict[str, Dict[str, RTSPCamera]] = {}  # map render_product name to a subprocess
        self._frame_id = 0
        self._tcp_retries = 0

        pynvml.nvmlInit()
        try:
            self.nDevices = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as error:
            carb.log_warn(error)
            self.nDevices = 1
            pass

        carb.log_warn("Setup streamer url successfully. at this location" + str(self.rtsp_stream_url))
        pynvml.nvmlShutdown()

    @staticmethod
    def default_device():
        return _default_device

    @staticmethod
    def default_annotator():
        return _default_annotator

    @staticmethod
    def supported_annotators():
        return _supported_annotators

    @staticmethod
    def default_rtsp_stream_url():
        return RTSPWriter._default_rtsp_stream_url

    @classmethod
    def allow_basic_writer_params(cls) -> bool:
        return False

    @classmethod
    def tooltip(cls):
        return f"""
            RTSPWriter
            - It will use the output path under 'Parameters'.
            - It supports one annotator at a time. All available annotators:
                {str(RTSPWriter.supported_annotators())}.


            """

    # def register_push_stream_fn(func):
    #     """Register a function dynamically as an instance method."""
    #     method_name = f"{func.__name__}{RTSPWriter._push_stream_fn_suffix}"
    #     setattr(RTSPWriter, method_name, functools.partial(func, RTSPWriter))  # Bind function
    #     print(f"Registered instance method: {method_name}")
    #     delattr(RTSPWriter, func.__name__)

    @classmethod
    def generate_stream_camera_data(cls, render_product_info: Any):
        """extract camera path, resolution info from the render product"""
        render_product_path = (
            render_product_info.path if isinstance(render_product_info, HydraTexture) else render_product_info
        )
        camera_path = str(render_product_info.hydra_texture.camera_path)
        image_width = render_product_info.hydra_texture.width
        image_height = render_product_info.hydra_texture.height
        # compose the image resolution
        camera_resolution = (image_width, image_height)

        return camera_path, camera_resolution

    def on_final_frame(self):
        """Run after final frame is written.

        Notes:
            When "stop" button is clicked in Isaac Sim UI, this function is called.
            The ffmpeg subprocesses are killed (SIGKILL).
        """
        super().on_final_frame()

        for camera_id, anno_streamer_dict in self.cameras.items():
            for anno_name, camera in anno_streamer_dict.items():
                if camera.pipe:
                    camera.pipe.stdin.close()
                    camera.pipe.wait()
                    camera.pipe.kill()
                    camera.pipe = None
                    carb.log_info(f'Subprocess on "{camera.prim_path}" has been terminated.')
            anno_streamer_dict.clear()

        self.cameras.clear()
        self._frame_id = 0

    # Failed in gitlab-master.nvidia.com:5005/isaac/omni_isaac_sim/isaac-sim:latest-2023.1 with
    # 2023-09-14 21:23:55 [74,802ms] [Error] [omni.kit.app.plugin] [py stderr]: IndexError: list index out of range
    #
    # At:
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(750): _attach
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(551): attach
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(388): attach
    #   /tmp/carb.h59qSD/script_1694726635.py(374): attach
    def _config_stream_cameras(self, render_product_list: List):
        """Attach one or a list of render products to the writer.

        This is the base function called by either attach() or attach_async().
        The function constructs the ffmpeg command for each render product.
        Each render product associates with an unique RTSP URL which is built
        by appending render product's camera prim path to self.rtsp_stream_url.

        Args:
            data: A dictionary containing the annotator data for the current frame.

        Returns:
            None
        """
        # Clear existing camera pipes
        if self.cameras:
            self.on_final_frame()
        # extract the streamable annotator according to the setting
        annotator_list = self.filter_streamable_annotator()
        rp_path_idx = 0
        for render_product_info in render_product_list:
            # Make keys in "self.cameras" match keys in "data".
            # When there is a single render product, key = <annotator>
            # When there are multiple render products, key = <annotator>-<rp_path[3:]>
            # Please refer to RTSPWriter.write() for details.
            for annotator_name in annotator_list:
                camera_path, camera_resolution = self.generate_stream_camera_data(render_product_info)
                # check whether the data stream id is generated successfully
                # Distribute ffmpeg HW encoding among multiple GPUs
                if self.nDevices > 1:
                    device = rp_path_idx % self.nDevices
                else:
                    device = 0

                camera = RTSPCamera(
                    device=device,
                    resolution=camera_resolution,
                    annotator=annotator_name,
                    rtsp_stream_url=self.rtsp_stream_url,
                    prim_path=camera_path,  # Camera prim path
                )

                camera.pipe = sp.Popen(camera.command, stdin=sp.PIPE)
                if not camera.pipe:
                    raise Exception(f"Can't start ffmpeg RTSP client writer on {camera.prim_path}.")

                if camera_path not in self.cameras:
                    self.cameras[camera_path] = {}

                if annotator_name not in self.cameras[camera_path]:
                    self.cameras[camera_path][annotator_name] = camera
                rp_path_idx += 1

    def filter_streamable_annotator(self) -> List[str]:
        """filter out the colorized/streamable annotator within the annotator list"""
        # TODO:: implement the filter to filter out colorized annotator depends on the settings
        annotator_list = []
        for annotator in self.annotators:
            annotator_name = annotator.get_name()
            # check whether the annotator is supportable
            if annotator_name in _supported_annotators:
                annotator_list.append(annotator_name)
            else:
                carb.log_warn(f"Warning:: {annotator_name} is not a streamable annotator type with current settings. ")

        return annotator_list

    def attach(
        self,
        render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]],
        trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
    ):
        """Attach writer to specified render products, rewrite original attach function, register the stream camera to the list.

        Args:
            render_products: Render Product prim path(s) to which to attach the writer.
            trigger: Function or replicator trigger that triggers the ``write`` function of the writer. If a function
                is supplied, it must return a boolean. If set to ``None``, the writer is set to a manual mode
                where it can be triggered by calling ``writer.schedule_write``.
        """
        super().attach(render_products=render_products, trigger=trigger)
        # set up the stream camera process
        self._config_stream_cameras(render_product_list=render_products)

    def write(self, data):
        """original writer would be discard templately"""
        # add a template test version

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        render_product = dict(data["renderProducts"])
        for key, annotator_dict in render_product.items():
            # get current camera_path
            self.stream_each_camera(annotator_dict=annotator_dict)

        self._frame_id += 1

    def fetch_anno_streamer(self, camera_path: str, annotator_name: str) -> RTSPCamera | None:
        anno_streamer_dict = self.cameras.get(camera_path, None)
        # check whether the camera has been registered succesfully
        if not anno_streamer_dict:
            return None
        # check whether the annotator streamer has been attached
        anno_streamer = anno_streamer_dict.get(annotator_name, None)
        return anno_streamer

    def stream_each_camera(self, annotator_dict: dict):
        """stream the data of every camera"""
        camera_path = str(annotator_dict["camera"])
        streamable_annotators = self.filter_streamable_annotator()
        for annotator_name in streamable_annotators:
            annotator_data = annotator_dict.get(annotator_name, None)
            if annotator_data is None:
                continue
            else:
                push_stream_fn_name = f"{annotator_name}_{self._push_stream_fn_suffix}"
                # fetch the target push stream fn name
                if hasattr(self, push_stream_fn_name):
                    default_output_function = getattr(self, push_stream_fn_name)
                    # fetch target streamer with the annotator and target camera path
                    anno_streamer = self.fetch_anno_streamer(camera_path=camera_path, annotator_name=annotator_name)
                    default_output_function(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def rgb_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream rgb to the remote machine"""
        data = annotator_data.get("data", None)
        if data is not None:
            anno_streamer.pipe.stdin.write(data.tobytes())

    def semantic_segmentation_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream segmentation to the remote machine"""
        self.segmentation_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def instance_segmentation_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream instance_segmentation to the remote machine"""
        self.segmentation_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def instance_segmentation_fast_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream instance_segmentation_fast to the remote machine"""
        self.segmentation_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def instance_id_segmentation_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream instance_id_segmentation to the remote machine"""
        self.segmentation_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def instance_id_segmentation_fast_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream instance_id_segmentation_fast to the remote machine"""
        self.segmentation_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)


    def segmentation_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream rgb to the remote machine"""
        seg_data = annotator_data.get("data", None)
        if seg_data is not None:
            # fetch the height and width of the image
            width, height = anno_streamer.get_resolution()
            seg_data = seg_data.view(np.uint8).reshape(height, width, -1)
            anno_streamer.pipe.stdin.write(seg_data.tobytes())

    def depth_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """handle the camera distance input"""
        depth_data = annotator_data.get("data", None)
        if depth_data is not None:
            # fetch the height and width of the image
            width, height = anno_streamer.get_resolution()
            depth_data = colorize_distance(depth_data, near=None, far=None)
            # compose a (height,width,4) shape np array
            depth_data = np.stack([depth_data] * 3 + [np.full((height, width), 255, dtype=np.uint8)], axis=-1)
            anno_streamer.pipe.stdin.write(depth_data.tobytes())

    def normals_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        """stream normals to the remote machine"""
        normals_data = annotator_data.get("data", None)
        if normals_data is not None:
            normals_data = colorize_normals(normals_data)
            anno_streamer.pipe.stdin.write(normals_data.tobytes())

    def distance_to_image_plane_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        self.depth_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def distance_to_camera_push_stream_frame(self, anno_streamer: RTSPCamera, annotator_data: Dict):
        self.depth_push_stream_frame(anno_streamer=anno_streamer, annotator_data=annotator_data)

    def write_metadata(self):
        # pass
        self._is_warning_backend_posted = True
        self._is_metadata_written = True


rep.WriterRegistry.register(RTSPWriter)
