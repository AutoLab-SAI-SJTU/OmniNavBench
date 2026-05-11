from collections import OrderedDict

from OmniNav.core.config import TaskCfg
from OmniNav.core.object.object import BaseObject, create_objects
from OmniNav.core.scene.scene import IScene


def init_objects(task_config: TaskCfg, scene: IScene) -> OrderedDict[str, BaseObject]:
    return create_objects(task_config, scene)
