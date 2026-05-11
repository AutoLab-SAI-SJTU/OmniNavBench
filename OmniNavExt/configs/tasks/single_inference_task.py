from typing import Optional

from OmniNav.core.config.task import TaskCfg


class SingleInferenceTaskCfg(TaskCfg):
    type: Optional[str] = 'SingleInferenceTask'
