from OmniNav.core.datahub import DataHub
from OmniNav.core.scene.scene import IScene
from OmniNav.core.task import BaseTask
from OmniNavExt.configs.tasks.finite_step_task import FiniteStepTaskCfg


@BaseTask.register('FiniteStepTask')
class FiniteStepTask(BaseTask):
    def __init__(self, config: FiniteStepTaskCfg, scene: IScene):
        super().__init__(config, scene)
        self.stop_count = 1
        self.max_steps = config.max_steps

    def is_done(self) -> bool:
        # Only end when stop is clicked, not based on step counter
        return DataHub.get_episode_finished(self.name)
