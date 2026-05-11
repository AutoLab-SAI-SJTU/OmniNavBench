from OmniNav.core.config import TaskCfg
from OmniNav.core.datahub import DataHub
from OmniNav.core.scene.scene import IScene
from OmniNav.core.task import BaseTask


@BaseTask.register('SingleInferenceTask')
class SimpleInferenceTask(BaseTask):
    def __init__(self, config: TaskCfg, scene: IScene):
        super().__init__(config, scene)

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        # Only end when stop is clicked
        return DataHub.get_episode_finished(self.name)
