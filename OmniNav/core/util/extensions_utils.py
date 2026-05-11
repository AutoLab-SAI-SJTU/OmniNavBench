def dump_extensions():

    extensions = {}
    from OmniNav.core.robot.controller import BaseController

    extensions['controllers'] = BaseController.controllers
    from OmniNav.core.util.interaction import BaseInteraction

    extensions['interactions'] = BaseInteraction.interactions
    from OmniNav.core.task.metric import BaseMetric

    extensions['metrics'] = BaseMetric.metrics
    from OmniNav.core.object.object import BaseObject

    extensions['objs'] = BaseObject.objs
    from OmniNav.core.robot.robot import BaseRobot

    extensions['robots'] = BaseRobot.robots
    from OmniNav.core.sensor.sensor import BaseSensor

    extensions['sensors'] = BaseSensor.sensors
    from OmniNav.core.task import BaseTask

    extensions['tasks'] = BaseTask.tasks
    return extensions


def reload_extensions(extensions):

    from OmniNav.core.robot.controller import BaseController

    BaseController.controllers = extensions['controllers']
    from OmniNav.core.util.interaction import BaseInteraction

    BaseInteraction.interactions = extensions['interactions']
    from OmniNav.core.task.metric import BaseMetric

    BaseMetric.metrics = extensions['metrics']
    from OmniNav.core.object.object import BaseObject

    BaseObject.objs = extensions['objs']
    from OmniNav.core.robot.robot import BaseRobot

    BaseRobot.robots = extensions['robots']
    from OmniNav.core.sensor.sensor import BaseSensor

    BaseSensor.sensors = extensions['sensors']
    from OmniNav.core.task import BaseTask

    BaseTask.tasks = extensions['tasks']
