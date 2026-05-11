import omni.kit
import omni.timeline
import carb.eventdispatcher
from isaacsim.replicator.incident.extension import get_instance
from omni.metropolis.utils.config_file.core import ConfigFile
from isaacsim.replicator.incident.incident_manager import IncidentManager

from omni.anim.people.python_ext import add_dynamic_obstacle_behavior_script

class IncidentBridge:
    """Bridge between IRA and IRI."""

    def __init__(self):
        incident_ext = get_instance()
        self._incident_manager: IncidentManager = incident_ext.create_incident_manager()

        self._stage_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
            on_event=self._on_stage_event,
            observer_name="isaacsim/replicator/agent/core/incident_bridge/ON_STAGE_EVENT",
        )

    def destroy(self):
        self._incident_manager = None

        self._timeline_sub.unsubscribe()
        self._timeline_sub = None

        self._stage_sub.unsubscribe()
        self._stage_sub = None

    def setup_incident_from_config(self, config_file: ConfigFile):
        if not config_file:
            return

        seed_prop = config_file.get_property("global", "seed")
        event_section = config_file.get_section("event")

        if not (seed_prop and event_section):
            return

        self._incident_manager.setup_incidents_from_config_file(seed_prop.get_resolved_value(), event_section)

        dynamic_obstacles = self._incident_manager.get_dynamic_obstacle_prim_paths()
        for dynamic_obstacle in dynamic_obstacles:
            add_dynamic_obstacle_behavior_script(dynamic_obstacle)

    def start_recording(self, dir_path: str):
        self._incident_manager.get_incident_report().start_recording(dir_path)

    def end_recording(self):
        self._incident_manager.get_incident_report().end_recording()

    def generate_incident_report(self, dir_path: str):
        report = self._incident_manager.get_incident_report()
        report.generate_report_file(dir_path)

    def _on_stage_event(self, e: carb.events.IEvent):
        # Reset incident manager when stage changed
        self._incident_manager.reset_incident_managers()

    def is_recording(self)->bool:
        """Check whether the recording has been triggered."""
        return self._incident_manager.get_incident_report().is_recording()