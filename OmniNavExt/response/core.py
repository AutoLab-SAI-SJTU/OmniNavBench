# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations
import carb
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict
import omni.timeline
from omni.metropolis.utils.triggers.core import TriggerBase, TriggersManager
from omni.metropolis.utils.carb_util import CarbUtil
from OmniNavExt.envset.agent_manager import AgentManager
from OmniNavExt.configs.scenes.defines import ResponseProperty, CommandResponseProperty, ResponseSection

from isaacsim.replicator.incident.incident_trigger import IncidentTrigger
import omni.anim.navigation.core as nav

"""

Agent Response module.

    - [AgentResponseManager] tracks all active responses and assign them to agents.
    - Each reponse has a prioirty. Higher priority response will interrupt the current response.
        - If incoming response sets resume to True, current response will be queued.
        - If incoming response sets resume to False, all responses on that agent will be cleared.
    - Each response can involve multiple agents. Response ends when all agents finish.
    - Each agent can have multiple active responses and will finish them by prioirty.

Example Usage:

    import carb
    from isaacsim.replicator.agent.core.agent_reponse.core import *
    cmd_response = CommandResponse(
        name = "check incident",
        priority = 1,
        pick_agent = ResponsePickAgent.NEAREST,
        resume = True,
        position = carb.Float3(0, 0, 0),
        commands = ["GoToResponse", "LookAround 5"]
    )
    manager = AgentResponseManager.get_instance().trigger_response(cmd_response)

"""


class ResponsePickAgent(str, Enum):
    ALL = "all"
    FIRST_AVAILABLE = "first_available"
    NEAREST = "nearest"
    FURTHEST = "furthest"


@dataclass
class ResponseBase:
    name: str = ""  # Response name. Responses should have different names.
    priority: int = 1  # Response prioirty. Used in multiple response handling.
    pick_agent: ResponsePickAgent = ResponsePickAgent.FIRST_AVAILABLE  # The rule to pick agents for response.
    resume: bool = True  # Should agent resumes its next action after the response.
    position: carb.Float3 = field(
        default_factory=lambda: carb.Float3(0, 0, 0)
    )  # Event position. Used for distance-based agent picking.

    def __eq__(self, other):
        if isinstance(other, ResponseBase):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


@dataclass
class CommandResponse(ResponseBase):
    name: str = "Default Command Response"  # Overwrite base class default
    commands: List[str] = field(default_factory=lambda: [])

    def __hash__(self):
        return hash(tuple(self.commands))


class AgentResponseManager:
    """
    Singleton class to dispatch and keep track of agent responses.
    """

    __instance: AgentResponseManager = None

    @classmethod
    def get_instance(cls) -> AgentResponseManager:
        if cls.__instance is None:
            AgentResponseManager()
        return cls.__instance

    def __init__(self):
        AgentResponseManager.__instance = self

        # Responses that are added but not yet triggered
        self.idle_response_list: List[ResponseBase] = []

        # Active responses and corresponding agent (names)
        self.active_response_agent_dict: Dict[ResponseBase, List[str]] = {}

        # Triggers for added idle responses
        self.triggers: Dict[str, TriggerBase] = {}

        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_event_handler = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

    def _on_timeline_event(self, event):
        if event.type in (omni.timeline.TimelineEventType.STOP.value, omni.timeline.TimelineEventType.PLAY.value):
            self.active_response_agent_dict.clear()

    def reset(self):
        self.idle_response_list.clear()
        self.active_response_agent_dict.clear()
        for trigger in self.triggers.values():
            trigger.destroy()
        self.triggers.clear()

    def get_agent_active_response(self, agent_name: str) -> List[ResponseBase]:
        result = []
        for response, agent_name_list in self.active_response_agent_dict.items():
            if agent_name in agent_name_list:
                result.append(response)
        return result

    def setup_responses_from_config_file(self, response_section: ResponseSection):
        """
        Set up response from config file.
        """
        response_list = response_section.get_property_group("response_list")
        if not response_list:
            carb.log_warn("Unable to get response_list from response section. Set up response fails.")
            return
        for response_prop in response_list.data_group:
            data_dict = response_prop.get_resolved_value()
            data_dict_no_trigger = data_dict.copy()
            del data_dict_no_trigger["trigger"]  # Remove trigger dict info

            response: ResponseBase = None
            if isinstance(response_prop, CommandResponseProperty):
                response = CommandResponse(**data_dict_no_trigger)

            if response:
                self.idle_response_list.append(response)
                carb.log_info(f"Register response '{response.name}'.")
            else:
                carb.log_warn(f"Unable to create response for {response_prop.name}.")
                continue

            trigger = TriggersManager.get_instance().create_trigger_by_dict(data_dict)
            if trigger:
                trigger.add_callback(lambda t, r=response: self.trigger_response(r, t))
                self.triggers[response.name] = trigger
                carb.log_info(f"Set up trigger '{trigger.to_dict()}' for response '{response.name}'.")
            else:
                carb.log_warn(f"Response '{response.name}' dose not have a trigger.")

    def trigger_response(self, response: ResponseBase, trigger: TriggerBase = None) -> bool:
        """
        Trigger a response immediately.
        """
        if response in self.active_response_agent_dict.keys():
            carb.log_warn(f"Response '{response.name}' is already active. Will not trigger again.")
            return False
        if isinstance(response, CommandResponse):
            return self._trigger_command_response(response, trigger)
        else:
            carb.log_error(f"Object '{response}' does not have a Response implementation.")
            return False

    def _trigger_command_response(self, response: CommandResponse, trigger: TriggerBase = None) -> bool:
        # Temp: special handle for incident trigger
        # TODO:: estabilish a formal data exchange rule to decouple with IRI
        if trigger and isinstance(trigger, IncidentTrigger):
            response.position = trigger.incident_data.event_position
        # Pre-process for special command
        for idx, cmd in enumerate(response.commands):
            if cmd == "GoToResponse":
                pos = response.position

                navmesh = nav.acquire_interface().get_navmesh()
                if not navmesh:
                    carb.log_error("No navmesh is available")
                    return

                pos = navmesh.query_closest_point(pos)

                if not pos:
                    carb.log_error("Couldn't get a closest navmesh point!")
                    return

                response.commands[idx] = f"GoTo {pos[0]} {pos[1]} {pos[2]} _"
        # Pick agent
        agent_name_list = self._pick_agent(response)
        if not agent_name_list:
            carb.log_warn(f"Unbale to find agents for response '{response.name}'. Trigger response fails.")
            return False
        # Keep track of agents and their current responses
        self.active_response_agent_dict[response] = []
        agent_manager = AgentManager.get_instance()
        for agent_name in agent_name_list:
            # Append command with agent name
            commands = [f"{agent_name} {cmd}" for cmd in response.commands]
            agent_manager.get_agent_script_instance_by_name(agent_name)
            if response.resume:
                agent_manager.inject_command(
                    agent_name=agent_name,
                    command_list=commands,
                    force_inject=True,
                    instant=True,
                    on_finished=(response.name, self._on_agent_finish_response),
                )
            else:
                agent_manager.replace_command(
                    agent_name=agent_name,
                    command_list=commands,
                    on_finished=(response.name, self._on_agent_finish_response),
                )

            # Track response
            self.active_response_agent_dict[response].append(agent_name)

        return True

    def _on_agent_finish_response(self, response_name: str, agent_name: str):
        """
        Remove agent from Response agent list, end Response if agent list is empty.
        """
        response_to_remove: List[ResponseBase] = []
        for response, agent_name_list in self.active_response_agent_dict.items():
            if response_name == response.name:
                if agent_name in agent_name_list:
                    agent_name_list.remove(agent_name)
                    carb.log_info(f"Agent {agent_name} has finished response {response_name}.")
                    if len(agent_name_list) == 0:
                        response_to_remove.append(response)
                        carb.log_info(f"Response {response_name} has finished.")

        for response in response_to_remove:
            self.active_response_agent_dict.pop(response)

    def _pick_agent(self, response: ResponseBase) -> List[str]:
        """
        Pick an agent to respond, return a list of agent names.
        """
        agent_manager = AgentManager.get_instance()
        agent_name_list = agent_manager.get_all_agent_names()

        # Filter agents that are free or in lower priority response
        valid_agent_names = []
        for name in agent_name_list:
            response_list = self.get_agent_active_response(name)
            if (not response_list) or all(res.priority < response.priority for res in response_list):
                valid_agent_names.append(name)

        if response.pick_agent == ResponsePickAgent.ALL:
            return valid_agent_names

        if response.pick_agent == ResponsePickAgent.FIRST_AVAILABLE:
            if valid_agent_names:
                return [valid_agent_names[0]]
            else:
                return []

        # Distance-based pick rules
        min_dist = sys.float_info.max
        min_dist_agent = valid_agent_names[0]
        max_dist = 0
        max_dist_agent = valid_agent_names[0]
        for agent in valid_agent_names:
            pos = agent_manager.get_agent_pos_by_name(agent)
            dist = CarbUtil.dist3(response.position, pos)
            if dist < min_dist:
                min_dist = dist
                min_dist_agent = agent
            if dist > max_dist:
                max_dist = dist
                max_dist_agent = agent

        if response.pick_agent == ResponsePickAgent.NEAREST:
            return [min_dist_agent]

        if response.pick_agent == ResponsePickAgent.FURTHEST:
            return [max_dist_agent]

        return None
