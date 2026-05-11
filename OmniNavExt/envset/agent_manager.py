# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import importlib
from typing import Any, Tuple, Optional, Callable



def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _require_module(module_name: str, reason: str):
    module = _import_module(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _import_carb():
    return _import_module("carb")


def _require_carb(reason: str):
    carb = _import_carb()
    if not carb:
        raise ModuleNotFoundError(
            f"carb is not available; {reason}. Initialize the simulator before calling."
        )
    return carb


def _import_omni(module_name: str):
    return _import_module(module_name)


def _require_omni(module_name: str, reason: str):
    module = _import_omni(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _import_pxr(module_name: str):
    return _import_module(module_name)


def _require_pxr(module_name: str, reason: str):
    module = _import_pxr(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _log_info(message: str):
    carb = _import_carb()
    if carb and hasattr(carb, "log_info"):
        carb.log_info(message)
    else:
        print(message)


def _log_warn(message: str):
    carb = _import_carb()
    if carb and hasattr(carb, "log_warn"):
        carb.log_warn(message)
    else:
        print(message)


def _log_error(message: str):
    carb = _import_carb()
    if carb and hasattr(carb, "log_error"):
        carb.log_error(message)
    else:
        print(message)


class AgentData:
    """
    Class that helps match character_label with primpath and skelroot path,
    allowing for fixed parameters and additional user-defined metadata.
    """

    # define several basic attribute:
    FIXED_ATTRIBUTES = {"label_path", "prim_path"}

    def __init__(
        self,
        label_path: Optional[str] = None,
        prim_path: Optional[str] = None,
        **kwargs,
    ):
        # Dynamically initialize fixed attributes
        for attr in self.FIXED_ATTRIBUTES:
            setattr(self, attr, locals()[attr])  # Set attributes dynamically

        # Metadata for additional attributes
        self.metadata = kwargs

    def __setattr__(self, name, value):
        """Check if the attribute is one of the fixed attributes"""
        if name in self.FIXED_ATTRIBUTES or name == "metadata":
            super().__setattr__(name, value)
        else:
            # Store additional attributes in metadata
            if value is not None:
                self.metadata[name] = value

    def __getattr__(self, name):
        """Allow access to metadata as if they were attributes"""
        return self.metadata.get(name, None)

    def get_metadata(self):
        """get all metadata of agent"""
        return self.metadata


class AgentManager:
    """Global class which stores current and predicted positions of all agents and moving objects."""

    __instance: AgentManager = None

    def __init__(self):
        if self.__instance is not None:
            raise RuntimeError("Only one instance of AgentManager is allowed")

        # This maps agent name with its BehaviorScript instance
        # Agents are register when Simulation starts, and are deregistered when simulation ends
        self._agent_name_to_script_inst = {}

        self._agent_registered_sub = None  # add subscription to agent register event
        self._metadata_updated_sub = None  # add subscription to metadata updated event
        self._stage_closing_event_sub = None
        self._stage_opened_event_sub = None
        self._stage_animation_stop_event_sub = None

        # fetching data for writers:
        # Create 3 dictionaries, allow user to search character in different way
        self.agents_by_label_path: dict[str, AgentData] = {}
        self.agents_by_primpath: dict[str, AgentData] = {}
        self.agents_by_skelpath: dict[str, AgentData] = {}
        # the design to record agent's action tag
        self.agents_by_agentname: dict[str, AgentData] = {}

        AgentManager.__instance = self
        self.register_event_function()

    def register_event_function(self):
        """
        register event functions for when an agent is registered and when its metadata is updated
        clean the registry when the simulation stops
        """
        carb = _require_carb("registering AgentManager events")
        AgentEvent = _require_omni(
            "omni.anim.people.settings",
            "registering AgentEvent listeners",
        ).AgentEvent
        omni_usd = _require_omni("omni.usd", "subscribing to stage events")
        # subscription to agent register event
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        print("[AgentManager][DEBUG] Registering AgentEvent.AgentRegistered observer...")
        self._agent_registered_sub = dispatcher.observe_event(
            event_name=AgentEvent.AgentRegistered,
            on_event=self.on_agent_registered,
            observer_name="isaacsim/replicator/agent/ON_AGENT_REGISTERED"
        )
        print(f"[AgentManager][DEBUG] AgentRegistered observer handle: {self._agent_registered_sub}")
        # subscription to metadata update event
        print("[AgentManager][DEBUG] Registering AgentEvent.MetadataUpdateEvent observer...")
        self._metadata_updated_sub = dispatcher.observe_event(
            event_name=AgentEvent.MetadataUpdateEvent,
            on_event=self.on_metadata_updated,
            observer_name="isaacsim/replicator/agent/ON_METADATA_UPDATED"
        )

        self._usd_context = omni_usd.get_context()
        if self._usd_context is not None and self._stage_closing_event_sub is None:
            self._stage_closing_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=self._usd_context.stage_event_name(omni_usd.StageEventType.CLOSING),
                on_event=self.__on_stage_event,
                observer_name="isaacsim/replicator/agent/CLEAN_REGISTERED_AGENT",
            )
            self._stage_opened_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=self._usd_context.stage_event_name(omni_usd.StageEventType.OPENED),
                on_event=self.__on_stage_event,
                observer_name="isaacsim/replicator/agent/CLEAN_REGISTERED_AGENT",
            )
            self._stage_animation_stop_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=self._usd_context.stage_event_name(omni_usd.StageEventType.SIMULATION_STOP_PLAY),
                on_event=self.__on_stage_event,
                observer_name="isaacsim/replicator/agent/CLEAN_REGISTERED_AGENT",
            )

    def clean_all_event(self):
        """remove all registered events"""
        self._bus = None
        self._agent_registered_sub = None
        self._metadata_updated_sub = None
        self._stage_closing_event_sub = None
        self._stage_opened_event_sub = None
        self._stage_animation_stop_event_sub = None

    def destroy(self):
        self.clear_agent()
        self.clean_all_event()
        self.clear_agent_data_dicts()

        AgentManager.__instance = None

    def __del__(self):
        self.destroy()

    @classmethod
    def get_instance(cls) -> AgentManager:
        if cls.__instance is None:
            AgentManager()
        return cls.__instance

    @classmethod
    def has_instance(cls) -> bool:
        if cls.__instance is None:
            return False
        return True

    def __on_stage_event(self, event):
        """at the end of simulation or stage is changed, clean all registered agent and agent info"""
        # clean all the registered agent
        self.clear_agent()
        # clean all character_data
        self.clear_agent_data_dicts()
        

    def clear_agent(self):
        self._agent_name_to_script_inst.clear()

    def on_agent_registered(self, e):
        """ "
        This function would be triggered when agent instance are created.
        It will register the agent to the manager
        """
        agent_info = e.payload
        # check whether agent info has content
        if agent_info is None:
            return
        # check whether agent name is correct
        agent_name = agent_info["agent_name"]
        agent_prim_path = agent_info["prim_path"]
        print(
            f"[AgentManager] on_agent_registered event: agent={agent_name}, prim={agent_prim_path}, "
            f"metadata={agent_info}"
        )
        self.register_agent(agent_name, agent_prim_path)
        _log_info(f"{agent_name} is registered with prim path {agent_prim_path}")

    def on_metadata_updated(self, e):
        """ "
        This function would be triggered when agent's metadata are updated.
        It will update the metadata tag in the dictionary
        """
        agent_info = e.payload
        # check whether agent info has content
        if agent_info is None:
            return
        # check whether agent name is correct
        agent_name = agent_info["agent_name"]
        data_name = agent_info["data_name"]
        data_value = agent_info["data_value"]
        self.set_metadata_value(agent_name=agent_name, data_name=data_name, data_value=data_value)

    def register_agent(self, agent_name, agent_prim_path):
        """Register the agent to the manager by creating mapping for agent name to its BehaviorScript instance"""
        # get the BehaviorScript inst
        SimulationUtil = _require_omni(
            "omni.metropolis.utils.simulation_util",
            "registering agents",
        ).SimulationUtil
        agent_inst = SimulationUtil.get_agent_script_instance_by_path(agent_prim_path)
        if agent_inst is None:
            print(
                f"[AgentManager][DEBUG] register_agent called but no BehaviorScript instance for {agent_name} "
                f"at {agent_prim_path}. Script initialization may still be pending."
            )
        else:
            print(
                f"[AgentManager][DEBUG] register_agent succeeded for {agent_name} "
                f"({agent_inst}) at {agent_prim_path}"
            )
        # add the agent inst to the dict
        self._agent_name_to_script_inst[agent_name] = agent_inst

    def agent_registered(self, agent_name) -> bool:
        """Check whether the given agent is registered in the dict"""
        if agent_name not in self._agent_name_to_script_inst.keys():
            _log_warn(f"Agent is not registered to Agent Manager: {agent_name}")
            return False
        return True

    def deregister_agent(self, agent_name):
        """Remove the agent from the agent script dict"""
        if self.agent_registered(agent_name):
            self._agent_name_to_script_inst[agent_name] = None

    def get_agent_script_instance_by_name(self, agent_name):
        """Get the agent behavior script by its name"""
        if self.agent_registered(agent_name):
            return self._agent_name_to_script_inst[agent_name]
        return None

    def get_agent_pos_by_name(self, name):
        """Get the agent position by its name"""
        agent = self.get_agent_script_instance_by_name(name)
        if agent:
            return agent.get_current_position()
        else:
            _log_error("Agent: {name} does not exist".format(name=name))
            return None

    def get_agent_name_list_with_injected_commands(self, command_list):
        """Get the list of agents that have injected commands"""
        agent_name_list = []
        for command in command_list:
            agent_name_list.append(str(command).strip().split(" ")[0])
        return agent_name_list

    def get_all_agent_names(self):
        """fetch the name of all agents"""
        return self._agent_name_to_script_inst.keys()

    def inject_command_for_all_agents(self, command_list, force_inject):
        """Inject command for all agents"""
        agent_list = self.get_agent_name_list_with_injected_commands(command_list)
        for agent_name, agent_inst in self._agent_name_to_script_inst.items():
            # if agent inst does exist, inject function to the agent
            if str(agent_name) in agent_list and agent_inst is not None:
                self.inject_command(str(agent_name), command_list, force_inject)

    def inject_command(
        self,
        agent_name,
        command_list,
        force_inject=False,
        instant=True,
        on_finished: Tuple[str, Callable[[str, str], None]] = None,
    ):
        """Inject command to target agent"""
        agent_obj = self.get_agent_script_instance_by_name(agent_name)
        if agent_obj is None:
            _log_warn(f"Fail to inject command to {agent_name}. Agent is not registered to Agent Manager")
            return

        # force inject will interrupt current command and immediately inject the new commands
        if force_inject and instant:
            agent_obj.end_current_command()
        # inject command to the agent        # TODO:: Find a better way to check if an agent is character or robot
        PrimPaths = _require_module(
            "OmniNavExt.envset.settings",
            "resolving agent prim path prefixes",
        ).PrimPaths
        if str(agent_obj.prim_path).startswith(PrimPaths.characters_parent_path()):
            agent_obj.inject_command(command_list=command_list, executeImmediately=instant, on_finished=on_finished)
        elif str(agent_obj.prim_path).startswith(PrimPaths.robots_parent_path()):
            # RobotBehavior don't have on_finished callback support yet
            agent_obj.inject_command(command_list=command_list, executeImmediately=instant)
        else:
            _log_warn(f"Unsupported agent type {type(agent_obj)} during inject command.")

    def replace_command(self, agent_name, command_list, on_finished: Tuple[str, Callable[[str, str], None]] = None):
        """Replace command to target agent"""
        agent_obj = self.get_agent_script_instance_by_name(agent_name)
        if agent_obj is None:
            _log_warn(f"Fail to replace command to {agent_name}. Agent is not registered to Agent Manager")
            return

        PrimPaths = _require_module(
            "OmniNavExt.envset.settings",
            "resolving agent prim path prefixes",
        ).PrimPaths
        if str(agent_obj.prim_path).startswith(PrimPaths.characters_parent_path()):
            agent_obj.replace_command(command_list=command_list, on_finished=on_finished)
        else:
            _log_warn(f"Unsupported agent type {type(agent_obj)} during replace command.")

    def extract_agent_semantic_prim_path(self, agent_prim: Usd.Prim) -> str:
        """extract target agent semantic info from target prim"""
        Usd = _require_pxr("pxr.Usd", "extracting agent semantic prim path")
        OmniScriptingSchema = _require_module(
            "OmniScriptingSchema",
            "checking for OmniScriptingAPI on prims",
        )

        # The behavior script always attached on the prim that has transparents updated correctly.
        # Therefore, the semantic lables are usually attached on the same prim path
        for agent_name, agent_script_instance in self._agent_name_to_script_inst.items():
            script_instance_path = str(agent_script_instance.prim_path)
            if script_instance_path.startswith(str(agent_prim.GetPrimPath())):
                return script_instance_path

        # catch the edge case, the agent script instance fail to be registered before the data generation
        for sub_prim in Usd.PrimRange(agent_prim):
            if sub_prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI):
                return str(sub_prim.GetPrimPath())



    def extract_agent_data(self):
        """Store all character semantic label, primpath, and skelpath to dicts"""
        Usd = _require_pxr("pxr.Usd", "extracting agent data")
        USDUtil = _require_omni(
            "omni.metropolis.utils.usd_util",
            "extracting agent data",
        ).USDUtil
        stage_util = _require_module(
            "OmniNavExt.envset.stage_util",
            "extracting agent data",
        )
        CharacterUtil = stage_util.CharacterUtil
        RobotUtil = stage_util.RobotUtil
        # get current character prim list in the stage
        # refresh the data
        self.clear_agent_data_dicts()
        character_prim_list = CharacterUtil.get_characters_root_in_stage()
        robot_prim_list = RobotUtil.get_robots_in_stage()

        # collect all agent prim in the stage
        agent_prim_list = []
        agent_prim_list.extend(character_prim_list)
        agent_prim_list.extend(robot_prim_list)

        # iterate through agent prim
        for agent_prim in agent_prim_list:
            # extract agent's url path
            agent_name = str(agent_prim.GetName())
            asset_url = USDUtil.get_object_reference(prim=agent_prim)
            # check whether the target prim is a reference object:
            # if so, store the reference url in agent data
            semantic_attach_path = self.extract_agent_semantic_prim_path(agent_prim=agent_prim)
            # extract agent's skeleton path # check whether this agent has skeleton.
            skeleton_path = None
            # iterate all child prims
            for prim_child in Usd.PrimRange(agent_prim):
                # check whether the prim is a skeleton type
                if prim_child.GetTypeName() == "Skeleton":
                    skeleton_prim = prim_child
                    # get agent's skeleton prim path
                    skeleton_path = str(skeleton_prim.GetPrimPath())
            # extract agent's prim path
            agent_prim_path = str(agent_prim.GetPrimPath())
            agent_data = AgentData(
                label_path=semantic_attach_path,
                prim_path=agent_prim_path,
                skelpath=skeleton_path,
                asset_url=asset_url,
            )
            # register the agent data structure in three dictionary.
            self.agents_by_label_path[semantic_attach_path] = agent_data
            self.agents_by_primpath[agent_prim_path] = agent_data
            self.agents_by_agentname[agent_name] = agent_data
            # match the agent with skeleton path
            if skeleton_path is not None:
                self.agents_by_skelpath[skeleton_path] = agent_data

    ## different way to query agent status.
    def get_agent_data_by_prim_path(self, prim_path: str) -> AgentData | None:
        # get agent status via prim path
        return self.agents_by_primpath.get(prim_path)

    # This method need agents to have unique semantic tag/label
    def get_agent_data_by_label_path(self, label_path: str) -> AgentData | None:
        """get agents status via label"""
        return self.agents_by_label_path.get(label_path, None)

    def get_agent_data_by_skelpath(self, skelpath: str) -> AgentData | None:
        """get agents status via skelpath"""
        return self.agents_by_skelpath.get(skelpath, None)

    def is_agent_semantic_prim_path(self, semantic_prim_path: str) -> bool:
        """check whether certain prim path point to a agent"""
        return semantic_prim_path in self.agents_by_label_path

    def clear_agent_data_dicts(self):
        """clean all data stored in agent data dicts"""
        self.agents_by_label_path.clear()
        self.agents_by_primpath.clear()
        self.agents_by_primpath.clear()
        self.agents_by_agentname.clear()

    def set_metadata_value(self, agent_name: str, data_name: str, data_value: Any):
        """set agent's metadata"""
        # check whether agent's metadata exist
        agent_metadata = self.get_agent_metadata_dict(agent_name=agent_name)

        if agent_metadata is None:
            _log_warn(
                f"Failed to add {data_name}:{str(data_value)} to agent metadata: "
                f"{agent_name} is not a valid character name"
            )
            return

        agent_metadata[data_name] = data_value

    def get_agent_metadata_dict(self, agent_name):
        """return the agent metadata dict"""
        if agent_name not in self._agent_name_to_script_inst.keys():
            _log_info(f"Warning: Failed to feature target agent '{agent_name}' in Info.")
            return None

        agent_data = self.agents_by_agentname.get(agent_name, None)
        if agent_data is None:
            return None

        return agent_data.get_metadata()

    def get_metadata_value(self, agent_name: str, data_name: str) -> str | None:
        """get agent's metadata value"""
        agent_metadata_dict = self.get_agent_metadata_dict(agent_name=agent_name)
        if agent_metadata_dict is None:
            return None

        return agent_metadata_dict.get(data_name, None)

    def get_agent_position(self, agent_name):
        """get agent's location in the stage, no matter whether simulation has been started"""
        if agent_name not in self._agent_name_to_script_inst.keys():
            _log_info(" Warning as message :: agent is not registered in the omni.anim.people ")
            # fetch agent name from the stage directly
            CharacterUtil = _require_module(
                "OmniNavExt.envset.stage_util",
                "reading character positions",
            ).CharacterUtil
            character_prim_list = CharacterUtil.get_characters_root_in_stage()
            for character_prim in character_prim_list:
                if character_prim.GetName() == agent_name:
                    character_position = CharacterUtil.get_character_pos(character_prim)
                    return character_position

            return None

        return self.get_agent_pos_by_name(agent_name)
