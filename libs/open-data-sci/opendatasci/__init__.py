"""
OpenDataSci — AI-powered data analytics SDK.

Quick start::

    from opendatasci import OpenDataSciConfig, create_agent

    async with create_agent("/data/sales.csv") as agent:
        async for event in agent.astream("What is the average revenue by region?"):
            print(event)

Package layout::

    opendatasci/
        configs.py      OpenDataSciConfig — all settings in one place
        agents/         agent orchestration: graph, state, memory, streaming
        workspace/      BaseWorkspace ABC and LocalWorkspace implementation
        context/        dataset context and session plan stores
        models/         LLM provider factory and per-provider adapters
        sandbox/        sandbox abstraction and SRT-backed implementation
        skills/         skill loading and registry
        streaming/      AgentStreamEvent types and stream processors
        tools/          LangChain tools available to the agent
"""

from opendatasci.agents.agents import Agent
from opendatasci.agents.agents_factory import create_agent
from opendatasci.agents.chat_memory import PreparedHistory
from opendatasci.configs import OpenDataSciConfig
from opendatasci.sandbox.base import SandboxExecResult
from opendatasci.streaming.events import AgentStreamEvent
from opendatasci.workspace import LocalWorkspace

__version__ = "0.1.0"

__all__ = [
    # Workspace
    "LocalWorkspace",
    # Agent layer
    "Agent",
    "AgentStreamEvent",
    "SandboxExecResult",
    "create_agent",
    "PreparedHistory",
    # Configuration
    "OpenDataSciConfig",
]
