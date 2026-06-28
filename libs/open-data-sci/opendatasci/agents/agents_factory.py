import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from opendatasci.agents.agents import Agent
from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.local import LocalContextStore
from opendatasci.sandbox.srt import SRTSandboxFactory
from opendatasci.skills.local import LocalSkillStore
from opendatasci.workspace.local import LocalWorkspace


def create_agent(
    path: str,
    session_id: str | None = None,
    config: OpenDataSciConfig | None = None,
) -> Agent:
    """Return a fully wired :class:`Agent` for a local file or directory.

    The agent must be used as an async context manager so its sandbox is
    created and closed correctly::

        async with create_agent("/data/sales.csv") as agent:
            async for event in agent.astream("summarise the data"):
                ...

    Resolves the workspace, sandbox factory, skill store, and persistence
    stores from *path*, then constructs and returns an agent ready to enter.

    Args:
        path: Path to the data file or workspace directory to load.
        config: LLM provider and model settings. Falls back to the library
            default when omitted.
        session_id: Identifier for this session's context store. Generated
            automatically when omitted.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    workspace = LocalWorkspace(path)

    config = config or OpenDataSciConfig()
    workspace_path = workspace.get_reference()
    sandbox_factory = SRTSandboxFactory()
    context_store = LocalContextStore(Path(workspace_path))

    session_id = session_id or uuid.uuid4().hex

    # Skills are loaded in increasing order of precedence: the bundled built-in
    # skills, then the workspace's own ``.opendatasci/skills/`` directory, then an
    # explicit ``SKILLS_DIRECTORY`` override. Missing directories are skipped.
    workspace_skills_directory = context_store.root / "skills"
    paths: list[Path] = [config.builtin_skills_directory, workspace_skills_directory]
    if config.skills_directory is not None:
        paths.append(config.skills_directory)
    skill_store = LocalSkillStore(paths)

    checkpointer = MemorySaver()

    return Agent(
        workspace=workspace,
        session_id=session_id,
        sandbox_factory=sandbox_factory,
        skill_store=skill_store,
        context_store=context_store,
        config=config,
        checkpointer=checkpointer,
    )
