# Skills

A **skill** is a named block of domain know-how that is injected into the agent's system prompt when it becomes active. Skills are organised into **skill domains** — maps of a broader task area that tell the agent which skills exist and when each applies.

The agent resolves skills at runtime via a `BaseSkillStore`. The default implementation, `LocalSkillStore`, loads skills from local filesystem directories.

## Built-in skills

The library ships with built-in skills and skill domains covering data science, machine learning, deep learning, competitive data science, quantitative analysis, data science education, and curated entry points into specific external sites (`kaggle.com`, `arxiv.org`, `huggingface.co`, `github.com`, `paperswithcode.com`, `finance.yahoo.com`). They are loaded automatically; no configuration is needed.

## Custom skills

`create_agent()` scans skills in three layers, in increasing order of precedence: the
bundled built-ins, then the current workspace's own `.opendatasci/skills/` directory
(`.opendatasci/skill_domains/` for skill domains), then an explicit
`OpenDataSciConfig.skills_directory` (or the `SKILLS_DIRECTORY` env var) override if set.
A name that clashes across layers resolves to the higher-precedence one.

The simplest way to add skills to a single workspace is to drop `.md` files straight into
its `.opendatasci/skills/` directory — no configuration needed. To share a skills directory
across workspaces, point `skills_directory` at it instead:

```
my_skills/
    finance.md          # standalone skill, keyed "finance"
    trading/
        execution.md    # domain skill, keyed "trading::execution"
        risk.md         # domain skill, keyed "trading::risk"
```

```python
from opendatasci import create_agent, OpenDataSciConfig

config = OpenDataSciConfig(skills_directory="my_skills/")

async with create_agent("data.csv", config=config) as agent:
    ...
```

Custom skills are merged with the built-in set and the workspace's own skills; a name that
exists at more than one layer resolves to the highest-precedence one (`skills_directory` >
workspace `.opendatasci/skills/` > built-in).

## Custom skill store

Subclass `BaseSkillStore` to load skills from any source (a database, remote store, etc.) and pass the instance to `Agent`:

```python
from opendatasci.skills.base import BaseSkillStore, Skill, SkillDomain
from opendatasci.agents.agents import Agent
from opendatasci import LocalWorkspace, OpenDataSciConfig

class RemoteSkillStore(BaseSkillStore):
    def load(self, name: str) -> Skill | None:
        # fetch from a remote store …
        ...

    def load_domain(self, name: str) -> SkillDomain | None: ...
    def list_skills(self) -> dict[str, Skill]: ...
    def list_domains(self) -> dict[str, SkillDomain]: ...

async with Agent(
    workspace=LocalWorkspace("data.csv"),
    skill_store=RemoteSkillStore(),
    config=OpenDataSciConfig(),
) as agent:
    ...
```

## Reference

::: opendatasci.skills.base.Skill
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.skills.base.SkillDomain
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.skills.base.BaseSkillStore
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.skills.local.LocalSkillStore
    options:
      show_root_heading: true
      show_source: false
