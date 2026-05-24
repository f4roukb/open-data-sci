import json
import logging
from pathlib import Path

from opendatasci.skills.base import BaseSkillStore, Skill

logger = logging.getLogger(__name__)

_BUILTIN_SKILLS_DIRECTORY = Path(__file__).resolve().parents[2] / "resources" / "skills"

_BUILTIN_NAMES = [
    "data_science",
    "competitive_data_science",
    "machine_learning",
    "deep_learning",
    "quantitative_analysis",
    "data_science_education",
]

SKILL_LABELS: dict[str, str] = {
    "data_science": "Data Scientist",
    "competitive_data_science": "Competitive Data Scientist",
    "machine_learning": "Machine Learning Eng.",
    "quantitative_analysis": "Quantitative Analyst",
    "data_science_education": "Data Science Educator",
    "deep_learning": "Deep Learning Eng.",
}


class LocalSkillStore(BaseSkillStore):
    """Loads skills from one or more local filesystem directories.

    Directories are scanned in order; later directories override earlier ones when
    skill names clash.  Each directory may contain:

    - ``.md`` files — loaded directly (filename stem used as skill name).
    - ``.yaml`` / ``.yml`` / ``.json`` files — parsed for ``name`` and ``prompt`` keys.

    When *paths* is ``None``, only the built-in skills directory is scanned.

    Args:
        paths: Ordered list of directories to scan.  ``None`` loads only the
            built-in skills bundled with the package.
        strict: When ``True``, raise ``ValueError``
            instead of warning if any structured skill file cannot be parsed.
    """

    def __init__(
        self,
        paths: list[Path] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self._paths: list[Path] = paths if paths is not None else [_BUILTIN_SKILLS_DIRECTORY]
        self._strict = strict

    def load(self, name: str) -> Skill | None:
        return self.list().get(name)

    def list(self) -> dict[str, Skill]:
        result: dict[str, Skill] = {}
        for d in self._paths:
            result.update(self._load_from_dir(d))
        return result

    def load_user_defined(self) -> dict[str, Skill]:
        """Return skills from all directories except the built-in skills directory."""
        result: dict[str, Skill] = {}
        for d in self._paths:
            if d != _BUILTIN_SKILLS_DIRECTORY:
                result.update(self._load_from_dir(d))
        return result

    def _load_from_dir(self, path: Path) -> dict[str, Skill]:
        if not path.is_dir():
            return {}

        result: dict[str, Skill] = {}
        failures: list[tuple[Path, str]] = []

        for file in sorted(path.iterdir()):
            if file.suffix == ".md":
                result[file.stem] = Skill(name=file.stem, content=file.read_text(encoding="utf-8"))
                continue

            if file.suffix not in {".yaml", ".yml", ".json"}:
                continue

            try:
                if file.suffix == ".json":
                    data = json.loads(file.read_text(encoding="utf-8"))
                else:
                    try:
                        import yaml  # type: ignore[import-untyped]

                        data = yaml.safe_load(file.read_text(encoding="utf-8"))
                    except ImportError:
                        data = json.loads(file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                failures.append((file, f"parse error: {exc}"))
                continue

            if not isinstance(data, dict):
                failures.append(
                    (file, f"expected a mapping at the top level, got {type(data).__name__}")
                )
                continue

            name = data.get("name")
            prompt = data.get("prompt")
            missing = [k for k, v in (("name", name), ("prompt", prompt)) if not v]
            if missing:
                failures.append((file, f"missing required key(s): {', '.join(missing)}"))
                continue

            result[str(name)] = Skill(name=str(name), content=str(prompt))

        if failures:
            summary = "; ".join(f"{f.name}: {reason}" for f, reason in failures)
            if self._strict:
                raise ValueError(
                    f"{len(failures)} skill file(s) in '{path}' could not be loaded: {summary}"
                )
            logger.warning(
                "%d skill file(s) in '%s' could not be loaded: %s",
                len(failures),
                path,
                summary,
            )

        return result
