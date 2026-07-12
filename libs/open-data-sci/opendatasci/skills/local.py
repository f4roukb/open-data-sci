from pathlib import Path

from opendatasci.skills.base import BaseSkillStore, Skill, SkillDomain

_BUILTIN_SKILLS_DIRECTORY = Path(__file__).resolve().parents[1] / "resources" / "skills"
_BUILTIN_DOMAINS_DIRECTORY = Path(__file__).resolve().parents[1] / "resources" / "skill_domains"

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


def _parse_skill_file(file: Path) -> tuple[str, str] | None:
    """Return ``(name, content)`` for ``.md`` files, or ``None`` for all others.

    The filename stem is used as the skill name and the file body as the prompt
    content. Files with any other extension are silently skipped.
    """
    if file.suffix != ".md":
        return None
    return file.stem, file.read_text(encoding="utf-8")


class LocalSkillStore(BaseSkillStore):
    """Loads skills and skill domains from local filesystem directories.

    Layout convention:

    - A *skills* directory may contain standalone ``.md`` skill files directly
      (``<skills_dir>/<skill_name>.md``) and/or subdirectories that group the
      skills belonging to a skill domain (``<skills_dir>/<domain_name>/<skill_name>.md``).
      Skills that belong to a skill domain are keyed by the qualified name
      ``"<domain_name>::<skill_name>"``.
    - A *skill domain* directory contains one subdirectory per domain, each
      holding a ``manifest.md`` file (``<domains_dir>/<domain_name>/manifest.md``).

    Directories are scanned in order; later directories override earlier ones
    when names clash. Files with extensions other than ``.md`` are silently
    skipped. When *paths* / *domain_paths* is ``None``, only the built-in
    directories are scanned.

    Args:
        paths: Ordered list of skills directories to scan. ``None`` loads only
            the built-in skills bundled with the package.
        domain_paths: Ordered list of skill-domain directories to scan.
            ``None`` loads only the built-in domains bundled with the package.
    """

    def __init__(
        self,
        paths: list[Path] | None = None,
        domain_paths: list[Path] | None = None,
    ) -> None:
        self._paths: list[Path] = paths if paths is not None else [_BUILTIN_SKILLS_DIRECTORY]
        self._domain_paths: list[Path] = (
            domain_paths if domain_paths is not None else [_BUILTIN_DOMAINS_DIRECTORY]
        )

    # ------------------------------------------------------------------
    # BaseSkillStore
    # ------------------------------------------------------------------

    def load(self, name: str) -> Skill | None:
        return self.list_skills().get(name)

    def load_domain(self, name: str) -> SkillDomain | None:
        return self.list_domains().get(name)

    def list_skills(self) -> dict[str, Skill]:
        result: dict[str, Skill] = {}
        for d in self._paths:
            result.update(self._scan_skills_dir(d))
        return result

    def list_domains(self) -> dict[str, SkillDomain]:
        result: dict[str, SkillDomain] = {}
        for d in self._domain_paths:
            result.update(self._scan_domains_dir(d))
        return result

    def load_user_defined(self) -> dict[str, Skill]:
        """Return skills from all directories except the built-in skills directory."""
        result: dict[str, Skill] = {}
        for d in self._paths:
            if d == _BUILTIN_SKILLS_DIRECTORY:
                continue
            result.update(self._scan_skills_dir(d))
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan_skills_dir(self, path: Path) -> dict[str, Skill]:
        """Scan *path* for standalone skills and one level of skill-domain subdirectories."""
        if not path.is_dir():
            return {}

        result: dict[str, Skill] = self._load_skill_files(path)
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            result.update(self._load_skill_files(sub, prefix=sub.name))
        return result

    def _load_skill_files(self, path: Path, *, prefix: str | None = None) -> dict[str, Skill]:
        result: dict[str, Skill] = {}
        for file in sorted(path.iterdir()):
            if file.is_dir():
                continue
            parsed = _parse_skill_file(file)
            if parsed is None:
                continue
            base_name, content = parsed
            qualified = f"{prefix}::{base_name}" if prefix else base_name
            result[qualified] = Skill(name=qualified, content=content)
        return result

    def _scan_domains_dir(self, path: Path) -> dict[str, SkillDomain]:
        if not path.is_dir():
            return {}

        result: dict[str, SkillDomain] = {}
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            manifest_file = sub / "manifest.md"
            if not manifest_file.is_file():
                continue
            result[sub.name] = SkillDomain(
                name=sub.name, content=manifest_file.read_text(encoding="utf-8")
            )
        return result
