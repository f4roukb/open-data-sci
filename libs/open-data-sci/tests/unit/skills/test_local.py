"""Unit tests for opendatasci.skills (LocalSkillStore)."""


from pathlib import Path

from opendatasci.skills import Skill, SkillDomain
from opendatasci.skills.local import (
    _BUILTIN_DOMAINS_DIRECTORY,
    _BUILTIN_SKILLS_DIRECTORY,
    LocalSkillStore,
    SKILL_LABELS,
)

_BUILTIN_COMPETITIVE_SKILLS = {
    "competitive_data_science::reconnaissance",
    "competitive_data_science::eda",
    "competitive_data_science::validation",
    "competitive_data_science::baseline",
    "competitive_data_science::feature_engineering",
    "competitive_data_science::model_development",
    "competitive_data_science::hyperparameter_tuning",
    "competitive_data_science::ensembling",
    "competitive_data_science::final_submission",
    "competitive_data_science::phase_wiring",
}

_BUILTIN_DATA_SCIENCE_SKILLS = {
    "data_science::framing_the_problem",
    "data_science::exploratory_analysis",
    "data_science::data_quality_preparation",
    "data_science::causality_confounding",
    "data_science::granularity_aggregation",
    "data_science::statistical_testing",
    "data_science::modeling_evaluation",
    "data_science::communicating_findings",
}

_BUILTIN_DATA_SCIENCE_EDUCATION_SKILLS = {
    "data_science_education::diagnosing_understanding",
    "data_science_education::building_intuition",
    "data_science_education::structuring_explanations",
    "data_science_education::worked_examples_code",
    "data_science_education::calibrating_depth",
    "data_science_education::feedback_correction",
    "data_science_education::connecting_concepts",
}

_BUILTIN_DEEP_LEARNING_SKILLS = {
    "deep_learning::library_stack",
    "deep_learning::when_to_use",
    "deep_learning::sklearn_mlp",
    "deep_learning::jax_fundamentals",
    "deep_learning::flax_nnx",
    "deep_learning::optax",
    "deep_learning::training_loop",
    "deep_learning::architecture_selection",
    "deep_learning::regularisation_overfitting",
    "deep_learning::hyperparameter_tuning",
    "deep_learning::evaluation",
}

_BUILTIN_MACHINE_LEARNING_SKILLS = {
    "machine_learning::problem_framing",
    "machine_learning::splitting_strategy",
    "machine_learning::feature_engineering_selection",
    "machine_learning::model_selection_complexity",
    "machine_learning::hyperparameter_tuning",
    "machine_learning::overfitting_regularisation",
    "machine_learning::evaluation_diagnostics",
    "machine_learning::class_imbalance",
    "machine_learning::interpretability",
}

_BUILTIN_QUANTITATIVE_ANALYSIS_SKILLS = {
    "quantitative_analysis::problem_formulation",
    "quantitative_analysis::mathematical_statistical_foundations",
    "quantitative_analysis::time_series_signal_analysis",
    "quantitative_analysis::risk_uncertainty_quantification",
    "quantitative_analysis::optimisation",
    "quantitative_analysis::backtesting_empirical_validation",
    "quantitative_analysis::communicating_results",
}

_BUILTIN_KAGGLE_SKILLS = {
    "kaggle.com::competitions",
    "kaggle.com::datasets",
    "kaggle.com::prior_editions_research",
}

_BUILTIN_ARXIV_SKILLS = {
    "arxiv.org::searching",
    "arxiv.org::reading_papers",
    "arxiv.org::credibility",
}

_BUILTIN_HUGGINGFACE_SKILLS = {
    "huggingface.co::leaderboards",
}

_BUILTIN_GITHUB_SKILLS = {
    "github.com::repository_reconnaissance",
    "github.com::issues_and_prs",
    "github.com::code_and_repo_search",
}

_BUILTIN_PAPERSWITHCODE_SKILLS = {
    "paperswithcode.com::sota_leaderboards",
}

_BUILTIN_FINANCE_YAHOO_SKILLS = {
    "finance.yahoo.com::yfinance_basics",
}

_BUILTIN_DOMAIN_NAMES = {
    "competitive_data_science",
    "data_science",
    "data_science_education",
    "deep_learning",
    "machine_learning",
    "quantitative_analysis",
    "kaggle.com",
    "arxiv.org",
    "huggingface.co",
    "github.com",
    "paperswithcode.com",
    "finance.yahoo.com",
}


class TestBuiltinSkills:
    def test_skills_registry_has_all_expected_keys(self) -> None:
        expected = (
            _BUILTIN_COMPETITIVE_SKILLS
            | _BUILTIN_DATA_SCIENCE_SKILLS
            | _BUILTIN_DATA_SCIENCE_EDUCATION_SKILLS
            | _BUILTIN_DEEP_LEARNING_SKILLS
            | _BUILTIN_MACHINE_LEARNING_SKILLS
            | _BUILTIN_QUANTITATIVE_ANALYSIS_SKILLS
            | _BUILTIN_KAGGLE_SKILLS
            | _BUILTIN_ARXIV_SKILLS
            | _BUILTIN_HUGGINGFACE_SKILLS
            | _BUILTIN_GITHUB_SKILLS
            | _BUILTIN_PAPERSWITHCODE_SKILLS
            | _BUILTIN_FINANCE_YAHOO_SKILLS
        )
        assert set(LocalSkillStore().list_skills().keys()) == expected

    def test_skill_labels_matches_builtin_domain_names(self) -> None:
        assert _BUILTIN_DOMAIN_NAMES.issubset(set(SKILL_LABELS.keys()))

    def test_all_skills_have_non_empty_content(self) -> None:
        for name, skill in LocalSkillStore().list_skills().items():
            assert skill.content.strip(), f"Skill '{name}' has empty content"

    def test_all_skills_are_skill_instances(self) -> None:
        for name, skill in LocalSkillStore().list_skills().items():
            assert isinstance(skill, Skill), f"Skill '{name}' is not a Skill instance"

    def test_competitive_skill_content_is_distinct_from_base(self) -> None:
        skills = LocalSkillStore().list_skills()
        assert (
            skills["data_science::exploratory_analysis"].content
            != skills["competitive_data_science::reconnaissance"].content
        )

    def test_skill_labels_are_non_empty_strings(self) -> None:
        for name, label in SKILL_LABELS.items():
            assert label, f"Label for '{name}' is empty"

    def test_contents_contain_meaningful_content(self) -> None:
        for name, skill in LocalSkillStore().list_skills().items():
            assert len(skill.content) > 200, f"Skill '{name}' content is suspiciously short"

    def test_default_paths_is_builtin(self) -> None:
        loader = LocalSkillStore()
        assert loader._paths == [_BUILTIN_SKILLS_DIRECTORY]

    def test_explicit_none_paths_uses_builtin(self) -> None:
        loader = LocalSkillStore(None)
        assert loader._paths == [_BUILTIN_SKILLS_DIRECTORY]

    def test_default_domain_paths_is_builtin(self) -> None:
        loader = LocalSkillStore()
        assert loader._domain_paths == [_BUILTIN_DOMAINS_DIRECTORY]

    def test_all_builtin_skill_names_match_key(self) -> None:
        for key, skill in LocalSkillStore().list_skills().items():
            assert skill.name == key

    def test_load_resolves_domain_scoped_data_science_skill(self) -> None:
        skill = LocalSkillStore().load("data_science::exploratory_analysis")
        assert skill is not None
        assert skill.name == "data_science::exploratory_analysis"

    def test_load_resolves_domain_scoped_skill(self) -> None:
        skill = LocalSkillStore().load("competitive_data_science::reconnaissance")
        assert skill is not None
        assert skill.name == "competitive_data_science::reconnaissance"

    def test_load_unknown_skill_returns_none(self) -> None:
        assert LocalSkillStore().load("nonexistent") is None


class TestBuiltinDomains:
    def test_domain_registry_has_expected_keys(self) -> None:
        assert set(LocalSkillStore().list_domains().keys()) == _BUILTIN_DOMAIN_NAMES

    def test_domains_are_skill_domain_instances(self) -> None:
        for name, domain in LocalSkillStore().list_domains().items():
            assert isinstance(domain, SkillDomain), f"'{name}' is not a SkillDomain instance"

    def test_domain_names_match_key(self) -> None:
        for key, domain in LocalSkillStore().list_domains().items():
            assert domain.name == key

    def test_domain_has_non_empty_content(self) -> None:
        domain = LocalSkillStore().load_domain("competitive_data_science")
        assert domain is not None
        assert domain.content.strip()

    def test_domain_content_points_to_its_skills(self) -> None:
        domain = LocalSkillStore().load_domain("competitive_data_science")
        assert domain is not None
        assert "skill: competitive_data_science::reconnaissance" in domain.content

    def test_load_domain_unknown_returns_none(self) -> None:
        assert LocalSkillStore().load_domain("nonexistent") is None

    def test_domain_without_manifest_md_is_ignored(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "skill_domains"
        (domains_dir / "empty_domain").mkdir(parents=True)
        result = LocalSkillStore(domain_paths=[domains_dir]).list_domains()
        assert result == {}

    def test_custom_domain_is_loaded(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "skill_domains" / "my_domain"
        domain_dir.mkdir(parents=True)
        (domain_dir / "manifest.md").write_text("# My Domain")
        result = LocalSkillStore(domain_paths=[tmp_path / "skill_domains"]).list_domains()
        assert "my_domain" in result
        assert result["my_domain"].content == "# My Domain"

    def test_later_domain_dir_overrides_earlier(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "m"
        dir_b = tmp_path / "b" / "m"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        (dir_a / "manifest.md").write_text("from_a")
        (dir_b / "manifest.md").write_text("from_b")
        result = LocalSkillStore(domain_paths=[tmp_path / "a", tmp_path / "b"]).list_domains()
        assert result["m"].content == "from_b"


class TestSkillDataclass:
    def test_create_skill(self) -> None:
        skill = Skill(name="my_skill", content="Do the thing.")
        assert skill.name == "my_skill"
        assert skill.content == "Do the thing."

    def test_skill_equality(self) -> None:
        assert Skill(name="s", content="x") == Skill(name="s", content="x")
        assert Skill(name="s", content="x") != Skill(name="s", content="y")
        assert Skill(name="a", content="x") != Skill(name="b", content="x")


class TestSkillDomainDataclass:
    def test_create_skill_domain(self) -> None:
        domain = SkillDomain(name="my_domain", content="Do the thing.")
        assert domain.name == "my_domain"
        assert domain.content == "Do the thing."

    def test_skill_domain_equality(self) -> None:
        assert SkillDomain(name="s", content="x") == SkillDomain(name="s", content="x")
        assert SkillDomain(name="s", content="x") != SkillDomain(name="s", content="y")


class TestDomainScopedSkillsUnderCustomSkillsDir:
    def test_skills_nested_under_domain_folder_are_qualified(self, tmp_path: Path) -> None:
        domain_skills_dir = tmp_path / "skills" / "my_domain"
        domain_skills_dir.mkdir(parents=True)
        (domain_skills_dir / "a_skill.md").write_text("content a")
        result = LocalSkillStore([tmp_path / "skills"]).list_skills()
        assert "my_domain::a_skill" in result
        assert result["my_domain::a_skill"].content == "content a"
        assert result["my_domain::a_skill"].name == "my_domain::a_skill"


class TestLocalSkillStoreUserDefined:
    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert result == {}

    def test_nonexistent_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        result = LocalSkillStore([tmp_path / "nonexistent"]).load_user_defined()
        assert result == {}

    def test_md_file_loaded_by_stem(self, tmp_path: Path) -> None:
        (tmp_path / "my_skill.md").write_text("# My Skill\nDo the thing.")
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert "my_skill" in result
        assert result["my_skill"].content == "# My Skill\nDo the thing."

    def test_skip_files_with_unsupported_extension(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("whatever")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "skill.json").write_text('{"name": "x", "prompt": "y"}')
        (tmp_path / "skill.yaml").write_text("name: x\nprompt: y\n")
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert result == {}

    def test_multiple_skills_loaded(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"skill{i}.md").write_text(f"prompt for skill {i}")
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert len(result) == 3

    def test_user_defined_overrides_builtin(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "data_science"
        domain_dir.mkdir()
        (domain_dir / "exploratory_analysis.md").write_text("custom data science prompt")
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, tmp_path]).list_skills()
        assert skills["data_science::exploratory_analysis"].content == "custom data science prompt"

    def test_builtin_dir_excluded_from_load_user_defined(self, tmp_path: Path) -> None:
        (tmp_path / "my_skill.md").write_text("user prompt")
        loader = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, tmp_path])
        user = loader.load_user_defined()
        assert "my_skill" in user
        assert "data_science::exploratory_analysis" not in user


class TestLocalSkillStoreMultiDir:
    def test_multiple_dirs_merged(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "skill_a.md").write_text("content a")
        (dir_b / "skill_b.md").write_text("content b")
        result = LocalSkillStore([dir_a, dir_b]).list_skills()
        assert "skill_a" in result
        assert "skill_b" in result

    def test_later_dir_overrides_earlier(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "skill.md").write_text("from_a")
        (dir_b / "skill.md").write_text("from_b")
        result = LocalSkillStore([dir_a, dir_b]).list_skills()
        assert result["skill"].content == "from_b"

    def test_empty_list_returns_empty_dict(self) -> None:
        result = LocalSkillStore([]).list_skills()
        assert result == {}
