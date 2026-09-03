# github.com Skill Domain

Curated usage patterns for interacting with GitHub through the `gh` CLI rather than a browser — the agent has no browser, so every GitHub interaction goes through `gh` (or `gh api`) via the CLI-execution tool. Load a skill here before reaching for ad-hoc `curl`/scraping of GitHub, since `gh` almost always has a purpose-built subcommand. These skills teach which subcommand group to reach for and why, not exact flags — `gh`'s flags and JSON field names change across versions, so `gh <command> --help` is always the authority on current syntax.

## Repository Reconnaissance

- skill: github.com::repository_reconnaissance

How to pull a repo's vitals, read a single file without cloning, and check a specific release's changelog, all via `gh repo view`/`gh api`/`gh release`; load when checking out or evaluating a specific repository.

## Issues & PRs

- skill: github.com::issues_and_prs

How to search a repo's issues and PRs for a known bug or its fix, read full comment threads, and check PR CI status via `gh issue`/`gh pr`; load when debugging a library's behavior or verifying whether an unmerged fix is safe to apply.

## Code & Repo Search

- skill: github.com::code_and_repo_search

How to search real-world code usage, find reference implementations, and check whether a symptom is a widespread issue across GitHub via `gh search`; load when looking for usage examples or corroborating a suspected library-wide bug.
