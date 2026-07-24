# github.com — Issues & PRs via `gh`

- When unsure of the current flags, `gh issue --help` / `gh pr --help` (and `--help` on the specific subcommand) is the authoritative reference — flags for search scope, state filters, and output fields shift between `gh` versions, so confirm rather than assume
- `gh issue list` with a search term checks whether a bug is already known before assuming it's novel — searching across all states (not just open) surfaces issues that were closed with a workaround or a fix in a later release, so don't default to open-only
- Reading an issue's full comment thread (not just its body) matters — the actual fix or workaround is frequently in a maintainer's reply partway down the thread
- `gh pr list` with a search term finds the PR that fixed a given bug; viewing that PR's diff shows exactly what changed, which is more reliable than inferring a fix from a changelog line
- A PR's CI status (surfaced via `gh pr checks` or equivalent) matters when deciding whether an unmerged fix is safe to apply manually before it lands in a release
- Structured output flags (JSON output plus a query/filter flag — check `--help` for the current names) turn free-text results into something parseable; prefer that over scraping human-formatted table output when only a specific field is needed

## Metadata

- parent domain: github.com
