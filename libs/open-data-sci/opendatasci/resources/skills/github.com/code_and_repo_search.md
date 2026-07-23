# github.com — Code & Repo Search via `gh`

- `gh search --help` (and `--help` on `code`/`repos`/`issues` beneath it) is the source of truth for available filters — treat any specific flag as a starting guess to verify, not a fact to memorize, since search filters are among the parts of `gh` most likely to gain or rename options over time
- `gh search code` finds real-world usage examples across all public repos — the fastest way to see how a library or API is actually called in practice, rather than inferring usage from docs alone
- `gh search repos` finds reference implementations for a paper, technique, or SOTA result — pair with the `arxiv.org`/`paperswithcode.com` skills when the paper itself links to no code, or the linked repo has gone stale
- `gh search issues` without scoping to one repo searches across all of GitHub — useful for confirming whether a symptom is a widespread dependency-compatibility issue rather than specific to the one project being debugged
- Results default to a relevance/popularity ranking; narrowing by language, recency, or star count (exact flag names via `--help`) before reading results avoids wading through abandoned or toy repos that happen to match the query text

## Metadata

- parent domain: github.com
