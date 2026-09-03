# github.com — Repository Reconnaissance via `gh`

- `gh` flags and JSON field names drift between versions — run `gh repo view --help` (or `gh <subcommand> --help` generally) to confirm current syntax before relying on a remembered invocation; the subcommand groups below (`repo`, `release`, `api`) are stable even when their flags aren't
- `gh repo view` returns a repo's vitals as structured data — prefer it over opening the page in a browser, since the agent has no browser and the human-facing page carries no information the API doesn't; `--help` shows which fields (stars, language, topics, archived status, default branch) are currently exposable via `--json`
- Checking whether a repo is archived or a fork of the real project (surfaced via `gh repo view`'s JSON fields) before trusting its code avoids building on dead weight
- `gh api <path>` reaches GitHub's REST/GraphQL surface directly and is the right tool for reading a single file (e.g. a README or config) without a full clone — `gh api --help` and the API's own reference (accessible by asking `gh` or checking its docs) describe the current endpoint shapes, which are versioned independently of the `gh` CLI itself
- `gh release list` / `gh release view` surface a specific version's changelog directly — faster and more precise than cloning and grepping `CHANGELOG.md`, and the right first check before upgrading a pinned dependency
- `gh repo clone` is worth the cost only once more than a couple of files are needed (browsing multiple modules, running the repo's own tests); for anything narrower, `gh api`'s single-file read avoids the clone entirely

## Metadata

- parent domain: github.com
