# Development and release procedure

This document is the reference for everyone working on `snt_utils`, developers and AI coding agents alike.
`snt_lib` is installed by other projects (the SNT pipelines), so a change that lands on a release branch
reaches those projects. Follow these steps exactly.

## Branches

| Branch | Role | Version in `pyproject.toml` | Who writes to it |
|--------|------|------------------------------|------------------|
| `main` | Development. All new work and all fixes land here first, through pull requests. | Frozen (`0.60.0`), never bumped | PRs only |
| `release-v2` | Current major version. Every version bump here publishes a `v2.x.y` release. | `2.x.y` | Merges from `main` + version bump |
| `release-v1` | Maintenance of v1, for projects not yet on v2. Receives backported fixes only. | `1.x.y` | Cherry-picked fixes + version bump |

Changes flow **out of `main`** only: into `release-v2` by merge, into `release-v1` by cherry-pick.
Nothing is ever merged from a release branch back into `main`.

## Versions and tags

Versions follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

| Change | Bump | Example |
|--------|------|---------|
| A public function is removed or renamed, a parameter is removed, renamed or made required, or a return type changes | MAJOR (needs a new `release-vN` branch, see below) | `2.3.1` → `3.0.0` |
| A new public function, or a new optional parameter with a default that keeps the old behaviour | MINOR | `2.3.1` → `2.4.0` |
| A bug fix that changes no signature | PATCH | `2.3.1` → `2.3.2` |

Tags are created by CI only, never by hand:

- The [release workflow](../.github/workflows/release.yml) runs on every push to a release branch. If the version in
  `pyproject.toml` has no `v<version>` tag yet, it creates the tag and a GitHub release. Otherwise it does nothing.
- `v*` tags are protected: they cannot be moved, deleted or force-pushed. A published version is permanent, so a
  mistake is fixed by releasing the next version, not by retagging.
- Tags belong to the repository, not to a branch: `v1.x.y` and `v2.x.y` tags sit side by side and never collide.

## Everyday development (fixes and features)

1. Create a branch from an up-to-date `main`:
   ```bash
   git switch main && git pull
   git switch -c <short-description>
   ```
2. Make the change. Keep `pyproject.toml`'s version untouched.
3. Before pushing, run the same checks as CI:
   ```bash
   ruff check snt_lib/ tests/
   ruff format --check snt_lib/ tests/
   pytest
   ```
4. Update [README.md](../README.md) when a public function is added, removed or changes signature.
5. Open a pull request into `main`. Commit messages and PR titles follow
   [Conventional Commits](https://www.conventionalcommits.org) (`type(scope): description`); mark breaking changes
   with `!` (`refactor!: ...`).

## Releasing a new v2 version

1. Merge `main` into `release-v2`:
   ```bash
   git switch release-v2 && git pull
   git merge origin/main
   ```
2. Bump `version` in `pyproject.toml` according to the table above, then commit it as `release: v2.x.y`.
3. Push. The release workflow creates the `v2.x.y` tag and the GitHub release.
4. Check that the release appears under [Releases](https://github.com/BLSQ/snt_utils/releases).

## Backporting a fix to v1

Only fixes go to v1, never new features or breaking changes.

1. Land the fix on `main` first (everyday development above), so v2 gets it too.
2. Cherry-pick it onto a branch from `release-v1`:
   ```bash
   git switch release-v1 && git pull
   git switch -c backport-<short-description>
   git cherry-pick <commit-sha-on-main>
   ```
   Cherry-pick only the fix commits: never a merge commit or a version bump. If the code differs too much,
   rewrite the fix for v1 on this branch instead.
3. Bump the PATCH version (`1.x.y` → `1.x.(y+1)`) in `pyproject.toml` and open a pull request into `release-v1`.
4. Merging it publishes `v1.x.y`. The v1 release workflow passes `--latest=false`, so the GitHub "Latest" badge stays
   on v2.

A fix needed only by v1 (the code no longer exists in v2) skips step 1.

## Starting a new major version (v3)

Only when breaking changes have landed on `main` and are ready to ship.

1. Create `release-v3` from an up-to-date `main`.
2. On `main` (through a PR), point both workflows at the new branch: `branches: [release-v3]` in
   `.github/workflows/release.yml`, and add `release-v3` to the push branches in `.github/workflows/ci.yaml`.
   Merge `main` into `release-v3` so it has them.
3. On `release-v3`, set `version = "3.0.0"` and push: this publishes `v3.0.0`.
4. On `release-v2`, add `--latest=false` to the `gh release create` line in `release.yml`, so v2 maintenance releases
   do not take the "Latest" badge. `release-v2` then follows the v1 backport procedure, and README.md is updated to
   point to `release-v3`.

## How consuming projects install the library

```bash
pip install git+https://github.com/BLSQ/snt_utils.git@v2.0.0       # exact version (reproducible, recommended)
pip install git+https://github.com/BLSQ/snt_utils.git@release-v2   # latest v2 release
pip install git+https://github.com/BLSQ/snt_utils.git@release-v1   # latest v1 release
```

A project changes major version only by editing its own install reference. List breaking changes and what replaces
them in the release notes of every MAJOR release.
