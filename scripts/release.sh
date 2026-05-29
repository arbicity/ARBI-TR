#!/usr/bin/env bash
#
# Cut a release: bump the version from conventional commits, update the changelog,
# regenerate the client, tag, push, and create a GitHub release.
#
# Usage:
#   ./scripts/release.sh                 # auto-detect bump (patch/minor/major) from commits
#   ./scripts/release.sh patch|minor|major
#   ./scripts/release.sh --dry-run       # preview, make no changes
#   ./scripts/release.sh --publish-client # ALSO build+publish the client to PyPI (see note)
#
# NOTE: publishing the generated client to PyPI is the DEFERRED "end step". It is
# OFF by default; pass --publish-client only once the package + PYPI token are ready.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# commitizen + the client generator live in the backend venv.
CZ="uv run --project backend cz"

DRY_RUN=""
INCREMENT=""
PUBLISH_CLIENT=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="1" ;;
    --publish-client) PUBLISH_CLIENT="1" ;;
    patch|minor|major) INCREMENT="--increment $arg" ;;
  esac
done

echo "=== ARBI-TR Release ==="

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ] && [ -z "$DRY_RUN" ]; then
  echo "WARNING: not on main (on $BRANCH)."
fi
if [ -n "$(git status --porcelain)" ] && [ -z "$DRY_RUN" ]; then
  echo "ERROR: working tree not clean — commit or stash first."
  exit 1
fi
CURRENT_VERSION=$($CZ version --project 2>/dev/null || echo unknown)
echo "Current version: $CURRENT_VERSION"

# ── 2. Bump + changelog ──────────────────────────────────────────────────────
if [ -n "$DRY_RUN" ]; then
  $CZ bump --dry-run --yes $INCREMENT </dev/null
  echo "DRY RUN — no changes made."
  exit 0
fi

# cz bump: updates version_files (pyproject.toml x3 + CLIENT_VERSION), writes the
# CHANGELOG, and creates an annotated git commit + tag (vX.Y.Z).
$CZ bump --changelog --yes $INCREMENT </dev/null
NEW_VERSION=$($CZ version --project)
echo "Bumped $CURRENT_VERSION -> $NEW_VERSION"

# ── 3. Regenerate the client at the new version ──────────────────────────────
# Version is now owned by commitizen for the release, so skip the diff-bump.
SKIP_VERSION_BUMP=1 SKIP_INSTALL=1 bash scripts/generate-client.sh >/dev/null
echo "Client regenerated."

# ── 4. (DEFERRED) Publish client to PyPI ─────────────────────────────────────
if [ -n "$PUBLISH_CLIENT" ]; then
  echo "Building + publishing client to PyPI..."
  [ -f CHANGELOG.md ] && cp CHANGELOG.md client/CHANGELOG.md
  ( cd backend && uv build --directory ../client )
  ( cd backend && uv publish --directory ../client )   # requires UV_PUBLISH_TOKEN
  rm -f client/CHANGELOG.md
  echo "Published arbi-tr-client==$NEW_VERSION"
else
  echo "Skipping PyPI publish (deferred end step — pass --publish-client to enable)."
fi

# ── 5. Push + GitHub release ─────────────────────────────────────────────────
git push && git push --tags
if command -v gh >/dev/null 2>&1; then
  NOTES=$(sed -n "/^## v\?${NEW_VERSION}/,/^## /{/^## /d; p;}" CHANGELOG.md)
  [ -z "$NOTES" ] && NOTES="Release v${NEW_VERSION}"
  gh release create "v${NEW_VERSION}" --title "v${NEW_VERSION}" --notes "$NOTES"
fi

echo "=== Released v${NEW_VERSION} ==="
