#!/usr/bin/env bash
#
# Auto-generate the ARBI-TR Python client from the OpenAPI spec.
#
# Usage:
#   ./scripts/generate-client.sh                 # generate spec from source code (default)
#   ./scripts/generate-client.sh --server        # fetch spec from http://localhost:8000
#   ./scripts/generate-client.sh --server http://host:8000
#
# Environment variables:
#   SKIP_VERSION_BUMP=1   Don't bump CLIENT_VERSION (use on PR branches — version bumps on main only)
#   SKIP_INSTALL=1        Don't `uv pip install -e client/` afterwards (CI publish-only / spec-only runs)
#   DIFF_BASE=<ref>       Git ref to diff the spec against (default: HEAD)
#
# Layout:
#   openapi.json            tracked spec — the diff baseline, reviewable in PRs
#   client/                 generated package (gitignored)
#   client.pyproject.toml   curated packaging template -> copied to client/pyproject.toml
#   client.README.md        curated readme template    -> copied to client/README.md
#   CLIENT_VERSION          semver of the generated client, bumped from the API diff
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BACKEND="backend"
CLIENT_DIR="client"
SPEC="openapi.json"

echo "========================================="
echo "ARBI-TR API Client Generator"
echo "========================================="

USE_SERVER=false
SERVER_URL="http://localhost:8000"
for arg in "$@"; do
    case "$arg" in
        --server) USE_SERVER=true ;;
        http*) SERVER_URL="$arg"; USE_SERVER=true ;;
    esac
done

# ── 1. Obtain the OpenAPI spec ────────────────────────────────────────────────
if [ "$USE_SERVER" = true ]; then
    echo "Mode: fetch from running server ($SERVER_URL)"
    curl -sf "$SERVER_URL/openapi.json" -o "$SPEC"
else
    echo "Mode: generate from source code"
    # main.py lives in backend/ and imports lazily (no GPU/model load at import time),
    # so app.openapi() runs fine without any runtime env.
    ( cd "$BACKEND" && uv run python -c "import json; from main import app; print(json.dumps(app.openapi(), indent=2))" ) > "$SPEC"
fi

# Normalize binary upload markers: FastAPI (OpenAPI 3.1) emits file fields as
# {"type":"string","contentMediaType":"application/octet-stream"}, which
# openapi-python-client does not recognize as a file (it would emit `str` and a
# broken text-part upload). Rewrite them to the 3.0-style {"format":"binary"} so
# the generated client uploads real bytes via File.to_tuple().
python3 - "$SPEC" <<'PYEOF'
import json, sys
def fix(o):
    if isinstance(o, dict):
        if o.get("type") == "string" and o.get("contentMediaType") == "application/octet-stream":
            o.pop("contentMediaType"); o["format"] = "binary"
        for v in o.values(): fix(v)
    elif isinstance(o, list):
        for v in o: fix(v)
spec = json.load(open(sys.argv[1])); fix(spec)
json.dump(spec, open(sys.argv[1], "w"), indent=2)
PYEOF
echo "OpenAPI spec -> $SPEC ($(wc -c < "$SPEC") bytes)"

# ── 2. Snapshot the previous spec for diffing ────────────────────────────────
HAS_OLD=false
if git rev-parse --verify HEAD >/dev/null 2>&1 && git ls-files --error-unmatch "$SPEC" >/dev/null 2>&1; then
    if git show "${DIFF_BASE:-HEAD}:$SPEC" > "$SPEC.old" 2>/dev/null; then
        HAS_OLD=true
    fi
fi

OLD_VERSION="0.1.0"
[ -f CLIENT_VERSION ] && OLD_VERSION="$(tr -d '[:space:]' < CLIENT_VERSION)"
echo "Current client version: $OLD_VERSION"

# ── 3. Generate the client ───────────────────────────────────────────────────
echo "Generating Python client..."
rm -rf "$CLIENT_DIR"
( cd "$BACKEND" && uv run openapi-python-client generate \
    --path "../$SPEC" --output-path "../$CLIENT_DIR" --overwrite --meta uv )

# openapi-python-client nests the package under client/<pkg>/ (derived from the API title)
CLIENT_PKG=$(find "$CLIENT_DIR" -maxdepth 1 -mindepth 1 -type d \
    ! -name '.*' ! -name '__pycache__' | head -1)
if [ -z "$CLIENT_PKG" ] || [ ! -f "$CLIENT_PKG/client.py" ] || [ ! -d "$CLIENT_PKG/models" ] || [ ! -d "$CLIENT_PKG/api" ]; then
    echo "❌ Client generation failed: missing client.py / models/ / api/ under $CLIENT_DIR"
    exit 1
fi
MODEL_COUNT=$(find "$CLIENT_PKG/models" -name '*.py' ! -name '__init__.py' | wc -l | tr -d ' ')
API_COUNT=$(find "$CLIENT_PKG/api" -name '*.py' ! -name '__init__.py' | wc -l | tr -d ' ')
echo "✅ Generated: $MODEL_COUNT models, $API_COUNT API modules (package: ${CLIENT_PKG##*/})"

# ── 4. Diff the spec and decide the semver bump ──────────────────────────────
SEMVER_BUMP="NONE"
mkdir -p "$CLIENT_DIR/diffs"
if [ "$HAS_OLD" = true ]; then
    DIFF_FILE="$CLIENT_DIR/diffs/schema-diff-$(git rev-parse --short HEAD)-$$.log"
    SEMVER_BUMP=$(python3 - "$SPEC.old" "$SPEC" "$DIFF_FILE" <<'PYEOF'
import json, sys
old = json.load(open(sys.argv[1])); new = json.load(open(sys.argv[2]))
out = open(sys.argv[3], "w")
def log(*a): print(*a, file=out)

def ops(spec):
    r = {}
    for path, methods in spec.get("paths", {}).items():
        for m, d in methods.items():
            if m in ("get", "post", "put", "delete", "patch"):
                r[d.get("operationId", f"{m}_{path}")] = (m.upper(), path, d)
    return r

oo, no = ops(old), ops(new)
added_ep = sorted(set(no) - set(oo))
removed_ep = sorted(set(oo) - set(no))
modified_ep = sorted(k for k in set(oo) & set(no)
                     if json.dumps(oo[k][2], sort_keys=True) != json.dumps(no[k][2], sort_keys=True))

os_, ns = old.get("components", {}).get("schemas", {}), new.get("components", {}).get("schemas", {})
added_sc = sorted(set(ns) - set(os_))
removed_sc = sorted(set(os_) - set(ns))
modified_sc = sorted(k for k in set(os_) & set(ns)
                     if json.dumps(os_[k], sort_keys=True) != json.dumps(ns[k], sort_keys=True))

breaking = bool(removed_ep or removed_sc)
# a modified endpoint that drops/renames params or changes the request body is breaking
for k in modified_ep:
    o, n = oo[k][2], no[k][2]
    o_p = {p.get("name") for p in o.get("parameters", [])}
    n_p = {p.get("name") for p in n.get("parameters", [])}
    if (o_p - n_p) or json.dumps(o.get("requestBody"), sort_keys=True) != json.dumps(n.get("requestBody"), sort_keys=True):
        breaking = True

log("ARBI-TR API client diff")
for label, items in (("ADDED endpoints", added_ep), ("REMOVED endpoints", removed_ep),
                     ("MODIFIED endpoints", modified_ep), ("ADDED schemas", added_sc),
                     ("REMOVED schemas", removed_sc), ("MODIFIED schemas", modified_sc)):
    if items:
        log(f"\n{label} ({len(items)}):")
        for it in items:
            log(f"  - {it}")

changed = bool(added_ep or removed_ep or modified_ep or added_sc or removed_sc or modified_sc)
bump = "MINOR" if breaking else ("PATCH" if changed else "NONE")
log(f"\nSEMVER BUMP: {bump}  ({'breaking' if breaking else 'compatible' if changed else 'no changes'})")
out.close()
print(bump)
PYEOF
)
    echo "API diff -> $DIFF_FILE (bump: $SEMVER_BUMP)"
    cat "$DIFF_FILE"
else
    echo "No previous spec in git — first generation, no diff."
fi
rm -f "$SPEC.old"

# ── 5. Compute the new client version ────────────────────────────────────────
NEW_VERSION="$OLD_VERSION"
if [ "${SKIP_VERSION_BUMP:-}" = "1" ]; then
    echo "Version: $OLD_VERSION (bump skipped — SKIP_VERSION_BUMP=1)"
elif [ "$SEMVER_BUMP" != "NONE" ]; then
    IFS='.' read -r MA MI PA <<< "$OLD_VERSION"
    if [ "$SEMVER_BUMP" = "MINOR" ]; then MI=$((MI + 1)); PA=0; else PA=$((PA + 1)); fi
    NEW_VERSION="${MA}.${MI}.${PA}"
    echo "Version bumped: $OLD_VERSION -> $NEW_VERSION ($SEMVER_BUMP)"
    echo "$NEW_VERSION" > CLIENT_VERSION
else
    echo "Version unchanged: $OLD_VERSION (no API changes)"
fi

# ── 6. Apply curated packaging templates (overwrite generator output) ────────
sed "s/__VERSION__/${NEW_VERSION}/g" client.pyproject.toml > "$CLIENT_DIR/pyproject.toml"
cp client.README.md "$CLIENT_DIR/README.md"

# ── 7. Install locally so tests/consumers can import it ──────────────────────
if [ "${SKIP_INSTALL:-}" != "1" ]; then
    echo "Installing client into the backend venv..."
    ( cd "$BACKEND" && uv pip install -e "../$CLIENT_DIR" --quiet )
    ( cd "$BACKEND" && uv run python -c "import ${CLIENT_PKG##*/}; print('✅ import ${CLIENT_PKG##*/} OK')" )
else
    echo "Skipping local install (SKIP_INSTALL=1)"
fi

echo ""
echo "========================================="
echo "Done. client=$CLIENT_DIR  version=$NEW_VERSION  models=$MODEL_COUNT  api=$API_COUNT"
echo "========================================="
