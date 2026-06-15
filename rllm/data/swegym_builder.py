"""Builder for the SWE-Gym sandbox benchmark.

SWE-Gym (``SWE-Gym/SWE-Gym``, plus ``SWE-Gym-Lite`` for held-out eval)
ships as a flat HuggingFace row dataset with the same schema as the
original SWE-bench (instance_id, repo, base_commit, patch, test_patch,
problem_statement, FAIL_TO_PASS, PASS_TO_PASS, version) but covers an
expanded set of repos — getmoto/moto, python/mypy, iterative/dvc,
Project-MONAI/MONAI, pydantic/pydantic, dask/dask, conan-io/conan,
facebookresearch/hydra, pandas-dev/pandas, modin-project/modin, and
more on top of the original SWE-bench Python set.

Each instance has a pre-built OpenHands-flavored Docker image at
``xingyaoww/sweb.eval.x86_64.<instance_id with __ → _s_>:latest`` —
the testbed lives at ``/testbed`` already checked out to base_commit
and a conda env ``testbed`` is preconfigured with the deps.

The official eval flow lives in the SWE-Gym fork of the swebench package
(``SWE-Gym/SWE-Bench-Package``, importable as ``swegym``). At dataset-build
time we ask that package to generate the per-instance ``eval_script`` —
the bash that activates the conda env, applies the test_patch, and runs
the repo-appropriate ``test_cmd`` — and bake it into the task tree. The
verifier ``tests/test.sh`` then layers on the agent's patch and parses
test outcomes against ``FAIL_TO_PASS ∪ PASS_TO_PASS`` to assign reward.

On-disk output (``<out_dir>/``)::

    swegym/
    ├── dataset.toml                       # type="sandbox"
    ├── log_parsers.py                     # copied from swegym at build time
    ├── <instance_id>/
    │   ├── task.toml                      # docker_image=xingyaoww/..., workdir=/testbed
    │   ├── instruction.md                 # problem_statement
    │   ├── environment/Dockerfile         # FROM xingyaoww/sweb.eval.x86_64.<...>
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier wrapper
    │   │   ├── eval.sh                    # swegym-generated per-instance eval bash
    │   │   └── instance.json              # repo, base_commit, F2P, P2P
    │   └── solution/solve.sh              # apply the gold patch from `patch`
    └── ...

Invoked from ``rllm dataset pull swegym`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "SWE-Gym/SWE-Gym"
DOCKERHUB_NAMESPACE = "xingyaoww/sweb.eval.x86_64"

# Per-instance resource defaults. SWE-Gym testbeds are full Python repos
# (pandas, dask, sympy, ...) — the test runs can OOM with the 1 GiB default
# on remote backends. Docker ignores these; Modal/Daytona honor them.
_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 16384,
    "storage_mb": 30720,
    "build_timeout_sec": 1800.0,
}

_DEFAULT_TIMEOUTS = {
    "agent_timeout_sec": 1800.0,
    "verifier_timeout_sec": 1800.0,
}


def _instance_image_tag(instance_id: str) -> str:
    """Docker Hub tag for an instance, mirroring OpenHands' ``__`` → ``_s_`` rule.

    Docker Hub repository names cannot contain ``__``. The SWE-Gym mirror
    at ``xingyaoww/sweb.eval.x86_64.*`` substitutes the double underscore
    (which separates org from repo in SWE-bench's ``instance_id``s) with
    ``_s_`` (``s`` for slash). The original-case suffix is preserved.
    """
    return instance_id.lower().replace("__", "_s_")


def _decode_json_list(value: Any) -> list[str]:
    """Parse a field that's either a JSON-encoded list or already a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            loaded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            logger.warning("[swegym] could not parse list-valued field: %r", text[:120])
            return []
    return [str(v) for v in (loaded or [])]


def _build_dockerfile(instance_id: str) -> str:
    """Dockerfile that pulls the pre-built OpenHands testbed and clears ENTRYPOINT.

    Same hazard as SWE-bench Pro: the base images may inherit an ENTRYPOINT
    from the parent (e.g. ``["/bin/bash"]``) that interferes with rLLM's
    ``sleep infinity`` keep-alive command. Reset to ``[]`` so docker runs
    ``sleep infinity`` directly.
    """
    tag = _instance_image_tag(instance_id)
    return f"FROM {DOCKERHUB_NAMESPACE}.{tag}:latest\nENTRYPOINT []\nWORKDIR /testbed\n"


def _build_task_toml(
    *,
    instance_id: str,
    repo: str,
    base_commit: str,
    version: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``."""
    tag = _instance_image_tag(instance_id)
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "swegym/{instance_id}"',
        f'description = "SWE-Gym: {repo} v{version}"',
        f'keywords = ["swe-gym", "{repo.split("/")[0]}"]',
        "",
        "[metadata]",
        f'instance_id = "{instance_id}"',
        f'repo = "{repo}"',
        f'version = "{version}"',
        f'base_commit = "{base_commit}"',
        "",
        "[environment]",
        f'docker_image = "{DOCKERHUB_NAMESPACE}.{tag}:latest"',
        'workdir = "/testbed"',
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[agent]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['agent_timeout_sec']}",
        "",
        "[verifier]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
        "",
    ]
    return "\n".join(lines)


# The verifier wraps the swegym-generated per-instance ``eval.sh``. The
# flow mirrors ``swegym.harness.run_evaluation.run_instance``:
#
#   1. Capture the agent's diff at /testbed as the model_patch.
#   2. Hard-reset /testbed to base_commit so the patch applies on a clean tree.
#   3. ``git apply --allow-empty`` the model_patch (with a ``patch -p1`` fallback,
#      matching the upstream eval).
#   4. Run the per-instance ``eval.sh`` (activates conda, applies test_patch,
#      runs the repo's ``test_cmd``) capturing its output to ``test_output.txt``.
#   5. Parse the output using the repo-appropriate parser from the bundled
#      ``log_parsers.py`` (copied from swegym at dataset-build time).
#   6. Compute reward = 1.0 iff every test in ``FAIL_TO_PASS ∪ PASS_TO_PASS``
#      is in the parser's PASSED set.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[verifier] $*"; }

write_failure() {
    python3 - "$1" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' > "$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /testbed 2>/dev/null || { write_failure "/testbed missing"; exit 0; }

git config --global --add safe.directory /testbed 2>/dev/null || true

INSTANCE_JSON=/tests/instance.json
EVAL_SCRIPT=/tests/eval.sh
LOG_PARSERS=/tests/log_parsers.py

for f in "$INSTANCE_JSON" "$EVAL_SCRIPT" "$LOG_PARSERS"; do
    if [ ! -f "$f" ]; then
        write_failure "missing $f"
        exit 0
    fi
done

BASE_COMMIT="$(python3 -c "import json; print(json.load(open('$INSTANCE_JSON'))['base_commit'])")"
if [ -z "$BASE_COMMIT" ]; then
    write_failure "base_commit empty"
    exit 0
fi

# Step 1: capture the agent's diff vs base_commit. Mirror swebench_pro:
# diff --cached vs base_commit so HEAD is irrelevant — these testbed
# images are at base_commit by construction, but the agent may have
# committed mid-run, and we want the cumulative edits anyway.
log "Capturing agent diff vs base_commit ($BASE_COMMIT)"
MODEL_PATCH=/tmp/model_patch.diff
git add -A . >/dev/null 2>&1 || true
git diff --cached --binary "$BASE_COMMIT" > "$MODEL_PATCH" 2>/dev/null || true
git reset >/dev/null 2>&1 || true

PATCH_BYTES=$(wc -c < "$MODEL_PATCH" 2>/dev/null || echo 0)
log "Captured patch: $PATCH_BYTES bytes"

# Step 2: reset to base_commit so the patch applies on a clean tree.
log "Resetting to base_commit"
git reset --hard "$BASE_COMMIT" >/dev/null 2>&1 || log "git reset --hard failed (continuing)"
git checkout "$BASE_COMMIT" >/dev/null 2>&1 || log "git checkout failed (continuing)"
git clean -fd >/dev/null 2>&1 || log "git clean -fd failed (continuing)"

# Step 3: apply the agent's patch with the same two-step fallback the
# upstream evaluator uses (git apply, then patch -p1 --fuzz=5).
if [ -s "$MODEL_PATCH" ]; then
    if ! git apply --allow-empty -v "$MODEL_PATCH" 2>&1 | tail -20; then
        log "git apply failed, trying patch --batch --fuzz=5 -p1"
        if ! patch --batch --fuzz=5 -p1 -i "$MODEL_PATCH" 2>&1 | tail -20; then
            log "patch fallback also failed — tests will run against base_commit"
        fi
    fi
else
    log "No agent changes detected"
fi

# Step 4: run swegym's per-instance eval.sh. Captures stdout+stderr to
# /tmp/test_output.txt; the script intentionally doesn't set -e, so a
# failing test command (the common case for a wrong-patch) still emits
# the per-test PASSED/FAILED markers the parser needs.
log "Running eval.sh"
chmod +x "$EVAL_SCRIPT"
bash "$EVAL_SCRIPT" > /tmp/test_output.txt 2>&1 || log "eval.sh exited non-zero (parser inspects log)"

# Step 5+6: parse the log and score against F2P + P2P.
python3 <<'PY'
import importlib.util, json, sys
REWARD = "/tmp/rllm/reward.json"

def _tail(path, n=2000):
    try:
        return open(path).read()[-n:]
    except Exception:
        return ""

try:
    inst = json.load(open("/tests/instance.json"))
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"instance.json missing: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

spec = importlib.util.spec_from_file_location("log_parsers", "/tests/log_parsers.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    parsers = getattr(mod, "MAP_REPO_TO_PARSER", {})
    repo = inst.get("repo", "").lower()
    parser = parsers.get(repo)
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"log_parsers import: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

if parser is None:
    json.dump({
        "reward": 0.0, "is_correct": False,
        "metadata": {"error": f"no parser registered for repo {repo!r}",
                     "log_tail": _tail("/tmp/test_output.txt")},
    }, open(REWARD, "w"))
    raise SystemExit(0)

try:
    log = open("/tmp/test_output.txt").read()
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"reading test_output.txt: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

status_map = parser(log)
passed = {name for name, status in status_map.items() if status == "PASSED"}

f2p = set(inst.get("FAIL_TO_PASS") or [])
p2p = set(inst.get("PASS_TO_PASS") or [])
required = f2p | p2p
missing = sorted(required - passed)
matched = sorted(required & passed)

reward = 1.0 if required and not missing else 0.0
json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "f2p_required": len(f2p),
        "p2p_required": len(p2p),
        "passed_required": len(matched),
        "missing_required": len(missing),
    },
    "metadata": {
        "missing": missing[:50],
        "passed_required_sample": matched[:10],
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    return _VERIFIER_TEMPLATE


def _build_solution_script(base_commit: str) -> str:
    """Oracle harness: reset to base_commit, then apply the gold patch."""
    return (
        "#!/bin/bash\n"
        "set -e\n"
        "cd /testbed\n"
        "git config --global --add safe.directory /testbed 2>/dev/null || true\n"
        f'git reset --hard "{base_commit}"\n'
        f'git checkout "{base_commit}"\n'
        "git apply -v /solution/gold.patch\n"
    )


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "{description}"',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
            "[verifier]",
            'script = "tests/test.sh"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def _import_swegym():
    """Import the swegym package or raise an instructive error.

    swegym is a SWE-Gym fork of the swebench harness that adds SPECS for
    repos like getmoto, pandas-dev, mypy, etc. that aren't in upstream
    ``swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS``. The fork is
    not on PyPI — install it from git.
    """
    try:
        import swegym.harness.log_parsers as log_parsers
        import swegym.harness.test_spec as test_spec

        return test_spec, log_parsers
    except ImportError as e:
        raise RuntimeError(
            f"The 'swegym' package is required to build the SWE-Gym dataset. Install it with:\n\n  pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git\n\n(import error: {e})"
        ) from e


def _materialize_task(
    task_dir: Path,
    row: dict,
    log_parsers_src_path: Path,
    make_test_spec,
) -> None:
    """Expand a single HF row into a Harbor-format task tree."""
    task_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    repo = row.get("repo", "")
    base_commit = row.get("base_commit", "")
    version = str(row.get("version", ""))

    # task.toml
    (task_dir / "task.toml").write_text(
        _build_task_toml(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            version=version,
        ),
        encoding="utf-8",
    )

    # instruction.md
    instruction = (row.get("problem_statement") or "").strip() + "\n"
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(instance_id), encoding="utf-8")

    # tests/
    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)

    # Generate the per-instance eval bash via swegym. The TestSpec returned
    # by ``make_test_spec`` exposes ``eval_script`` — the same script
    # ``swegym.harness.run_evaluation`` copies into the container as
    # ``/eval.sh`` before running the model_patch.
    spec = make_test_spec(row)
    (tests_dst / "eval.sh").write_text(spec.eval_script, encoding="utf-8")
    (tests_dst / "eval.sh").chmod(0o755)

    # Copy the bundled log_parsers.py once per task (small, ~600 lines).
    # Per-task rather than dataset-root so each task dir is self-contained
    # and the loader's task_dir-rooted mount works as-is.
    shutil.copy2(log_parsers_src_path, tests_dst / "log_parsers.py")

    fail_to_pass = _decode_json_list(row.get("FAIL_TO_PASS"))
    pass_to_pass = _decode_json_list(row.get("PASS_TO_PASS"))
    instance_data = {
        "instance_id": instance_id,
        "repo": repo,
        "version": version,
        "base_commit": base_commit,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    # solution/ — gold patch + oracle solve.sh
    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    (sol_dst / "gold.patch").write_text(row.get("patch") or "", encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(base_commit), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)


def _load_rows(hf_repo_id: str, hf_split: str, *, retries: int = 4, backoff_sec: float = 10.0) -> list[dict]:
    """Load HF rows with retries on transient Hub connection errors.

    ``rllm eval`` auto-pulls the dataset before the run, so a one-shot
    ``LocalEntryNotFoundError`` from a DNS/SSL blip would crash the
    whole eval session. Retry a handful of times before giving up —
    snapshot_download caches each successful resolver request, so a
    later attempt resumes from cache instead of re-downloading.
    """
    import time

    from datasets import load_dataset

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ds = load_dataset(hf_repo_id, split=hf_split)
            return [dict(r) for r in ds]
        except Exception as e:  # broad: huggingface_hub raises several distinct types here
            last_exc = e
            if attempt == retries:
                break
            wait = backoff_sec * attempt
            logger.warning("[swegym] load_dataset(%s) failed (attempt %d/%d): %s — retry in %.0fs", hf_repo_id, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({hf_repo_id!r}, split={hf_split!r}) failed after {retries} attempts") from last_exc


def build_benchmark(
    *,
    name: str = "swegym",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "mini-swe-agent",
    hf_repo_id: str | None = None,
    hf_split: str = "train",
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize SWE-Gym into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``,
            ``default_agent``, and ``hf_repo_id`` are read from it when present.
        task_ids: Build only these ``instance_id`` values. Default: all rows.
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_repo_id: Override the HF dataset to pull. Defaults to the
            catalog ``source`` or ``SWE-Gym/SWE-Gym`` (2,438 rows). Set
            to ``SWE-Gym/SWE-Gym-Lite`` for the 230-row eval cut.
        hf_split: HF split to load (SWE-Gym only ships ``train``).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry``.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        hf_repo_id = hf_repo_id or catalog_entry.get("source")
    hf_repo_id = hf_repo_id or HF_REPO_ID

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[swegym] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    test_spec_mod, log_parsers_mod = _import_swegym()
    make_test_spec = test_spec_mod.make_test_spec
    log_parsers_src_path = Path(log_parsers_mod.__file__)

    logger.info("[swegym] loading HF dataset %s split=%s ...", hf_repo_id, hf_split)
    rows = _load_rows(hf_repo_id, hf_split)
    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("instance_id") in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[swegym] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    written = 0
    skipped = 0
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id:
            logger.warning("[swegym] row missing instance_id, skipping")
            skipped += 1
            continue
        task_dst = out / instance_id
        if task_dst.exists():
            shutil.rmtree(task_dst)
        try:
            _materialize_task(task_dst, row, log_parsers_src_path, make_test_spec)
        except KeyError as e:
            # make_test_spec raises KeyError when (repo, version) is unknown
            # to the swegym SPECS map — skip and log rather than aborting.
            logger.warning("[swegym] %s: no eval spec (%s), skipping", instance_id, e)
            shutil.rmtree(task_dst, ignore_errors=True)
            skipped += 1
            continue
        written += 1

    description = (catalog_entry or {}).get("description") or (
        f"SWE-Gym ({hf_repo_id}): real-world Python SWE tasks across expanded repo set, evaluated against pre-built OpenHands testbed images (F2P/P2P pytest grading)."
    )
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[swegym] wrote %d task dirs to %s (skipped %d)", written, out, skipped)

    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = []
            for row in rows:
                iid = row.get("instance_id")
                if not iid:
                    continue
                task_dst = out / iid
                if not (task_dst / "task.toml").exists():
                    continue
                reg_rows.append(
                    {
                        "id": iid,
                        "instruction": (task_dst / "instruction.md").read_text(encoding="utf-8"),
                        "task_path": str(task_dst),
                        "repo": row.get("repo", ""),
                        "version": str(row.get("version", "")),
                    }
                )
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=hf_repo_id,
                description=description,
                category=(catalog_entry or {}).get("category", "code"),
            )
        except Exception:
            logger.warning("[swegym] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.swegym_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize SWE-Gym into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="swegym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-repo-id", default=None, help="Override HF source repo (default: SWE-Gym/SWE-Gym).")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="mini-swe-agent")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("RLLM_LOG_LEVEL", "INFO"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        default_agent=args.default_agent,
        hf_repo_id=args.hf_repo_id,
        hf_split=args.hf_split,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()
