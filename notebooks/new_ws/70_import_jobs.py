# Databricks notebook source
# 70_import_jobs  (NEW workspace) — Jobs Migration steps 2-3 (Option A, in-notebook)
#
# Lift-and-shift the 5 practice jobs from jobs_export.zip (6 job objects:
# 5 practice jobs + the internal practice-job-4b-target that J4 triggers):
#   1. Unpack jobs_export.json from the staging volume
#   2. Adapt definitions: serverless compute (drop old clusters),
#      remap task notebook paths, inject CATALOG/SCHEMA/PARTITION, PAUSE schedules
#   3. Create all jobs via Jobs API 2.1 (dependent trigger remapped last)
#   4. Trigger one validation run each and print a PASS/FAIL matrix
# PREREQ: task notebooks imported at /Users/<you>/migration_practice/tasks/
#         and lib/config_lib imported at /Users/<you>/migration_practice/lib/
#         and 00_setup_sample_data run once in THIS workspace.
# (Option B — the video-style Databricks Asset Bundle — is in /bundle.)

# COMMAND ----------



dbutils.widgets.text("token", "", "Personal access token (NEW workspace)")
dbutils.widgets.text("CATALOG", "practice_ml", "Landing-zone catalog (zip location)")
dbutils.widgets.text("SCHEMA", "models", "Landing-zone schema (zip location)")
dbutils.widgets.text("VOLUME", "staging_artifacts", "Landing-zone volume (zip location)")
dbutils.widgets.text("JOB_CATALOG", "auto", "Catalog for imported jobs ('auto' = detect)")
dbutils.widgets.text("JOB_SCHEMA", "practice", "Schema for imported jobs (OLD schema name)")
dbutils.widgets.text("JOB_MODEL_CATALOG", "practice_ml", "Model registry catalog for imported jobs")

TOKEN = dbutils.widgets.get("token").strip()
CATALOG = dbutils.widgets.get("CATALOG").strip()
SCHEMA = dbutils.widgets.get("SCHEMA").strip()
VOLUME = dbutils.widgets.get("VOLUME").strip()
JOB_CATALOG = dbutils.widgets.get("JOB_CATALOG").strip()
JOB_SCHEMA = dbutils.widgets.get("JOB_SCHEMA").strip()
JOB_MODEL_CATALOG = dbutils.widgets.get("JOB_MODEL_CATALOG").strip()
assert TOKEN, "Paste a NEW-workspace personal access token into the 'token' widget"

# Environment parameters injected into every notebook task of the imported jobs.
# Use JOB_CATALOG/JOB_SCHEMA to match where the tables were lifted (Phase 2.4)
# and JOB_MODEL_CATALOG to point model tasks at the new-side registry.
ENV_PARAMS = {"CATALOG": JOB_CATALOG, "SCHEMA": JOB_SCHEMA, "PARTITION": "all",
              "CONFIG_PATH": "", "MODEL_CATALOG": JOB_MODEL_CATALOG}

# COMMAND ----------



import json, os, shutil, time
import requests

try:
    HOST = dbutils.notebook.getContext().apiUrl().get().rstrip("/")
except Exception:
    dbutils.widgets.text("host", "", "New workspace URL")
    HOST = dbutils.widgets.get("host").strip().rstrip("/")

USER = dbutils.notebook.getContext().userName().get()
TASKS_NEW = f"/Users/{USER}/migration_practice/tasks"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}"})

def api(method, path, body=None, params=None):
    r = S.request(method, f"{HOST}{path}", json=body, params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()

print(f"Target workspace : {HOST}")
print(f"Task notebooks   : {TASKS_NEW}")

# COMMAND ----------



ZIP_PATH = f"{VOL_ROOT}/jobs_export.zip"
assert os.path.exists(ZIP_PATH), f"{ZIP_PATH} not found — upload jobs_export.zip into the volume first."
UNPACK_DIR = f"{VOL_ROOT}/jobs_export"
shutil.rmtree(UNPACK_DIR, ignore_errors=True)
shutil.unpack_archive(ZIP_PATH, UNPACK_DIR)

with open(f"{UNPACK_DIR}/jobs_export.json") as f:
    exports = json.load(f)
assert len(exports) == 6, (
    f"Expected 6 job objects (5 practice jobs + practice-job-4b-target), found {len(exports)}"
)
print(f"Loaded {len(exports)} exported job definitions:")
for e in exports:
    print(f"  - {e['name']}")

# COMMAND ----------



source_id_to_name = {str(e["source_job_id"]): e["name"] for e in exports}

def adapt_task(t):
    t = json.loads(json.dumps(t))
    for k in ("existing_cluster_id", "new_cluster", "libraries", "job_cluster_key"):
        t.pop(k, None)
    nb = t.get("notebook_task")
    if nb and "notebook_path" in nb:
        nb["notebook_path"] = f"{TASKS_NEW}/{os.path.basename(nb['notebook_path'])}"
        # 70's new-side env params WIN over the exported ones: the import must
        # re-point jobs at the new workspace's tables AND model registry (the
        # exported "MODEL_CATALOG": "auto" would resolve to the OLD registry).
        nb["base_parameters"] = {**(nb.get("base_parameters") or {}), **ENV_PARAMS}
    return t

def adapt_settings(e):
    s = json.loads(json.dumps(e["settings"]))
    s.pop("job_id", None)
    tasks = [adapt_task(t) for t in s.get("tasks", [])]
    s["tasks"] = tasks
    sch = s.get("schedule")
    if sch:
        sch["pause_status"] = "PAUSED"
    return s

adapted = {e["name"]: adapt_settings(e) for e in exports}
print("Adapted for the new workspace: serverless compute, remapped task paths, PAUSED schedules,")
print(f"injected env params: {ENV_PARAMS}")

# COMMAND ----------



created, failures = {}, []
for name, settings in adapted.items():
    has_run_job = any("run_job_task" in t for t in settings["tasks"])
    if has_run_job:
        continue
    try:
        result = api("POST", "/api/2.1/jobs/create", settings)
        created[name] = result["job_id"]
        print(f"CREATED  {name}  job_id={result['job_id']}")
    except Exception as ex:
        failures.append((name, str(ex)))
        print(f"FAILED   {name}: {ex}")

# COMMAND ----------



# Dependent trigger last: remap run_job_task to the NEW target job id.
for name, settings in adapted.items():
    if name in created:
        continue
    try:
        for t in settings["tasks"]:
            rj = t.get("run_job_task")
            if rj and "job_id" in rj:
                old_target = source_id_to_name.get(str(rj["job_id"]))
                assert old_target and old_target in created, f"Target job for {name} not created yet"
                rj["job_id"] = created[old_target]
                print(f"REMAPPED  {name}: run_job -> {old_target} (job_id={rj['job_id']})")
        result = api("POST", "/api/2.1/jobs/create", settings)
        created[name] = result["job_id"]
        print(f"CREATED  {name}  job_id={result['job_id']}")
    except Exception as ex:
        failures.append((name, str(ex)))
        print(f"FAILED   {name}: {ex}")

assert len(created) == 6, f"Expected 6 created job objects, got {len(created)} — see failures above"
print(f"\nAll job objects lift-and-shifted: {HOST}/jobs")

# COMMAND ----------



# Validation: one run for each of the 5 practice jobs (the 4b target runs
# inside the dependent job, so it is not fired separately), PASS/FAIL matrix.
def run_once(name, job_id, timeout_secs=600):
    run_id = api("POST", "/api/2.1/jobs/run-now", {"job_id": job_id})["run_id"]
    started = time.time()
    while time.time() - started < timeout_secs:
        run = api("GET", "/api/2.1/jobs/runs/get", None, {"run_id": run_id})
        state = run.get("state", {})
        if state.get("life_cycle_state") == "TERMINATED":
            ok = state.get("result_state") == "SUCCEEDED"
            return ok, f"result_state={state.get('result_state')}"
        time.sleep(15)
    return False, "timeout"

results = []
for name, job_id in sorted(created.items()):
    if name.endswith("-4b-target"):
        continue
    ok, detail = run_once(name, job_id)
    results.append({"job": name, "run_validation": "PASS" if ok else "FAIL", "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")

import pandas as pd
display(pd.DataFrame(results))
assert len(results) == 5, f"Expected 5 practice-job validations, got {len(results)}"
assert all(r["run_validation"] == "PASS" for r in results), "At least one job failed — check its run output in the UI"
print("\nAll 5 migrated jobs ran green. Schedules are PAUSED (zero cost).")
print("Update MIGRATION_CHECKLIST.md rows J1-J5.")
