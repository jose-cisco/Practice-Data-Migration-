# Databricks notebook source
# ============================================================================
# config_lib — config layer for the OLD workspace (Databricks Free Edition).
# Import from a task notebook (own cell, relative path):
#   %run ../lib/config_lib
# Import from a migration_practice-root notebook (e.g. 16_config_pattern):
#   %run ./lib/config_lib
#
# OLD-WORKSPACE RULES (this file):
#   - Tables use 2-PIECE names only: schema.table  (no catalog).
#   - The catalog appears in exactly TWO platform-forced places (Free Edition
#     Unity Catalog has no 2-part form for these):
#       * the model registry  -> CFG.model("m") = <current_catalog>.default.m
#       * UC Volumes          -> data_catalog()  (35_export_tables staging)
#   - The FULL catalog-driven CFG (CATALOG/MODEL_CATALOG widgets, dev/stg/prd
#     stage catalogs, 3-part table names) lives ONLY on the new platform:
#     bundle/src/lib/config_lib.py.
#
# Real-world pattern this mirrors:
#   run_date = dbutils.widgets.get("run_date")
#   CFG = setup_config(config_path=CONFIG_PATH, run_date=run_date)   # run_date REQUIRED
#   df = spark.read.table(CFG.table("schema_name.table_name"))
#
# Forgetting run_date reproduces the real error:
#   TypeError: setup_config() missing 1 required positional argument: 'run_date'
#
# Partition discovery uses the corrected SHOW PARTITIONS pattern (the raw
# output has ONE string column named `partition`, e.g. 'event_partition_date=
# 2026-10-18', so it cannot be date-ordered directly — extract first).
#
# Also provides the old-school project bootstrap (production line kept verbatim):
#   PROJECT_PATH = setup_wd_path(dbutils)
#   REQ_PATH = prepare_requirements(PROJECT_PATH)   # stages a SPACE-FREE /tmp copy
#   %pip install -r "$REQ_PATH" -q                   # <- unchanged production line
# (The raw line fails when the project path has spaces — see runbook Appendix A.4.)
# ============================================================================

# COMMAND ----------

import re

from pyspark.sql import functions as F

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_DATA_CATALOG_CACHE = []


def _identifier(value):
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""), f"Invalid identifier: {value!r}"
    return value

# COMMAND ----------

def data_catalog():
    """The ONLY catalog lookup on the old workspace (cached).

    Free Edition Unity Catalog has no 2-part form for the model registry or
    for Volumes, so those two features need the workspace catalog name. Tables
    never use it (2-part schema.table everywhere else). Returns None on a
    classic/Hive workspace (spark_catalog / hive_metastore) — callers fall
    back to plain names / driver /tmp paths.
    """
    if _DATA_CATALOG_CACHE:
        return _DATA_CATALOG_CACHE[0]
    try:
        current = spark.sql("SELECT current_catalog()").collect()[0][0]
    except Exception:
        current = "spark_catalog"
    if current in ("spark_catalog", "", None, "hive_metastore"):
        current = None
    _DATA_CATALOG_CACHE.append(current)
    return current

# COMMAND ----------

def latest_partition(table, k=0, column="ingest_date"):
    """k-th newest partition date of `table` as 'YYYY-MM-DD' (k=0 latest, k=1 prev).

    Corrected real-world pattern — SHOW PARTITIONS returns ONE string column named
    `partition` ('event_partition_date=2026-10-18'), so this is what the original
    `.orderBy("event_partition_date")` SHOULD have been:

        spark.sql(f"SHOW PARTITIONS {table}")
            .select(F.regexp_extract("partition", r"(\\d{4}-\\d{2}-\\d{2})", 1).alias("dt"))
            .where(F.col("dt") != "")
            .orderBy(F.desc("dt"))

    Falls back to `SELECT max(column)` (the real pipeline's simpler variant) when
    SHOW PARTITIONS is unusable (unpartitioned table / no rows / k must be 0).
    """
    try:
        rows = (
            spark.sql(f"SHOW PARTITIONS {table}")
            .select(F.regexp_extract("partition", r"(\d{4}-\d{2}-\d{2})", 1).alias("dt"))
            .where(F.col("dt") != "")
            .orderBy(F.desc("dt"))
            .collect()
        )
    except Exception:
        rows = []

    if rows:
        if len(rows) <= k:
            raise ValueError(f"{table}: only {len(rows)} partition(s), need k={k}+1")
        return rows[k]["dt"]

    if k > 0:
        raise ValueError(f"{table}: SHOW PARTITIONS returned nothing usable — k>0 unsupported here")
    row = spark.sql(f"SELECT max({_identifier(column)}) AS d FROM {table}").collect()
    if not row or row[0]["d"] is None:
        raise ValueError(f"{table}: no partitions found — run 00_setup_sample_data first")
    return str(row[0]["d"])

# COMMAND ----------

class Config:
    """CFG object — schema/run_date config for the OLD workspace.

    NO catalog on this side: CFG.table() always returns 2-part schema.table.
    The single catalog exception is model naming (Free Edition UC registry
    REQUIRES catalog.schema.name — 2-part namespace alone is rejected):

    CFG.table("t")                 -> schema.t                        (2-part)
    CFG.table("schema.table")      -> honors the embedded schema
    CFG.model("m")                 -> <current_catalog>.default.m     (registry)
                                      load via mlflow.pyfunc.load_model(
                                      f"models:/{CFG.model('m')}@latest")
    CFG.partition_filter(df, tbl)  -> applies run_date (all|latest|prev|date)
    CFG.write_partitioned(df, "t") -> dynamic-partition-overwrite Delta write
    """

    def __init__(self, schema, run_date):
        self.schema = schema
        self.run_date = run_date        # all | latest | prev | YYYY-MM-DD

    def model(self, simple_name):
        """Model-registry name — the ONLY catalog use on the old side.
        Free Edition (Unity Catalog) requires catalog.schema.model for
        registration and inference; classic/Hive workspaces fall back to the
        plain name (data_catalog() returns None there)."""
        name = _identifier(simple_name)
        catalog = data_catalog()
        if catalog is None:
            return name
        return f"{catalog}.default.{name}"

    def table(self, simple_name):
        parts = simple_name.split(".")
        if len(parts) == 2:
            schema, name = parts
        elif len(parts) == 1:
            schema, name = self.schema, simple_name
        else:
            raise ValueError(f"Bad table name (old workspace = 2-part schema.table): {simple_name!r}")
        return f"{_identifier(schema)}.{_identifier(name)}"

    def resolved_partition_date(self, table, column="ingest_date"):
        """Concrete 'YYYY-MM-DD' for latest/prev/date; None for 'all' (no filter)."""
        mode = (self.run_date or "all").strip().lower()
        if mode == "all":
            return None
        if mode == "latest":
            return latest_partition(table, k=0, column=column)
        if mode == "prev":
            return latest_partition(table, k=1, column=column)
        if _DATE_RE.fullmatch(mode):
            return mode
        raise ValueError(f"run_date/PARTITION must be all|latest|prev|YYYY-MM-DD, got {self.run_date!r}")

    def partition_filter(self, df, table, column="ingest_date"):
        date = self.resolved_partition_date(table, column=column)
        if date is None:
            return df
        return df.filter(F.col(column) == F.lit(date).cast("date"))

    def write_partitioned(self, df, simple_name, column="ingest_date"):
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .partitionBy(column)
            .saveAsTable(self.table(simple_name))
        )

# COMMAND ----------

def setup_config(config_path, run_date, schema=None):
    """Real-world config entrypoint (old workspace). run_date is REQUIRED on
    purpose — the real C360 setup_config has the same signature, and omitting
    it raises:
        TypeError: setup_config() missing 1 required positional argument: 'run_date'

    CFG = setup_config(config_path=CONFIG_PATH, run_date=run_date)

    run_date accepts the practice widget semantics: 'all' | 'latest' | 'prev' | 'YYYY-MM-DD'.
    config_path is accepted for signature parity with the real pipeline (and the
    new-platform lib) but the old workspace has no stage catalogs to resolve.
    """
    return Config((schema or "default").strip(), run_date)

# COMMAND ----------

# ----------------------------------------------------------------------------
# Training-table utils — ONE definition of the table chosen for model training
# (00 pins the choice; 01-05 train on it; 30/65/15_model_task use the same
# feature assembly).
# ----------------------------------------------------------------------------

FEATURES = ["tenure_months", "churn_risk", "seg_enterprise", "seg_retail"]
LABEL = "monthly_revenue"


def features_from_df(sdf):
    """Pandas frame (customer_id, ingest_date, FEATURES..., label) from a Spark
    practice-table DataFrame, ordered by customer_id — lets the job tasks
    (15_model_task) score a partition-filtered read with the same feature
    assembly used in training."""
    return (
        sdf.select(
            F.col("customer_id"),
            F.col("ingest_date"),
            F.col("tenure_months").cast("double").alias("tenure_months"),
            F.col("churn_risk").cast("double").alias("churn_risk"),
            (F.col("segment") == "enterprise").cast("double").alias("seg_enterprise"),
            (F.col("segment") == "retail").cast("double").alias("seg_retail"),
            F.col(LABEL).cast("double").alias("label"),
        )
        .orderBy("customer_id")
        .toPandas()
    )


def _features_pandas(table):
    """Feature/label frame from a practice table (segment one-hot encoded),
    ordered by customer_id so parity rows are deterministic."""
    return features_from_df(spark.table(table))


def load_training_data(table):
    """(X, y, ids) numpy arrays for the table chosen in 00 (workflow notebooks)."""
    pdf = _features_pandas(table)
    return (
        pdf[FEATURES].to_numpy(dtype=float),
        pdf["label"].to_numpy(dtype=float),
        pdf["customer_id"].to_numpy(),
    )


def parity_rows(table, n=5):
    """First-n (by customer_id) feature rows — the fixed parity reference."""
    pdf = _features_pandas(table).head(n)
    return pdf[FEATURES].to_numpy(dtype=float), pdf["customer_id"].tolist()

# COMMAND ----------

# ----------------------------------------------------------------------------
# Project bootstrap — old-school requirements install, made space-safe.
# Production shape (kept verbatim; see runbook Appendix A.4 for the error it
# originally produced when the project path contained spaces):
#
#   PROJECT_PATH = setup_wd_path(dbutils)
#   REQ_PATH = os.path.join(PROJECT_PATH, "requirements.txt")
#   %pip install -r "$REQ_PATH" -q
#
# The %pip magic word-splits the interpolated path on whitespace (no quote
# handling), so a path like 'Shared/DS Use Case/...' made pip receive
# 'Case/biap-.../requirements.txt' as a package REQUIREMENT:
#   ERROR: Invalid requirement: 'Case/.../requirements.txt'
# prepare_requirements() stages the file at a SPACE-FREE absolute path so the
# SAME install line works unchanged on DBR 12.2 and serverless.
# ----------------------------------------------------------------------------

REQ_TMP_PATH = "/tmp/practice_requirements.txt"
REQ_DEFAULTS = "xgboost\n"


def setup_wd_path(dbutils):
    """Absolute project-root path (real-world utils/setup_path.setup_wd_path pattern).

    Resolves from THIS notebook's workspace path, walking up out of known
    subfolders (tasks/, lib/, old_ws/, new_ws/). Returns a /Workspace absolute
    path; readability is probed separately (DBR 12.2 has no /Workspace FUSE,
    so prepare_requirements falls back to built-ins there).
    """
    try:
        ctx = dbutils.notebook.getContext().notebookPath()
        nb_path = ctx.get() if ctx.isDefined() else ""
    except Exception:
        nb_path = ""

    root = nb_path.rsplit("/", 1)[0] if "/" in nb_path else ""
    while root and root.rsplit("/", 1)[-1] in ("tasks", "lib", "old_ws", "new_ws"):
        root = root.rsplit("/", 1)[0]

    if not root or root.startswith("/Workspace"):
        return root or "/Workspace"
    return "/Workspace" + root


def prepare_requirements(project_path=None, defaults=REQ_DEFAULTS, tmp_path=REQ_TMP_PATH):
    """Stage requirements at a SPACE-FREE absolute path; return it as REQ_PATH.

    Source resolution: <project_path>/requirements.txt when readable, else the
    built-in default list (DBR 12.2 cannot read /Workspace files at all).
    Then the unchanged production line works:
        %pip install -r "$REQ_PATH" -q
    """
    import os
    import shutil

    src = None
    if project_path:
        candidate = os.path.join(project_path, "requirements.txt")
        if os.path.exists(candidate):
            src = candidate

    if src is None:
        text = str(defaults)
        with open(tmp_path, "w") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        print(f"requirements staged at {tmp_path} (built-in default: {str(defaults).strip()})")
        print("  source file not readable on this runtime — e.g. DBR 12.2 has no /Workspace FUSE")
    else:
        shutil.copy(src, tmp_path)
        print(f"requirements staged at {tmp_path} (source: {src})")

    return tmp_path
