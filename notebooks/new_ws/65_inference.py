# Databricks notebook source
# 65_inference  (NEW workspace) — completes the notebook arc:
#
# setup -> table choice -> training (old) -> migration -> inference (new).
# Loads the migrated linear-regression model @latest and scores the lifted
# practice_customers_clean table with the SAME feature assembly as training.
# Display-only (no table writes; sklearn is preinstalled — no %pip needed).
# ONE ACCESS, TWO CATALOGS (Appendix A.5): the lifted tables via CFG.table()
# and the model registry via CFG.model() — both through the same CFG.
# Widgets: MODEL_CATALOG locates the UC model registry (default practice_ml);
#          CATALOG/SCHEMA locate the lifted tables (auto/practice).

# COMMAND ----------



%run ./lib/config_lib

# COMMAND ----------



dbutils.widgets.text("MODEL_CATALOG", "practice_ml", "Model registry catalog")
dbutils.widgets.text("CATALOG", "auto", "Catalog of the lifted tables ('auto' = detect)")
dbutils.widgets.text("SCHEMA", "practice", "Schema of the lifted tables (OLD schema name)")

CFG = setup_config(
    config_path="",
    run_date="all",
    catalog=dbutils.widgets.get("CATALOG").strip(),
    schema=dbutils.widgets.get("SCHEMA").strip(),
    model_catalog=dbutils.widgets.get("MODEL_CATALOG").strip(),
)

# COMMAND ----------



import mlflow
import pandas as pd

MODEL_NAME = CFG.model("practice_linear_regression")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@latest")
print(f"Loaded {MODEL_NAME} @latest (mlflow {mlflow.__version__})")

# COMMAND ----------



# Score the lifted table with the same feature assembly used in training.
X, y, ids = load_training_data(CFG.table("practice_customers_clean"))
preds = model.predict(X)

out = pd.DataFrame({
    "customer_id": ids,
    "monthly_revenue": y,
    "predicted_revenue": preds,
})
display(out)

rmse = float(((out["monthly_revenue"] - out["predicted_revenue"]) ** 2).mean() ** 0.5)
print(f"rows scored    : {len(out)}")
print(f"RMSE vs actual : {rmse:.4f} (informational — the acceptance test is 60_verify_models)")
print("\nINFERENCE DONE — setup -> table choice -> training -> migration -> inference arc complete.")
