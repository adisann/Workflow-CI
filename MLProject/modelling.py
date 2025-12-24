
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os

# Init DagsHub Connection
# GitHub Actions environment variables
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")
DAGSHUB_REPO_NAME = "Eksperimen_SML"

if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    # Set Tracking URI manually to avoid dagshub.init() auth issues
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
else:
    print("Warning: DAGSHUB_USERNAME or DAGSHUB_TOKEN not found. Tracking may fail.")

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def main():
    print("Mulai Training...")
    # Load Data
    try:
        df = pd.read_csv('data/processed/train_processed.csv')
    except FileNotFoundError:
        print("Data processed tidak ditemukan. Pastikan preprocessing sudah jalan.")
        return

    # Ambil 1 SKU saja untuk demo cepat
    target_sku = 216418
    print(f"Filtering for SKU: {target_sku}")
    
    # Debug: Check avaliable SKUs
    print(f"Available SKUs in data: {df['sku_id'].unique()}")
    
    df = df[df['sku_id'] == target_sku]
    
    if df.empty:
        print(f"Error: No data found for SKU {target_sku}!")
        return
    
    # Pilih Fitur
    features = [c for c in df.columns if 'lag_' in c or 'ma_' in c or 'store_encoded' in c or 'total_price' in c]
    X = df[features]
    y = df['units_sold']
    
    mlflow.set_experiment("Retail_Demand_Ops")

    # --- OPTUNA TUNING ---
    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 5)
            }
            
            model = RandomForestRegressor(**params, random_state=42)
            
            # Validasi sederhana
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.maximum(preds, 0)
            
            error = rmsle(y_val, preds)
            
            # Log ke DagsHub
            mlflow.log_params(params)
            mlflow.log_metric("rmsle", error)
            
            return error

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3) # 3x percobaan saja biar cepat
    
    # --- FINAL TRAINING ---
    with mlflow.start_run(run_name="Champion_Model"):
        best_params = study.best_params
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X, y)
        
        # 1. Log Model ke DagsHub
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(best_params)
        
        # 2. Buat Artifact (Grafik)
        plt.figure(figsize=(8, 5))
        pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        
        # 3. Upload Artifact ke DagsHub
        mlflow.log_artifact("feature_importance.png")
        
        # 4. Upload Artifact Tambahan (Requirement) - Agar poin Advance (Artifact > 2)
        mlflow.log_artifact("Membangun_model/requirements.txt")
        print("Model, Feature Importance, dan Requirements berhasil di-log ke DagsHub!")

if __name__ == "__main__":
    main()
