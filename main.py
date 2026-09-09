from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

model = joblib.load("models/model.pkl")

app = FastAPI(title="Fraud Detection API")

# Chemin du fichier où on va stocker l'historique des transactions reçues
LOG_PATH = "logs/predictions_log.csv"
os.makedirs("logs", exist_ok=True)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data_dict = transaction.dict()

    data = np.array([[
        transaction.Time, transaction.V1, transaction.V2, transaction.V3,
        transaction.V4, transaction.V5, transaction.V6, transaction.V7,
        transaction.V8, transaction.V9, transaction.V10, transaction.V11,
        transaction.V12, transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18, transaction.V19,
        transaction.V20, transaction.V21, transaction.V22, transaction.V23,
        transaction.V24, transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Amount
    ]])

    proba = model.predict_proba(data)[0][1]
    is_fraud = bool(proba > 0.5)

    # --- NOUVEAU : on enregistre cette transaction + son résultat dans le fichier de logs ---
    log_entry = data_dict.copy()
    log_entry["fraud_probability"] = float(proba)
    log_entry["is_fraud"] = is_fraud
    log_entry["timestamp"] = datetime.utcnow().isoformat()

    log_df = pd.DataFrame([log_entry])
    # Si le fichier existe déjà, on ajoute une ligne (mode "a") sans réécrire l'en-tête
    log_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
    # --- FIN DE L'AJOUT ---

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": is_fraud,
        "risk_level": "high" if proba > 0.7 else "medium" if proba > 0.3 else "low"
    }


@app.get("/monitoring/status")
def monitoring_status():
    """Petite route utilitaire pour vérifier combien de transactions ont été loguées."""
    if not os.path.exists(LOG_PATH):
        return {"total_predictions_logged": 0}
    log_df = pd.read_csv(LOG_PATH)
    return {"total_predictions_logged": len(log_df)}

from fastapi.responses import HTMLResponse

@app.get("/monitoring/drift-report", response_class=HTMLResponse)
def drift_report():
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if not os.path.exists(LOG_PATH):
        return "<h1>Aucune transaction enregistrée pour le moment.</h1><p>Effectue d'abord quelques appels à /predict.</p>"

    current_data = pd.read_csv(LOG_PATH)
    current_data = current_data.drop(columns=["fraud_probability", "is_fraud", "timestamp"])
    reference_data = pd.read_csv("data/reference_data.csv")

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=current_data, reference_data=reference_data)

    my_eval.save_html("temp_report.html")
    with open("temp_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content