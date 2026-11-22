import json
import logging
from pathlib import Path
import sys

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Household(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    rooms: int
    members: int
    ac_hours_per_day: float = Field(ge=0)
    avg_temp_c: float = Field(le=65)
    house_area_sqft: float
    solar_kw: float
    connection: str
    feeder: str


class PredictionFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"event": record.getMessage(), "level": record.levelname})


log = logging.getLogger("mp132.prediction")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PredictionFormatter())
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

app = FastAPI(title="Patna household electricity use")
model = joblib.load(Path(__file__).resolve().parents[1] / "models/model.joblib")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(household: Household):
    row = pd.DataFrame([household.model_dump()])
    value = float(model.predict(row)[0])
    log.info("prediction")
    return {"prediction": value}
