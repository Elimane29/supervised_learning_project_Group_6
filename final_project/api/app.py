"""Prediction endpoint: one discharged encounter in, calibrated 30-day readmission risk out.

Run:  uvicorn api.app:app --reload      then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from readmission.features import MEDICATIONS
from readmission.predict import load_bundle, score

MedStatus = Literal["No", "Steady", "Up", "Down"]


class EncounterBase(BaseModel):
    """Information available at discharge (prediction-time test, Session 1)."""
    gender: Literal["Female", "Male"] = "Female"
    age: str = Field("[70-80)", description="bracket, e.g. [70-80)")
    admission_type_id: int = 1
    discharge_disposition_id: int = Field(1, description="1 = discharged home")
    admission_source_id: int = 7
    time_in_hospital: int = Field(4, ge=1, le=14)
    payer_code: str = "?"
    medical_specialty: str = "?"
    num_lab_procedures: int = 44
    num_procedures: int = 1
    num_medications: int = 16
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = Field(0, description="inpatient visits in the previous year")
    diag_1: str = Field("428", description="primary ICD-9 code")
    diag_2: str = "250"
    diag_3: str = "401"
    number_diagnoses: int = 8
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "Yes"
    n_prior_encounters: int = Field(0, ge=0, description="earlier stays in hospital records")


# The 21 medication columns share one type and default: generate them instead of listing.
Encounter = create_model("Encounter", __base__=EncounterBase,
                         **{m: (MedStatus, "No") for m in MEDICATIONS})


class Prediction(BaseModel):
    probability_readmission_30d: float
    flag_for_follow_up: bool
    threshold: float


app = FastAPI(title="30-day readmission risk",
              description="Diabetes 130-US hospitals - calibrated risk for bed planning.")


@app.get("/health")
def health() -> dict:
    b = load_bundle()
    return {"status": "ok", "model": b["model_name"], "calibration": b["calibration"],
            "threshold": b["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(encounter: Encounter) -> Prediction:
    row = pd.DataFrame([encounter.model_dump()])
    return Prediction(**score(row).iloc[0].to_dict())


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(encounters: list[Encounter]) -> list[Prediction]:
    rows = pd.DataFrame([e.model_dump() for e in encounters])
    return [Prediction(**r) for r in score(rows).to_dict(orient="records")]
