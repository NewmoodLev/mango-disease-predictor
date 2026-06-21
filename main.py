"""
FastAPI backend — Mango Disease Predictor
รัน: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "farmer_app"))
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

import orchard_core as core

app = FastAPI(title="Mango Disease Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/provinces")
async def get_provinces():
    return list(core.THAI_LOCATIONS.keys())


class WeatherReq(BaseModel):
    province: str


@app.post("/api/weather")
async def get_weather(req: WeatherReq):
    try:
        if req.province not in core.THAI_LOCATIONS:
            raise HTTPException(404, "ไม่พบจังหวัด")
        lat, lon = core.THAI_LOCATIONS[req.province]
        result = core.fetch_live_weather(lat, lon)
        if result is None:
            raise HTTPException(503, "ดึงข้อมูลอากาศไม่สำเร็จ")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}")


class OrchardReq(BaseModel):
    rows: int = 10
    cols: int = 10
    row_spacing: float = 6.0
    tree_spacing: float = 6.0


@app.post("/api/orchard")
async def build_orchard(req: OrchardReq):
    try:
        positions, grid_index = core.build_orchard_layout(
            req.rows, req.cols, req.row_spacing, req.tree_spacing
        )
        return {
            "positions": positions.tolist(),
            "grid_index": [[r, c] for r, c in grid_index],
        }
    except Exception as e:
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}")


class PredictReq(BaseModel):
    rows: int = 10
    cols: int = 10
    row_spacing: float = 6.0
    tree_spacing: float = 6.0
    infected: List[int]
    severity: float = 0.4
    humidity: float = 0.6
    rainfall: float = 1.0
    temperature: float = 28.0
    wind: float = 1.0
    horizon: int = 7
    variety: Optional[str] = "NamDokMai"


@app.post("/api/predict")
async def run_predict(req: PredictReq):
    if not req.infected:
        raise HTTPException(400, "กรุณาเลือกต้นที่เป็นโรคอย่างน้อย 1 ต้น")
    try:
        positions, grid_index = core.build_orchard_layout(
            req.rows, req.cols, req.row_spacing, req.tree_spacing
        )
        N = len(positions)
        spacing = min(req.row_spacing, req.tree_spacing)

        scenario = core.match_weather_to_scenario(
            req.humidity, req.rainfall, req.temperature, req.wind
        )

        ei, ew   = core.build_graph(positions, spacing)
        variety  = req.variety or "NamDokMai"

        # สร้าง feature window จาก SEIR simulation (ตรง distribution ของ training)
        feats = core.synthesize_features(
            positions   = positions,
            edge_index  = ei,
            edge_weight = ew,
            infected_idx= req.infected,
            severity    = req.severity,
            scenario    = scenario,
            variety     = variety,
            age_years   = 6,
            health      = 0.75,
        )

        # รัน ST-GNN model
        horizons = core.list_available_horizons(scenario)
        use_h    = req.horizon if req.horizon in horizons else (horizons[0] if horizons else None)
        if use_h is None:
            raise HTTPException(500, f"ไม่พบโมเดล — scenario={scenario} ROOT={core.ROOT}")

        risk = core.predict(feats, ei, ew, scenario, use_h)
        risk = np.clip(risk, 0.0, 1.0)

        avg  = float(risk.mean())
        high = int((risk >= 0.66).sum())
        mid  = int(((risk >= 0.33) & (risk < 0.66)).sum())
        low  = int((risk < 0.33).sum())

        csv_rows = []
        for i in range(N):
            r, c = grid_index[i]
            level = "สูง" if risk[i] >= 0.66 else ("ปานกลาง" if risk[i] >= 0.33 else "ต่ำ")
            csv_rows.append({
                "ต้น": i, "แถว": r + 1, "ต้นที่": c + 1,
                "ความเสี่ยง_%": round(float(risk[i]) * 100, 1), "ระดับ": level
            })

        return {
            "risk":        risk.tolist(),
            "avg_risk":    avg,
            "high":        high,
            "mid":         mid,
            "low":         low,
            "scenario":    scenario,
            "scenario_th": core.SCENARIO_LABELS_TH.get(scenario, scenario),
            "horizon":     use_h,
            "csv_rows":    csv_rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
