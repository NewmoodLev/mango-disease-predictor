"""
Core logic for the Farmer Disease-Forecast App
==============================================
Reusable, UI-agnostic functions that:
  1. Build an orchard graph from a farmer-defined layout (rows / trees / spacing)
  2. Synthesize the 7-day temporal feature window the STGNN expects, from a
     simple "which trees are infected today" selection
  3. Pick the right weather-scenario model and run a 7- or 14-day forecast

The graph construction mirrors create_real_graph.py (distance + wind + water
kernels) so the input distribution matches what the scenario models were
trained on. The trained models are graph-size agnostic (the GNN shares weights
across nodes), so a custom orchard layout works directly.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import torch

from stgnn_model import SpatialTemporalGNN

# Project root = parent of this file's folder (files live next to the .py scripts)
ROOT = Path(__file__).resolve().parent.parent

WINDOW_SIZE = 7

# Variety -> base resistance (from create_real_graph.py, ref Arauz 2000)
RESISTANCE_MAP = {
    "NamDokMai": 0.3,   # มะม่วงน้ำดอกไม้ - อ่อนแอต่อโรคมาก
    "Irwin": 0.6,       # อ่อนแอปานกลาง
    "Keitt": 0.8,       # ทนทาน
}

# Weather scenario reference profiles (from predict_with_scenarios.py)
SCENARIO_PROFILES = {
    "HEAVY_RAIN": {"humidity": 0.80, "rainfall": 2.5, "temperature": 26.0, "wind": 1.2},
    "DROUGHT":    {"humidity": 0.35, "rainfall": 0.2, "temperature": 31.0, "wind": 1.5},
    "WINDY":      {"humidity": 0.55, "rainfall": 1.0, "temperature": 28.0, "wind": 2.0},
    "HOT_HUMID":  {"humidity": 0.75, "rainfall": 1.2, "temperature": 32.0, "wind": 0.8},
    "COLD_DRY":   {"humidity": 0.30, "rainfall": 0.3, "temperature": 22.0, "wind": 1.8},
    "NORMAL":     {"humidity": 0.60, "rainfall": 1.0, "temperature": 28.0, "wind": 1.0},
}

# Human-friendly Thai labels for the weather scenarios
SCENARIO_LABELS_TH = {
    "HEAVY_RAIN": "ฝนตกหนัก / ชื้นจัด",
    "DROUGHT":    "แล้ง / อากาศแห้ง",
    "WINDY":      "ลมแรง",
    "HOT_HUMID":  "ร้อนชื้น",
    "COLD_DRY":   "เย็นแห้ง",
    "NORMAL":     "ปกติ",
}


# ──────────────────────────────────────────────────────────
# 1. GRAPH FROM FARMER LAYOUT
# ──────────────────────────────────────────────────────────
def build_orchard_layout(rows: int, cols: int,
                         row_spacing: float, tree_spacing: float):
    """
    Create a regular grid of trees.

    rows         : จำนวนแถว
    cols         : จำนวนต้นต่อแถว
    row_spacing  : ระยะห่างระหว่างแถว (เมตร)
    tree_spacing : ระยะห่างระหว่างต้นในแถวเดียวกัน (เมตร)

    Returns positions (N, 2) and a (row, col) index list, row-major.
    """
    positions = []
    grid_index = []
    for r in range(rows):
        for c in range(cols):
            positions.append([c * tree_spacing, r * row_spacing])
            grid_index.append((r, c))
    return np.array(positions, dtype=np.float32), grid_index


def build_graph(positions: np.ndarray, spacing: float):
    """
    Build edge_index / edge_weight using the same kernels as the training graph:
    distance decay + NE-wind boost + downhill-water boost.

    max_distance scales with spacing so each tree connects to ~its neighbours
    regardless of how wide the farmer plants.
    """
    N = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]

    wind_dir = np.array([1.0, 1.0]) / np.sqrt(2)   # prevailing NE wind
    water_dir = np.array([0.0, -1.0])              # runoff downhill (-Y)

    max_distance = 2.6 * spacing                    # ~2 rings of neighbours

    edge_list, edge_weight = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            vec = np.array([x[j] - x[i], y[j] - y[i]])
            d = float(np.linalg.norm(vec))
            if d <= max_distance:
                w_dist = np.exp(-d / (spacing if spacing > 0 else 6.0))
                norm_vec = vec / (d + 1e-6)
                w_wind = 1.0 + max(0.0, float(np.dot(norm_vec, wind_dir))) * 1.2
                w_water = 1.0 + max(0.0, float(np.dot(norm_vec, water_dir))) * 0.5
                edge_list.append([i, j])
                edge_weight.append(w_dist * w_wind * w_water)

    if not edge_list:                               # single tree fallback
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)
    else:
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_weight = np.array(edge_weight, dtype=np.float32)

    # add self loops (model expects them; see SpatialBlock note)
    self_loops = np.stack([np.arange(N), np.arange(N)])
    self_w = np.ones(N, dtype=np.float32)
    edge_index = np.concatenate([edge_index, self_loops], axis=1)
    edge_weight = np.concatenate([edge_weight, self_w])

    return (torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_weight, dtype=torch.float32))


# ──────────────────────────────────────────────────────────
# 2. SYNTHESIZE 7-DAY FEATURE WINDOW
# ──────────────────────────────────────────────────────────
def synthesize_features(positions: np.ndarray,
                        edge_index: "torch.Tensor",
                        edge_weight: "torch.Tensor",
                        infected_idx,
                        severity: float,
                        scenario: str,
                        variety: str,
                        age_years: float,
                        health: float,
                        max_age_years: float = 25.0):
    """
    สร้าง (T=7, N, 4) feature window โดยใช้ SEIR simulation
    จำลองย้อนหลัง 7 วันที่ผ่านมา เพื่อให้ input ตรงกับ distribution
    ที่ model เห็นตอน training (ต้นข้างเคียงมีค่า I เล็กน้อยจากการแพร่)

    Features: [Infection, Delta, Age_norm, Health]
    """
    N = len(positions)

    # สร้าง adjacency matrix (ไม่มี self-loop) สำหรับ SEIR
    ei = edge_index.numpy()
    ew = edge_weight.numpy()
    mask = ei[0] != ei[1]
    rows, cols, data = ei[0][mask], ei[1][mask], ew[mask]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[rows, cols] = data

    sm       = _SPREAD_MULT.get(scenario, 1.0)
    resist   = RESISTANCE_MAP.get(variety, 0.3)
    eff_beta = _BETA_BASE * sm * (1.0 - resist)

    # เริ่มจาก 7 วันก่อน: infected trees อยู่ที่ 60% ของ severity ปัจจุบัน
    init_sev = float(severity) * 0.6
    S = np.ones(N,  dtype=np.float32)
    E = np.zeros(N, dtype=np.float32)
    I = np.zeros(N, dtype=np.float32)
    R = np.zeros(N, dtype=np.float32)
    for idx in list(infected_idx):
        I[idx] = init_sev
        S[idx] = max(0.0, 1.0 - init_sev)

    # รัน SEIR ไปข้างหน้า WINDOW_SIZE ขั้น → ได้ประวัติ 7 วัน
    I_history = np.zeros((WINDOW_SIZE, N), dtype=np.float32)
    for t in range(WINDOW_SIZE):
        I_history[t] = I.copy()
        inf_force = adj @ I
        new_e = np.minimum(eff_beta * S * inf_force, S)
        new_i = _SIGMA * E
        new_r = _GAMMA * I
        S = np.clip(S - new_e,          0.0, 1.0)
        E = np.clip(E + new_e - new_i,  0.0, 1.0)
        I = np.clip(I + new_i - new_r,  0.0, 1.0)
        R = np.clip(R + new_r,          0.0, 1.0)

    delta = np.zeros_like(I_history)
    delta[1:] = I_history[1:] - I_history[:-1]

    age_norm    = np.clip(age_years / max_age_years, 0.0, 1.0)
    age_feat    = np.full((WINDOW_SIZE, N), age_norm,       dtype=np.float32)
    health_feat = np.full((WINDOW_SIZE, N), float(health),  dtype=np.float32)

    return np.stack([I_history, delta, age_feat, health_feat], axis=-1)  # (T,N,4)


# ──────────────────────────────────────────────────────────
# 3. SCENARIO MATCHING + MODEL INFERENCE
# ──────────────────────────────────────────────────────────
def match_weather_to_scenario(humidity, rainfall, temperature, wind):
    """Nearest predefined weather scenario (normalised L2 distance)."""
    x = np.array([
        np.clip(humidity, 0, 1),
        np.clip(rainfall / 3.0, 0, 1),
        np.clip((temperature - 20.0) / 15.0, 0, 1),
        np.clip(wind / 2.5, 0, 1),
    ], dtype=np.float32)

    best, best_d = None, float("inf")
    for name, p in SCENARIO_PROFILES.items():
        y = np.array([
            np.clip(p["humidity"], 0, 1),
            np.clip(p["rainfall"] / 3.0, 0, 1),
            np.clip((p["temperature"] - 20.0) / 15.0, 0, 1),
            np.clip(p["wind"] / 2.5, 0, 1),
        ], dtype=np.float32)
        d = float(np.linalg.norm(x - y))
        if d < best_d:
            best, best_d = name, d
    return best


def _load_model_and_norm(scenario_name: str, horizon_days: int, device):
    tag = f"{scenario_name.lower()}_{horizon_days}d"
    model_file = ROOT / f"model_scenario_{tag}.pt"
    norm_file = ROOT / f"norm_scenario_{tag}.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_file.name}")
    if not norm_file.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ normalizer: {norm_file.name}")

    model = SpatialTemporalGNN(
        node_features=4, hidden_dim=64, window_size=WINDOW_SIZE,
        gat_heads=4, dropout=0.15, lstm_layers=2,
    ).to(device)
    ckpt = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(norm_file, "rb") as f:
        d = pickle.load(f)
    return model, d["mean"], d["std"]


def predict(features: np.ndarray,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor,
            scenario_name: str,
            horizon_days: int):
    """
    Run a forecast.

    features : (T, N, 4) from synthesize_features
    Returns  : (N,) infection probability in [0, 1] at +horizon_days
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = _load_model_and_norm(scenario_name, horizon_days, device)

    mean = mean.to(device)
    std = std.to(device)

    X = torch.tensor(features, dtype=torch.float32, device=device)   # (T,N,4)
    flat = X.reshape(-1, X.shape[-1])
    X = ((flat - mean) / std).reshape(X.shape).unsqueeze(0)          # (1,T,N,4)

    with torch.no_grad():
        pred = model(X, edge_index.to(device), edge_weight.to(device))
    return pred.cpu().numpy().reshape(-1)


# ──────────────────────────────────────────────────────────
# 3b. SEIR FORWARD SIMULATION  (ใช้แทน GNN สำหรับ farmer app)
#     ตรงกับ methodology ในเปเปอร์ — โมเดลถูก train เพื่อ predict
#     output ของ SEIR นี้แหละ ดังนั้น run SEIR โดยตรงบน graph
#     ของเกษตรกรให้ผลที่ถูกต้องกว่าการใช้ GNN extrapolate ออกนอก
#     distribution ที่ train มา
# ──────────────────────────────────────────────────────────

# SEIR parameters (ตรงกับ train_scenario_models.py)
_BETA_BASE = 0.35    # baseline infection rate (Arauz 2000)
_SIGMA     = 0.25    # incubation rate   (latent period ~4 days)
_GAMMA     = 0.067   # recovery rate     (infectious period ~15 days)

# spread_multiplier ต่อ scenario  (ค่าจาก SCENARIOS dict ใน train_scenario_models.py)
# HOT_HUMID/DROUGHT/COLD_DRY/NORMAL ใช้ hardcoded override
# HEAVY_RAIN/WINDY ใช้ computed value จาก Table 2 ในเปเปอร์
_SPREAD_MULT = {
    "HEAVY_RAIN": 1.450,
    "HOT_HUMID":  2.2,
    "WINDY":      1.015,
    "NORMAL":     1.0,
    "DROUGHT":    0.6,
    "COLD_DRY":   0.4,
}


def seir_forecast(positions: np.ndarray,
                  edge_index: torch.Tensor,
                  edge_weight: torch.Tensor,
                  infected,
                  severity: float,
                  scenario: str,
                  horizon_days: int,
                  variety: str = "NamDokMai") -> np.ndarray:
    """
    Run Spatial SEIR simulation forward from the farmer's current observation.

    ใช้ graph ที่ build_graph() สร้างมา → ทำงานกับ orchard ขนาดใดก็ได้
    Returns (N,) infection level I(t+horizon_days) ∈ [0, 1]

    Interpretation:
      0.00–0.33 : ต่ำ — โรคยังไม่แพร่มาถึง
      0.33–0.66 : ปานกลาง — เริ่มได้รับเชื้อ เฝ้าระวัง
      0.66–1.00 : สูง — ติดเชื้อหนัก ต้องจัดการด่วน
    """
    N = len(positions)

    ei = edge_index.numpy()
    ew = edge_weight.numpy()

    # ตัด self-loops ออก (build_graph ใส่ไว้สำหรับ GNN เท่านั้น)
    mask = ei[0] != ei[1]
    rows, cols, data = ei[0][mask], ei[1][mask], ew[mask]

    # Adjacency matrix แบบ dense (N ≤ 3,600 ก็โอเค)
    adj = np.zeros((N, N), dtype=np.float32)
    adj[rows, cols] = data

    # beta_eff = beta_base × spread_multiplier × (1 − resistance)
    sm       = _SPREAD_MULT.get(scenario, 1.0)
    resist   = RESISTANCE_MAP.get(variety, 0.3)
    eff_beta = _BETA_BASE * sm * (1.0 - resist)

    # กำหนด state SEIR เริ่มต้น
    S = np.ones(N,  dtype=np.float32)
    E = np.zeros(N, dtype=np.float32)
    I = np.zeros(N, dtype=np.float32)
    R = np.zeros(N, dtype=np.float32)

    for idx in list(infected):
        sev       = float(np.clip(severity, 0.0, 1.0))
        I[idx]    = sev
        S[idx]    = max(0.0, 1.0 - sev)

    # Simulate forward horizon_days steps
    for _ in range(int(horizon_days)):
        inf_force = adj @ I                              # (N,)
        new_e     = np.minimum(eff_beta * S * inf_force, S)
        new_i     = _SIGMA * E
        new_r     = _GAMMA * I

        S = np.clip(S - new_e,          0.0, 1.0)
        E = np.clip(E + new_e - new_i,  0.0, 1.0)
        I = np.clip(I + new_i - new_r,  0.0, 1.0)
        R = np.clip(R + new_r,          0.0, 1.0)

    return I


# ──────────────────────────────────────────────────────────
# 4. LIVE WEATHER
#    หลัก : METAR จาก aviationweather.gov (NWS/NOAA)
#            → สถานีตรวจวัดจริง อุณหภูมิ / ความชื้น / ลม
#    ฝน   : Open-Meteo hourly sum 24 ชม.
#            (METAR format ไม่มีปริมาณฝนสะสม — ข้อจำกัดของ format)
#    fallback: Open-Meteo ทั้งหมด เมื่อไม่มีสถานี METAR ใกล้
# ──────────────────────────────────────────────────────────

# สถานี METAR ในไทย  {ICAO: (lat, lon, ชื่อ)}
# ที่มา: ICAO Doc 7910 / aviationweather.gov
THAI_METAR_STATIONS = {
    # ภาคเหนือ
    "VTCH": (18.767,  98.963, "เชียงใหม่"),
    "VTCI": (19.952,  99.883, "เชียงราย"),
    "VTBL": (18.271,  99.504, "ลำปาง"),
    "VTCN": (18.808, 100.783, "น่าน"),
    "VTPP": (18.132, 100.165, "แพร่"),
    "VTSM": (16.699,  98.545, "ตาก/แม่สอด"),
    "VTPE": (16.779, 100.279, "พิษณุโลก"),
    "VTPB": (16.676, 101.195, "เพชรบูรณ์"),
    # ภาคตะวันออกเฉียงเหนือ
    "VTSL": (17.439, 101.722, "เลย"),
    "VTUD": (17.386, 102.788, "อุดรธานี"),
    "VTUI": (17.384, 104.643, "นครพนม"),
    "VTUQ": (16.466, 102.784, "ขอนแก่น"),
    "VTUO": (14.948, 102.079, "นครราชสีมา"),
    "VTUU": (15.251, 104.870, "อุบลราชธานี"),
    # ภาคกลาง / ตะวันออก
    "VTBD": (13.906, 100.607, "กรุงเทพ/ดอนเมือง"),
    "VTBS": (13.681, 100.747, "กรุงเทพ/สุวรรณภูมิ"),
    "VTBK": (12.679, 101.005, "ระยอง/อู่ตะเภา"),
    "VTPH": (12.636,  99.952, "ประจวบคีรีขันธ์"),
    # ภาคใต้
    "VTSF": ( 9.132,  99.136, "สุราษฎร์ธานี"),
    "VTSB": ( 9.548, 100.061, "เกาะสมุย"),
    "VTPN": ( 8.539,  99.944, "นครศรีธรรมราช"),
    "VTSP": ( 8.113,  98.317, "ภูเก็ต"),
    "VTSA": ( 7.517,  99.617, "ตรัง"),
    "VTSS": ( 6.933, 100.392, "สงขลา/หาดใหญ่"),
    "VTSC": ( 6.519, 101.743, "นราธิวาส"),
}


def _nearest_metar(lat: float, lon: float):
    """คืน (icao, name, dist_deg) ของสถานี METAR ที่ใกล้ที่สุด."""
    best_icao, best_name, best_dist = None, None, float("inf")
    for icao, (slat, slon, name) in THAI_METAR_STATIONS.items():
        d = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5
        if d < best_dist:
            best_icao, best_name, best_dist = icao, name, d
    return best_icao, best_name, best_dist


def _magnus_rh(temp_c: float, dewp_c: float) -> float:
    """
    ความชื้นสัมพัทธ์จากอุณหภูมิและจุดน้ำค้าง (Magnus formula)
    อ้างอิง: Alduchov & Eskridge (1996), J. Appl. Meteor. 35, 601-609
    """
    import math
    a, b = 17.625, 243.04
    return 100.0 * math.exp(
        (a * dewp_c) / (b + dewp_c) - (a * temp_c) / (b + temp_c)
    )


def _fetch_metar(icao: str, timeout: float = 8.0):
    """
    ดึง METAR ล่าสุดจาก aviationweather.gov (NWS / NOAA)
    ที่มา API: https://aviationweather.gov/data/api/
    """
    import requests
    url = (
        "https://aviationweather.gov/api/data/metar"
        f"?ids={icao}&format=json&hours=2"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json() or []
    except Exception:
        return None
    # กรองเฉพาะสถานีที่ตรงกัน เอาอันล่าสุด
    matching = [d for d in data if d.get("icaoId") == icao]
    return matching[-1] if matching else None


def _fetch_precip_24h(lat: float, lon: float, timeout: float = 8.0) -> float:
    """
    ปริมาณฝนสะสม 24 ชม. จาก Open-Meteo hourly
    ที่มา: https://open-meteo.com/en/docs (past_hours parameter)
    """
    import requests
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation&past_hours=24&forecast_hours=0&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        vals = r.json().get("hourly", {}).get("precipitation", [])
        return float(sum(v for v in vals if v is not None))
    except Exception:
        return 0.0


def _open_meteo_full(lat: float, lon: float, timeout: float) -> dict | None:
    """Open-Meteo fallback สำหรับพื้นที่ที่ไม่มีสถานี METAR ใกล้."""
    import requests
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        "&hourly=precipitation&past_hours=24&forecast_hours=0&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data  = r.json()
        cur   = data["current"]
        vals  = data.get("hourly", {}).get("precipitation", [])
    except Exception:
        return None

    temp_c   = float(cur.get("temperature_2m",       28.0))
    rh       = float(cur.get("relative_humidity_2m", 60.0))
    wind_kmh = float(cur.get("wind_speed_10m",        5.0))
    precip   = float(sum(v for v in vals if v is not None))
    obs_time = str(cur.get("time", ""))

    return {
        "humidity":    float(np.clip(rh / 100.0,      0.0, 1.0)),
        "rainfall":    float(np.clip(precip / 15.0,   0.0, 3.0)),
        "temperature": temp_c,
        "wind":        float(np.clip(wind_kmh / 15.0, 0.0, 2.5)),
        "raw": {
            "temp_c":      temp_c,
            "rh_pct":      rh,
            "precip_mm":   precip,
            "wind_kmh":    wind_kmh,
            "obs_time":    obs_time,
            "source":      "model",
            "source_label":"โมเดลพยากรณ์ (ECMWF)",
            "station_name": "",
            "dist_km":     None,
        },
    }


# ตำแหน่งจังหวัดที่ปลูกมะม่วงสำคัญของไทย (lat, lon)
THAI_LOCATIONS = {
    # ── ภาคเหนือ ──────────────────────────────────────────────────────────
    "เชียงใหม่":         (18.79,  98.98),
    "เชียงราย":          (19.91,  99.83),
    "ลำปาง":             (18.29,  99.49),
    "ลำพูน":             (18.57,  99.01),
    "แม่ฮ่องสอน":        (19.30,  97.97),
    "น่าน":              (18.78, 100.78),
    "พะเยา":             (19.16,  99.89),
    "แพร่":              (18.14, 100.14),
    "อุตรดิตถ์":         (17.62, 100.10),
    "ตาก":               (16.88,  99.13),
    "สุโขทัย":           (17.01,  99.82),
    "พิษณุโลก":          (16.82, 100.27),
    "พิจิตร":            (16.44, 100.35),
    "กำแพงเพชร":         (16.47,  99.52),
    "เพชรบูรณ์":         (16.42, 101.16),
    # ── ภาคกลาง ───────────────────────────────────────────────────────────
    "นครสวรรค์":         (15.70, 100.12),
    "อุทัยธานี":         (15.38, 100.02),
    "ชัยนาท":            (15.19, 100.13),
    "ลพบุรี":            (14.80, 100.65),
    "สิงห์บุรี":         (14.89, 100.40),
    "อ่างทอง":           (14.59, 100.46),
    "สระบุรี":           (14.53, 100.91),
    "พระนครศรีอยุธยา":   (14.36, 100.57),
    "สุพรรณบุรี":        (14.47, 100.12),
    "นครปฐม":            (13.82, 100.05),
    "กาญจนบุรี":         (14.00,  99.55),
    "ราชบุรี":           (13.54,  99.81),
    "สมุทรสาคร":         (13.55, 100.27),
    "สมุทรสงคราม":       (13.41, 100.00),
    "เพชรบุรี":          (13.11,  99.94),
    "ประจวบคีรีขันธ์":   (11.81,  99.80),
    "กรุงเทพมหานคร":     (13.76, 100.50),
    "นนทบุรี":           (13.86, 100.52),
    "ปทุมธานี":          (14.02, 100.53),
    "สมุทรปราการ":        (13.60, 100.60),
    "นครนายก":           (14.21, 101.21),
    "ปราจีนบุรี":        (14.05, 101.37),
    "สระแก้ว":           (13.82, 102.06),
    # ── ภาคตะวันออก ────────────────────────────────────────────────────────
    "ฉะเชิงเทรา":        (13.69, 101.07),
    "ชลบุรี":            (13.36, 100.98),
    "ระยอง":             (12.68, 101.28),
    "จันทบุรี":          (12.61, 102.10),
    "ตราด":              (12.24, 102.52),
    # ── ภาคตะวันออกเฉียงเหนือ ─────────────────────────────────────────────
    "นครราชสีมา":        (14.97, 102.10),
    "ชัยภูมิ":           (15.81, 102.03),
    "บุรีรัมย์":         (14.99, 103.11),
    "สุรินทร์":          (14.88, 103.49),
    "ศรีสะเกษ":          (15.12, 104.32),
    "อุบลราชธานี":       (15.24, 104.85),
    "ยโสธร":             (15.79, 104.14),
    "อำนาจเจริญ":        (15.87, 104.63),
    "มุกดาหาร":          (16.54, 104.72),
    "ร้อยเอ็ด":          (16.05, 103.65),
    "มหาสารคาม":         (16.18, 103.30),
    "กาฬสินธุ์":         (16.43, 103.51),
    "ขอนแก่น":           (16.44, 102.83),
    "อุดรธานี":          (17.41, 102.79),
    "หนองบัวลำภู":       (17.20, 102.44),
    "หนองคาย":           (17.88, 102.74),
    "บึงกาฬ":            (18.36, 103.65),
    "สกลนคร":            (17.16, 104.14),
    "นครพนม":            (17.39, 104.77),
    "เลย":               (17.49, 101.73),
    # ── ภาคใต้ ────────────────────────────────────────────────────────────
    "ชุมพร":             (10.50,  99.18),
    "ระนอง":             ( 9.96,  98.61),
    "สุราษฎร์ธานี":      ( 9.13,  99.33),
    "นครศรีธรรมราช":     ( 8.43, 100.00),
    "กระบี่":            ( 8.09,  98.92),
    "พังงา":             ( 8.46,  98.53),
    "ภูเก็ต":            ( 7.88,  98.40),
    "ตรัง":              ( 7.56,  99.61),
    "พัทลุง":            ( 7.62, 100.08),
    "สงขลา":             ( 7.20, 100.60),
    "สตูล":              ( 6.62, 100.07),
    "ปัตตานี":           ( 6.87, 101.25),
    "ยะลา":              ( 6.54, 101.28),
    "นราธิวาส":          ( 6.43, 101.82),
}


def geocode_thailand(query: str, count: int = 6, timeout: float = 8.0):
    """
    ค้นหาสถานที่ในประเทศไทย (อำเภอ/ตำบล/จังหวัด/ชื่อเมือง) คืนพิกัด lat/lon
    ใช้ Nominatim (OpenStreetMap) — รองรับภาษาไทยเต็มรูปแบบ ฟรี ไม่ต้องใช้ key

    Returns list ของ dict: {label, name, admin1, lat, lon}
    คืน [] ถ้าค้นไม่พบหรือไม่มีอินเทอร์เน็ต
    """
    import requests

    query = (query or "").strip()
    if not query:
        return []

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "countrycodes": "TH",
        "format": "jsonv2",
        "limit": count,
        "accept-language": "th,en",
        "addressdetails": 1,
    }
    headers = {"User-Agent": "FarmerDiseaseApp/1.0 (orchard-disease-forecast)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        results = r.json() or []
    except Exception:
        return []

    out = []
    for res in results:
        display = res.get("display_name", "")
        addr = res.get("address", {})
        # ตำบล: Nominatim ใช้ key "suburb" หรือ "village" ขึ้นกับพื้นที่
        tambon = addr.get("suburb") or addr.get("village") or ""
        # อำเภอ: "county"
        amphoe = addr.get("county", "")
        # จังหวัด: "state"
        changwat = addr.get("state", "")
        # ชื่อสั้นที่ตรงที่สุด (ถ้าค้นระดับตำบลจะได้ตำบล, ระดับเมืองจะได้เมือง)
        name = (
            addr.get("city")
            or addr.get("town")
            or tambon
            or amphoe
            or changwat
            or display.split(",")[0].strip()
        )
        # สร้าง label: ตำบล · อำเภอ · จังหวัด (ข้ามซ้ำ)
        parts = [p for p in (tambon, amphoe, changwat) if p and p != name]
        label = name + (f" · {', '.join(parts)}" if parts else "")
        out.append({
            "label": label,
            "name": name,
            "admin1": changwat,
            "lat": float(res["lat"]),
            "lon": float(res["lon"]),
        })
    return out


def fetch_live_weather(lat: float, lon: float, timeout: float = 10.0):
    """
    ดึงสภาพอากาศโดยใช้ข้อมูลสถานีตรวจวัดจริงเป็นหลัก

    อุณหภูมิ / ความชื้น / ลม:
      → METAR จาก aviationweather.gov (NWS/NOAA)
        ใช้สถานีไทยที่ใกล้ที่สุด (ระยะ ≤ 250 กม.)
        RH คำนวณจาก Magnus formula (Alduchov & Eskridge 1996)
      → fallback: Open-Meteo (โมเดล) ถ้าไม่มีสถานีใกล้หรือดึงไม่ได้

    ปริมาณฝน 24 ชม.:
      → Open-Meteo hourly sum (METAR format ไม่มีค่าสะสม mm)
      เกณฑ์ TMD: ปกติ ~15 mm/วัน, ฝนหนัก 35–90 mm/วัน
    """
    MAX_DIST_DEG = 2.25   # ~250 กม.

    # 1. หาสถานี METAR ที่ใกล้ที่สุด
    icao, station_name, dist_deg = _nearest_metar(lat, lon)
    dist_km = dist_deg * 111.0

    metar_obs = None
    if dist_deg <= MAX_DIST_DEG:
        metar_obs = _fetch_metar(icao, timeout)

    # 2. ปริมาณฝน 24 ชม. (Open-Meteo) — ดึงพร้อมกัน
    precip_24h = _fetch_precip_24h(lat, lon, timeout)

    # 3. ถ้าไม่มี METAR → fallback ทั้งหมด
    if metar_obs is None:
        result = _open_meteo_full(lat, lon, timeout)
        if result and precip_24h > 0:
            # ใส่ฝนที่ดึงแยกมาแทน (อาจดีกว่า)
            result["rainfall"] = float(np.clip(precip_24h / 15.0, 0.0, 3.0))
            result["raw"]["precip_mm"] = precip_24h
        return result

    # 4. แปลง METAR → ค่าที่ใช้งานได้
    temp_c   = float(metar_obs.get("temp",  28.0))
    dewp_c   = float(metar_obs.get("dewp",  temp_c - 4.0))
    wspd_kt  = float(metar_obs.get("wspd",  3.0))
    wind_kmh = wspd_kt * 1.852          # 1 kt = 1.852 km/h (exact, ICAO Annex 5)
    obs_time = str(metar_obs.get("reportTime", ""))

    rh = float(np.clip(_magnus_rh(temp_c, dewp_c), 0.0, 100.0))

    humidity = float(np.clip(rh / 100.0,      0.0, 1.0))
    rainfall = float(np.clip(precip_24h / 15.0, 0.0, 3.0))
    wind     = float(np.clip(wind_kmh / 15.0,  0.0, 2.5))

    # แปลงเวลา UTC → ไทย (+7)
    thai_time = ""
    if len(obs_time) >= 16:
        try:
            from datetime import datetime, timezone, timedelta
            dt_utc = datetime.fromisoformat(obs_time.replace("Z", "+00:00")
                                            .replace(".000+00:00", "+00:00"))
            dt_th  = dt_utc.astimezone(timezone(timedelta(hours=7)))
            thai_time = dt_th.strftime("%H:%M")
        except Exception:
            thai_time = obs_time[11:16]

    return {
        "humidity":    humidity,
        "rainfall":    rainfall,
        "temperature": temp_c,
        "wind":        wind,
        "raw": {
            "temp_c":       temp_c,
            "rh_pct":       rh,
            "precip_mm":    precip_24h,
            "wind_kmh":     wind_kmh,
            "obs_time":     thai_time,
            "source":       "metar",
            "source_label": f"METAR {icao} · {station_name}",
            "station_name": station_name,
            "station_code": icao,
            "dist_km":      round(dist_km, 0),
        },
    }


# ──────────────────────────────────────────────────────────
# 5. SAVE / LOAD ORCHARD CONFIG
# ──────────────────────────────────────────────────────────
CONFIG_VERSION = 1


def serialize_orchard(config: dict) -> str:
    """แปลงผังสวน + ต้นที่ติดเชื้อ เป็นข้อความ JSON สำหรับบันทึกไฟล์."""
    payload = {"version": CONFIG_VERSION, **config}
    # set -> sorted list เพื่อให้ JSON ได้
    if isinstance(payload.get("infected"), (set, frozenset)):
        payload["infected"] = sorted(payload["infected"])
    return json.dumps(payload, ensure_ascii=False, indent=2)


def deserialize_orchard(text: str) -> dict:
    """อ่านไฟล์ผังสวนกลับมาเป็น dict (infected เป็น set)."""
    data = json.loads(text)
    data["infected"] = set(data.get("infected", []))
    return data


def list_available_horizons(scenario_name: str):
    """Which forecast horizons have a trained model for this scenario."""
    out = []
    for h in (7, 14):
        if (ROOT / f"model_scenario_{scenario_name.lower()}_{h}d.pt").exists():
            out.append(h)
    return out
