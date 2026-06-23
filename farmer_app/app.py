"""
แอปพยากรณ์การแพร่ระบาดของโรคในสวนมะม่วง
รัน: streamlit run farmer_app/app.py
"""

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import orchard_core as core

st.set_page_config(
    page_title="พยากรณ์โรคในสวนมะม่วง",
    page_icon="🌿",
    layout="wide",
)

# ── palette ────────────────────────────────────────────────────────────────
G900 = "#1B3A2A"   # header / primary text on dark
G700 = "#2D5A40"   # primary brand
G500 = "#4A8C64"   # active / accent
G100 = "#EAF4EE"   # light tint background
RISK_HI  = "#C62828"
RISK_MID = "#E65100"
RISK_LOW = "#2E7D32"
INK  = "#1A1A1A"
SOFT = "#5A6672"
LINE = "#D4E2DA"
BG   = "#F5F8F6"
WH   = "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
  font-family: 'Noto Sans Thai', sans-serif !important;
}}

/* app background */
.stApp {{ background: {BG}; }}
.block-container {{
  padding: 0 !important;
  max-width: 1200px;
}}

/* ── top bar ── */
.topbar {{
  background: {G900};
  padding: 16px 32px 14px;
  margin: 0 0 24px;
}}
.topbar-title {{
  font-size: 1.25rem;
  font-weight: 700;
  color: {WH};
  letter-spacing: -0.2px;
}}
.topbar-sub {{
  font-size: .8rem;
  color: rgba(255,255,255,.6);
  margin-top: 2px;
}}

/* ── steps ── */
.steps {{
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 32px 16px;
  border-bottom: 1px solid {LINE};
  margin-bottom: 24px;
}}
.st-step {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: .82rem;
  font-weight: 600;
  color: #B0BEC5;
  flex: 1;
}}
.st-step.done   {{ color: {G500}; }}
.st-step.active {{ color: {G700}; }}
.st-num {{
  width: 26px; height: 26px;
  border-radius: 50%;
  border: 2px solid #CFD8DC;
  display: flex; align-items: center; justify-content: center;
  font-size: .75rem; font-weight: 700;
  background: {WH};
  flex-shrink: 0;
}}
.st-step.done   .st-num {{ background:{G500}; border-color:{G500}; color:{WH}; }}
.st-step.active .st-num {{ background:{G700}; border-color:{G700}; color:{WH}; }}
.st-line {{ width: 28px; height: 1px; background: #CFD8DC; flex-shrink:0; }}
.st-line.done {{ background: {G500}; }}

/* ── section heading ── */
.sh {{
  font-size: .9rem;
  font-weight: 700;
  color: {G700};
  text-transform: uppercase;
  letter-spacing: .6px;
  padding-bottom: 8px;
  border-bottom: 2px solid {G700};
  margin-bottom: 14px;
}}

/* ── panel ── */
.panel {{
  background: {WH};
  border: 1px solid {LINE};
  border-radius: 8px;
  padding: 18px 20px;
  margin-bottom: 16px;
}}

/* ── weather grid ── */
.wx-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}}
.wx-item {{
  background: {BG};
  border: 1px solid {LINE};
  border-radius: 6px;
  padding: 9px 12px;
}}
.wx-label {{ font-size: .72rem; color: {SOFT}; font-weight: 600; text-transform: uppercase; letter-spacing:.4px; }}
.wx-value {{ font-size: 1.1rem; font-weight: 700; color: {INK}; margin-top: 2px; }}
.wx-source {{ font-size: .75rem; color: {SOFT}; margin-bottom: 2px; }}

/* ── map legend ── */
.legend {{
  display: flex;
  gap: 14px;
  font-size: .78rem;
  font-weight: 600;
  color: {SOFT};
  padding: 7px 0;
  margin-bottom: 4px;
}}
.ldot {{
  width: 10px; height: 10px;
  border-radius: 50%;
  display: inline-block;
  vertical-align: middle;
  margin-right: 4px;
}}

/* ── stat numbers ── */
.stat-row {{ display: flex; gap: 10px; margin-bottom: 14px; }}
.stat-item {{
  flex: 1;
  background: {WH};
  border: 1px solid {LINE};
  border-radius: 8px;
  padding: 12px 10px;
  text-align: center;
}}
.stat-item .val {{
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}}
.stat-item .lbl {{
  font-size: .72rem;
  color: {SOFT};
  font-weight: 600;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: .3px;
}}
.risk-pill {{
  display: inline-block;
  padding: 3px 12px;
  border-radius: 4px;
  font-size: .82rem;
  font-weight: 700;
  color: {WH};
}}

/* ── scenario tag ── */
.sc-tag {{
  display: inline-block;
  padding: 2px 9px;
  border-radius: 3px;
  font-size: .78rem;
  font-weight: 700;
  background: {G100};
  color: {G700};
  border: 1px solid #B2DFCF;
  margin-left: 6px;
}}

/* ── recommendation ── */
.rec {{
  border-left: 3px solid {RISK_HI};
  padding: 12px 14px;
  background: {WH};
  border-radius: 0 6px 6px 0;
  margin-top: 14px;
}}
.rec.ok {{ border-left-color: {RISK_LOW}; }}
.rec-title {{
  font-size: .88rem;
  font-weight: 700;
  color: {INK};
  margin-bottom: 8px;
}}
.rec-list {{
  list-style: none;
  margin: 0; padding: 0;
}}
.rec-list li {{
  font-size: .82rem;
  color: {SOFT};
  padding: 3px 0;
  display: flex;
  gap: 8px;
}}
.rec-list li::before {{ content: "–"; color: {LINE}; }}
.rec-note {{
  font-size: .76rem;
  color: {SOFT};
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid {LINE};
}}

/* ── streamlit overrides ── */
.stButton > button {{
  font-family: 'Noto Sans Thai', sans-serif !important;
  font-size: .88rem !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  padding: .42rem .85rem !important;
  border: 1px solid {LINE} !important;
  background: {WH} !important;
  color: {INK} !important;
  box-shadow: none !important;
  transition: border-color .12s, background .12s !important;
}}
.stButton > button:hover {{
  border-color: {G500} !important;
  background: {G100} !important;
  color: {G700} !important;
}}
.stButton > button[kind="primary"] {{
  background: {G700} !important;
  color: {WH} !important;
  border-color: {G700} !important;
  font-weight: 700 !important;
  font-size: .92rem !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: {G900} !important;
  border-color: {G900} !important;
}}
.stDownloadButton > button {{
  font-family: 'Noto Sans Thai', sans-serif !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  border: 1px solid {LINE} !important;
  background: {WH} !important;
  color: {G700} !important;
}}

label, .stSlider label, .stRadio label, .stSelectbox label, .stNumberInput label {{
  font-family: 'Noto Sans Thai', sans-serif !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
  color: {INK} !important;
}}
.stSlider [data-baseweb="slider"] {{ margin-top: 2px; }}

[data-testid="stExpander"] {{
  border: 1px solid {LINE} !important;
  border-radius: 8px !important;
  background: {WH} !important;
  margin-bottom: 10px !important;
}}
[data-testid="stExpander"] summary {{
  font-size: .88rem !important;
  font-weight: 700 !important;
  color: {INK} !important;
  padding: 10px 14px !important;
}}

section[data-testid="stSidebar"] {{
  background: {WH} !important;
  border-right: 1px solid {LINE} !important;
}}
section[data-testid="stSidebar"] * {{
  font-family: 'Noto Sans Thai', sans-serif !important;
}}

.stAlert {{ border-radius: 6px !important; font-size: .85rem !important; }}
hr {{ border-color: {LINE} !important; }}

/* ═══════════════════════════════════════════
   RESPONSIVE — Tablet & Mobile
   ═══════════════════════════════════════════ */

/* ── Tablet ≤1024px ── */
@media (max-width: 1024px) {{
  .block-container {{ max-width: 100% !important; padding: 0 !important; }}
  .topbar {{ padding: 12px 20px 10px; }}
  .steps  {{ padding: 0 20px 12px; }}
}}

/* ── Phone (<640px): stack all columns ── */
@media (max-width: 639px) {{
  [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap !important;
    gap: 0 !important;
  }}
  [data-testid="column"] {{
    flex: 1 1 100% !important;
    min-width: 100% !important;
    width: 100% !important;
  }}
  /* topbar */
  .topbar {{ padding: 10px 14px 8px; margin-bottom: 14px; }}
  .topbar-title {{ font-size: 1rem !important; }}
  .topbar-sub {{ font-size: .72rem !important; }}
  /* steps: hide text, show only numbers */
  .steps {{ padding: 0 14px 10px; gap: 0; overflow-x: auto; }}
  .st-step {{ font-size: 0 !important; gap: 4px !important; flex: unset; }}
  .st-step .st-num {{ font-size: .7rem !important; }}
  .st-line {{ width: 18px !important; }}
  /* bigger buttons for thumb */
  .stButton > button {{
    min-height: 48px !important;
    font-size: 1rem !important;
    padding: .6rem .9rem !important;
    width: 100%;
  }}
  /* weather grid full-width cells */
  .wx-grid {{ grid-template-columns: 1fr 1fr !important; gap: 6px !important; }}
  .wx-value {{ font-size: .95rem !important; }}
  /* stat row */
  .stat-item .val {{ font-size: 1.55rem !important; }}
  /* expander touch area */
  [data-testid="stExpander"] summary {{ padding: 13px 14px !important; min-height: 48px; }}
  /* map padding */
  [data-testid="stPlotlyChart"] {{ margin: 0 -4px; }}
}}

/* ── Tablet only (640–1023px): 2-col max ── */
@media (min-width: 640px) and (max-width: 1023px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
  [data-testid="column"] {{
    flex: 1 1 calc(50% - 10px) !important;
    min-width: calc(50% - 10px) !important;
  }}
  .topbar-title {{ font-size: 1.12rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────
def topbar():
    st.markdown(
        '<div class="topbar">'
        '<div class="topbar-title">ระบบพยากรณ์การแพร่ระบาดของโรคในสวนมะม่วง</div>'
        '<div class="topbar-sub">ระบุต้นที่เป็นโรค · ดึงสภาพอากาศ · ดูความเสี่ยงล่วงหน้า 7–14 วัน</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def steps(current: int):
    labels = ["ข้อมูลสวน", "ต้นที่เป็นโรค", "อากาศและพยากรณ์", "ผลลัพธ์"]
    parts = []
    for i, lbl in enumerate(labels, 1):
        cls  = "done" if i < current else ("active" if i == current else "")
        icon = "✓"   if i < current else str(i)
        parts.append(
            f'<div class="st-step {cls}">'
            f'<span class="st-num">{icon}</span>{lbl}</div>'
        )
        if i < len(labels):
            lc = "done" if i < current else ""
            parts.append(f'<div class="st-line {lc}"></div>')
    st.markdown(
        f'<div class="steps">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def sh(text: str):
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)


def legend():
    st.markdown(
        '<div class="legend">'
        f'<span><span class="ldot" style="background:{RISK_HI}"></span>เสี่ยงสูง ≥66%</span>'
        f'<span><span class="ldot" style="background:{RISK_MID}"></span>ปานกลาง 33–65%</span>'
        f'<span><span class="ldot" style="background:{RISK_LOW}"></span>เสี่ยงต่ำ &lt;33%</span>'
        f'<span><span class="ldot" style="background:#B2CCBA;border:1px solid #8CA89A"></span>ปกติ</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def risk_of(p: float):
    if p >= 0.66: return "สูง",      RISK_HI
    if p >= 0.33: return "ปานกลาง", RISK_MID
    return "ต่ำ", RISK_LOW


def stat_row(high, mid, low):
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-item"><div class="val" style="color:{RISK_HI}">{high}</div>'
        f'<div class="lbl">เสี่ยงสูง</div></div>'
        f'<div class="stat-item"><div class="val" style="color:{RISK_MID}">{mid}</div>'
        f'<div class="lbl">ปานกลาง</div></div>'
        f'<div class="stat-item"><div class="val" style="color:{RISK_LOW}">{low}</div>'
        f'<div class="lbl">เสี่ยงต่ำ</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def wx_card(d: dict):
    src      = d.get("source", "model")
    is_metar = src == "metar"
    badge_bg = G500   if is_metar else RISK_MID
    badge_tx = "สถานีตรวจวัด" if is_metar else "โมเดลพยากรณ์"
    src_lbl  = d.get("src_label", "")
    dist_km  = d.get("dist_km")
    dist_txt = f"  ห่าง {dist_km:.0f} กม." if dist_km else ""
    note_txt = (
        f"อุณหภูมิ / ความชื้น / ลม: METAR จริง{dist_txt}  ·  ฝน: โมเดล ECMWF (24 ชม.)"
        if is_metar else
        "ข้อมูลทั้งหมดจากโมเดลพยากรณ์ ECMWF — ไม่มีสถานีตรวจวัดใกล้"
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'<strong style="font-size:.85rem">{d["place"]}</strong>'
        f'{"<span style=\"font-size:.75rem;color:" + SOFT + "\">  อัปเดต " + d["time"] + "</span>" if d["time"] else ""}'
        f'<span style="margin-left:auto;background:{badge_bg};color:#fff;'
        f'font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:3px">'
        f'{badge_tx}</span>'
        f'</div>'
        f'<div class="wx-grid">'
        f'<div class="wx-item"><div class="wx-label">อุณหภูมิ</div><div class="wx-value">{d["temp"]}</div></div>'
        f'<div class="wx-item"><div class="wx-label">ความชื้นสัมพัทธ์</div><div class="wx-value">{d["rh"]}</div></div>'
        f'<div class="wx-item"><div class="wx-label">ฝนสะสม 24 ชม.</div><div class="wx-value">{d["rain"]}</div></div>'
        f'<div class="wx-item"><div class="wx-label">ความเร็วลม</div><div class="wx-value">{d["wind"]}</div></div>'
        f'</div>'
        f'<div style="font-size:.7rem;color:{SOFT};margin-top:5px">{note_txt}</div>',
        unsafe_allow_html=True,
    )


def _arrow():
    return dict(
        x=0.97, y=0.97, xref="x domain", yref="y domain",
        ax=0.84, ay=0.84, axref="x domain", ayref="y domain",
        showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2,
        arrowcolor=G700, text="ทิศลม", font=dict(size=11, color=G700),
    )


def orchard_map(positions, grid_index, values, title,
                discrete=False, key=None, selectable=False, arrow=False):
    if discrete:
        colors = [RISK_HI if v >= 0.5 else "#B2CCBA" for v in values]
        marker = dict(size=16, color=colors, line=dict(width=1.5, color=WH))
    else:
        cscale = [
            [0.00, RISK_LOW], [0.33, RISK_LOW],
            [0.33, RISK_MID], [0.66, RISK_MID],
            [0.66, RISK_HI],  [1.00, RISK_HI],
        ]
        marker = dict(
            size=16, color=values, colorscale=cscale, cmin=0, cmax=1,
            colorbar=dict(
                title=dict(text="ความเสี่ยง", font=dict(size=11, color=SOFT)),
                tickvals=[0.165, 0.495, 0.83],
                ticktext=["ต่ำ", "ปานกลาง", "สูง"],
                tickfont=dict(size=10, color=SOFT),
                len=0.55, thickness=12,
            ),
            line=dict(width=1.5, color=WH),
        )

    hover = [
        f"ต้น #{i}  แถว {r+1}  ต้นที่ {c+1}<br>ค่า {v:.2f}"
        for i, ((r, c), v) in enumerate(zip(grid_index, values))
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1],
        mode="markers", marker=marker, text=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=SOFT), x=0),
        xaxis_title="แนวนอน (ม.)",
        yaxis_title="แนวตั้ง (ม.)",
        yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor="#ECEFF1"),
        xaxis=dict(gridcolor="#ECEFF1"),
        font=dict(size=11, color=SOFT),
        height=430,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WH,
        dragmode="pan",
        clickmode="event+select" if selectable else "event",
    )
    if arrow:
        fig.add_annotation(**_arrow())
    if selectable:
        return st.plotly_chart(
            fig, width='stretch', key=key,
            on_select="rerun", selection_mode=("points", "box"),
        )
    st.plotly_chart(fig, width='stretch', key=key)


def gauge(value: float, key=None):
    pct = float(np.clip(value, 0, 1) * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=30, color=INK)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor="#CFD8DC", tickfont=dict(size=9, color=SOFT)),
            bar=dict(color=INK, thickness=0.12),
            borderwidth=0,
            steps=[
                dict(range=[0,  33], color=RISK_LOW),
                dict(range=[33, 66], color=RISK_MID),
                dict(range=[66,100], color=RISK_HI),
            ],
        ),
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=2, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color=SOFT),
    )
    st.plotly_chart(fig, width='stretch', key=key)


def animated_map(positions, start_vals, end_vals, horizon, key=None):
    s, e = np.asarray(start_vals, float), np.asarray(end_vals, float)
    n    = int(horizon)
    cscale = [
        [0.00, RISK_LOW], [0.33, RISK_LOW],
        [0.33, RISK_MID], [0.66, RISK_MID],
        [0.66, RISK_HI],  [1.00, RISK_HI],
    ]

    def mk(vals):
        return go.Scatter(
            x=positions[:, 0], y=positions[:, 1], mode="markers",
            marker=dict(
                size=16, color=vals, colorscale=cscale, cmin=0, cmax=1,
                colorbar=dict(title=dict(text="ความเสี่ยง", font=dict(size=10))),
                line=dict(width=1.5, color=WH),
            ),
            hoverinfo="skip",
        )

    def fv(t):
        return s + (e - s) * ((t / n) ** 0.85)

    fig = go.Figure(
        data=[mk(fv(0))],
        frames=[go.Frame(data=[mk(fv(t))], name=str(t)) for t in range(n + 1)],
    )
    btns = dict(
        type="buttons", showactive=False, x=0.0, y=1.1, xanchor="left",
        buttons=[
            dict(label="เล่น", method="animate",
                 args=[None, dict(frame=dict(duration=420, redraw=True),
                                  fromcurrent=True, transition=dict(duration=200))]),
            dict(label="หยุด", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ],
    )
    slider = dict(
        active=0, x=0.0, len=1.0, y=0, pad=dict(t=32),
        currentvalue=dict(prefix="วันที่ ", font=dict(size=11, color=SOFT)),
        steps=[
            dict(method="animate", label=str(t),
                 args=[[str(t)], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True))])
            for t in range(n + 1)
        ],
    )
    fig.update_layout(
        title=dict(text=f"การลุกลามของโรค  วันที่ 0 – {n}",
                   font=dict(size=13, color=SOFT), x=0),
        xaxis_title="แนวนอน (ม.)",
        yaxis_title="แนวตั้ง (ม.)",
        yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor="#ECEFF1"),
        xaxis=dict(gridcolor="#ECEFF1"),
        height=450,
        margin=dict(l=8, r=8, t=52, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WH,
        updatemenus=[btns], sliders=[slider],
        font=dict(size=11, color=SOFT),
    )
    fig.add_annotation(**_arrow())
    st.plotly_chart(fig, width='stretch', key=key)


def scroll_result():
    components.html(
        """<script>
        setTimeout(function(){
          var a = window.parent.document.getElementById('result-anchor');
          if (a){ a.scrollIntoView({behavior:'smooth', block:'start'}); return; }
          var m = window.parent.document.querySelector('section.main');
          if (m) m.scrollTo({top:m.scrollHeight, behavior:'smooth'});
        }, 350);
        </script>""",
        height=0,
    )


# ── session ────────────────────────────────────────────────────────────────
def boot():
    d = {
        "infected":           set(),
        "infected_history":   [],
        "map_gen":            0,
        "humidity":           0.60,
        "rainfall":           1.0,
        "temperature":        28.0,
        "wind":               1.0,
        "severity":           0.4,
        "weather_note":       "",
        "wx_data":            None,
        "wx_lat":             None,
        "wx_lon":             None,
        "geo_results":        [],
        "last_pred":          None,
        "_pending_wx_prov":   None,
        "layout_mode":        "grid",
        "csv_positions":      None,
        "csv_spacing":        6.0,
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


boot()


def _snap(v, step):
    """ปัดค่าให้ตรง step ของ slider."""
    return round(round(v / step) * step, 10)


def save_wx(w: dict, place: str, lat: float = None, lon: float = None):
    """บันทึกค่าอากาศที่ดึงได้ลง session_state — ปัดค่าให้ตรง slider step."""
    hum  = max(0.0,  min(1.0,  _snap(w["humidity"],    0.05)))
    rain = max(0.0,  min(3.0,  _snap(w["rainfall"],    0.10)))
    temp = max(10.0, min(40.0, _snap(w["temperature"], 0.50)))
    wnd  = max(0.0,  min(2.5,  _snap(w["wind"],        0.10)))
    st.session_state.update(
        humidity=hum,  rainfall=rain,
        temperature=temp, wind=wnd,
        sl_hum=hum,    sl_rain=rain,
        sl_temp=temp,  sl_wind=wnd,
        weather_note=place,
    )
    r = w["raw"]
    src       = r.get("source", "model")          # "metar" หรือ "model"
    src_label = r.get("source_label", "")
    dist_km   = r.get("dist_km")
    obs_time  = r.get("obs_time", "")
    st.session_state.wx_data = {
        "place":     place,
        "time":      obs_time + " น." if obs_time else "",
        "temp":      f"{r['temp_c']:.0f} °C",
        "rh":        f"{r['rh_pct']:.0f}%",
        "rain":      f"{r['precip_mm']:.1f} มม.",
        "wind":      f"{r['wind_kmh']:.0f} กม./ชม.",
        "source":    src,
        "src_label": src_label,
        "dist_km":   dist_km,
    }
    if lat is not None:
        st.session_state.wx_lat = lat
        st.session_state.wx_lon = lon


def _queue_prov_fetch():
    """on_change callback — queue ให้ auto-fetch อากาศเมื่อเลือกจังหวัดใหม่."""
    st.session_state._pending_wx_prov = st.session_state._prov_sel


_trees   = st.session_state.infected
_has_t   = len(_trees) > 0
_has_wx  = bool(st.session_state.weather_note)
_has_r   = st.session_state.last_pred is not None
_step    = 4 if _has_r else (3 if (_has_wx or _has_t) else 2)


# ── layout ─────────────────────────────────────────────────────────────────
topbar()
steps(_step)

# container ใหญ่ใส่ padding ซ้าย-ขวา
outer = st.container()

with outer:

    # ── step 1 accordion ──────────────────────────────────────────────────
    # defaults (ใช้ถ้า expander ยุบอยู่ หรือโหมด CSV)
    rows = 10; cols_n = 10; row_spacing = 6.0; tree_spacing = 6.0
    age_years = 10; health = 0.80

    with st.expander("ขั้นตอนที่ 1  —  ข้อมูลสวน", expanded=not _has_t):

        _lmode = st.radio(
            "รูปแบบผังสวน",
            ["ตาราง (ปลูกเป็นแถวเท่ากัน)", "กำหนดเอง (นำเข้าพิกัด CSV)"],
            horizontal=True,
            index=0 if st.session_state.layout_mode == "grid" else 1,
        )
        st.session_state.layout_mode = "grid" if _lmode.startswith("ตาราง") else "csv"

        if st.session_state.layout_mode == "grid":
            c1, c2, c3, c4 = st.columns(4)
            rows         = c1.number_input("จำนวนแถว",      1, 60, 10)
            cols_n       = c2.number_input("ต้นต่อแถว",     1, 60, 10)
            row_spacing  = c3.number_input("ระยะแถว (ม.)", 1.0, 30.0, 6.0, step=0.5)
            tree_spacing = c4.number_input("ระยะต้น (ม.)", 1.0, 30.0, 6.0, step=0.5)
        else:
            st.markdown(
                f'<div style="font-size:.82rem;color:{SOFT};background:{G100};'
                f'border:1px solid #B2DFCF;border-radius:6px;padding:10px 14px;margin-bottom:8px">'
                f'<strong>รูปแบบ CSV:</strong> สองคอลัมน์ <code>x_m</code> และ <code>y_m</code> '
                f'(หน่วยเมตร จากมุมอ้างอิงสวน)<br>'
                f'ตัวอย่าง: ต้นแรกที่มุมซ้ายล่าง = x=0, y=0  / ต้นถัดไปในแนวนอน = x=6, y=0<br>'
                f'ดาวน์โหลดตัวอย่าง CSV ได้ด้านล่าง</div>',
                unsafe_allow_html=True,
            )
            _sample_csv = "x_m,y_m\n0,0\n6,0\n12,0\n0,6\n6,6\n12,6\n"
            st.download_button(
                "ดาวน์โหลด CSV ตัวอย่าง",
                _sample_csv.encode(),
                file_name="orchard_positions_sample.csv",
                mime="text/csv",
            )
            _upload = st.file_uploader("อัปโหลดไฟล์พิกัดต้น (.csv)", type=["csv"])
            if _upload is not None:
                try:
                    import pandas as pd
                    _df = pd.read_csv(_upload)
                    # รองรับชื่อคอลัมน์หลายแบบ
                    _xcol = next((c for c in _df.columns
                                  if c.lower().replace("_","").replace(" ","") in ("xm","x")), None)
                    _ycol = next((c for c in _df.columns
                                  if c.lower().replace("_","").replace(" ","") in ("ym","y")), None)
                    if _xcol and _ycol and len(_df) >= 2:
                        st.session_state.csv_positions = list(
                            zip(_df[_xcol].astype(float), _df[_ycol].astype(float))
                        )
                        st.success(f"โหลดพิกัดสำเร็จ {len(st.session_state.csv_positions)} ต้น")
                    else:
                        st.error("ไม่พบคอลัมน์ x_m / y_m หรือข้อมูลน้อยกว่า 2 แถว")
                except Exception as e:
                    st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
            if st.session_state.csv_positions:
                _sp1, _sp2 = st.columns(2)
                st.session_state.csv_spacing = _sp1.number_input(
                    "ระยะห่างเฉลี่ยระหว่างต้น (ม.)", 1.0, 30.0,
                    float(st.session_state.csv_spacing), step=0.5,
                )

        st.divider()
        c5, c6, c7 = st.columns([2, 1, 1])
        variety = c5.selectbox(
            "พันธุ์มะม่วง", list(core.RESISTANCE_MAP.keys()),
            format_func=lambda v: {
                "NamDokMai": "น้ำดอกไม้  (อ่อนแอต่อโรค)",
                "Irwin":     "Irwin  (ทนปานกลาง)",
                "Keitt":     "Keitt  (ทนทาน)",
            }[v],
        )
        age_years = c6.slider("อายุเฉลี่ย (ปี)", 1, 25, age_years)
        health    = c7.slider("สุขภาพต้น", 0.0, 1.0, health, step=0.05,
                              help="0 = ทรุดโทรม  1 = สมบูรณ์ดี")

    # ── คำนวณ positions / grid_index ตามโหมด ────────────────────────────
    _lm = st.session_state.layout_mode
    if _lm == "grid":
        N        = rows * cols_n
        positions, grid_index = core.build_orchard_layout(rows, cols_n, row_spacing, tree_spacing)
        spacing  = min(row_spacing, tree_spacing)
        _is_grid = True
        def _tlabel(i):
            return f"ต้น #{i}  แถว {grid_index[i][0]+1}  ต้นที่ {grid_index[i][1]+1}"
    else:
        _csv_pos = st.session_state.csv_positions
        if not _csv_pos:
            st.info("กรุณานำเข้าไฟล์พิกัดต้นในขั้นตอนที่ 1 ก่อน")
            st.stop()
        positions  = np.array(_csv_pos, dtype=np.float32)
        N          = len(positions)
        grid_index = [(0, i) for i in range(N)]
        spacing    = float(st.session_state.csv_spacing)
        _is_grid   = False
        def _tlabel(i):
            return f"ต้น #{i}  ({positions[i,0]:.0f}ม, {positions[i,1]:.0f}ม)"
    st.session_state.infected = {i for i in st.session_state.infected if i < N}

    # ── two-column main ───────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="medium")

    # ════════════════════════════════
    # ซ้าย — step 2
    # ════════════════════════════════
    with left:
        _map_hdr = f"{rows}×{cols_n}" if _is_grid else "กำหนดเอง"
        sh(f"ขั้นตอนที่ 2  —  เลือกต้นที่เป็นโรค  ({N} ต้น, {_map_hdr})")
        st.caption("จิ้มต้นเพื่อเลือก/ยกเลิก  ·  ใช้ปุ่ม □ บน toolbar เพื่อลากครอบหลายต้น  ·  ลากพื้นที่ว่างเพื่อเลื่อนแผนที่")

        legend()

        vals = np.zeros(N)
        for i in st.session_state.infected:
            vals[i] = 1.0

        ev = orchard_map(
            positions, grid_index, vals, "ผังสวน",
            discrete=True,
            key=f"map_{st.session_state.map_gen}",
            selectable=True, arrow=True,
        )

        if ev and ev.get("selection"):
            pts = ev["selection"].get("points", [])
            picked = {
                int(p.get("point_index", p.get("point_number", -1)))
                for p in pts
                if p.get("point_index", p.get("point_number")) is not None
            }
            picked = {i for i in picked if 0 <= i < N}
            if picked:
                st.session_state.infected_history.append(
                    frozenset(st.session_state.infected)
                )
                if len(picked) == 1:
                    # คลิกเดี่ยว → toggle (ติดเชื้อ↔ปกติ)
                    i = next(iter(picked))
                    if i in st.session_state.infected:
                        st.session_state.infected.discard(i)
                    else:
                        st.session_state.infected.add(i)
                else:
                    # ลากครอบหลายต้น → เพิ่มทั้งหมด
                    st.session_state.infected |= picked
                st.session_state.map_gen += 1
                st.rerun()

        cnt = len(st.session_state.infected)
        st.markdown(
            f'<div style="font-size:.85rem;font-weight:600;color:{SOFT};'
            f'margin:2px 0 8px">เลือกแล้ว {cnt} ต้น</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("เลิกล่าสุด", width='stretch',
                     disabled=not st.session_state.infected_history):
            prev = st.session_state.infected_history.pop()
            st.session_state.infected = set(prev)
            st.session_state.map_gen += 1
            st.rerun()
        if _is_grid:
            if b2.button("แถวบนสุด", width='stretch'):
                st.session_state.infected_history.append(frozenset(st.session_state.infected))
                st.session_state.infected |= {i for i in range(N) if grid_index[i][0] == 0}
                st.session_state.map_gen += 1
                st.rerun()
        else:
            if b2.button("เลือกทั้งหมด", width='stretch'):
                st.session_state.infected_history.append(frozenset(st.session_state.infected))
                st.session_state.infected = set(range(N))
                st.session_state.map_gen += 1
                st.rerun()
        if b3.button("ล้างทั้งหมด", width='stretch'):
            st.session_state.infected_history.append(frozenset(st.session_state.infected))
            st.session_state.infected = set()
            st.session_state.map_gen += 1
            st.rerun()

        with st.expander("เพิ่ม / นำต้นออกด้วยตนเอง"):
            chosen = st.multiselect(
                "ต้นที่เป็นโรค", list(range(N)),
                default=sorted(st.session_state.infected),
                format_func=_tlabel,
            )
            if set(chosen) != st.session_state.infected:
                st.session_state.infected_history.append(
                    frozenset(st.session_state.infected)
                )
                st.session_state.infected = set(chosen)
                st.session_state.map_gen += 1
                st.rerun()

        st.divider()

        sh("ความรุนแรงของโรคที่สังเกตได้")

        # ระดับ 5 ขั้น พร้อมอาการที่เห็นได้จริง
        _SEV = {
            0.2: ("1 — เพิ่งพบ",
                  "ใบมีจุดสีน้ำตาลหรือดำกระจาย  ·  พื้นที่เสียหาย < 5% ของใบ  ·  ต้นยังดูแข็งแรงปกติ"),
            0.4: ("2 — เล็กน้อย",
                  "แผลบนใบชัดขึ้น 5–25% ของพื้นที่ใบ  ·  ยอดอ่อนบางส่วนเหี่ยวหรือแห้ง  ·  โรคอยู่แค่กิ่งล่าง"),
            0.6: ("3 — ปานกลาง",
                  "แผลลามเกิน 25% ของใบ  ·  ยอดอ่อนแห้งหลายส่วน  ·  ใบร่วงมากกว่าปกติ"),
            0.8: ("4 — รุนแรง",
                  "ใบร่วงจำนวนมาก  ·  กิ่งแห้งตาย 30% ขึ้นไป  ·  ผลเน่าหรือร่วงถ้ามี"),
            1.0: ("5 — วิกฤต",
                  "ต้นเสียหายมากกว่าครึ่ง  ·  ยอดหลักแห้งตาย  ·  อาจต้องตัดออกหรือโค่น"),
        }
        # snap ค่าเก่าที่อาจเป็น 0.1/0.3/0.5 ฯลฯ ให้ตรงกับ key ใหม่
        _cur_sev = min(_SEV.keys(), key=lambda k: abs(k - st.session_state.severity))

        severity = st.select_slider(
            "ระดับ",
            options=list(_SEV.keys()),
            value=_cur_sev,
            format_func=lambda v: _SEV[v][0],
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div style="background:{G100};border:1px solid #B2DFCF;border-radius:6px;'
            f'padding:9px 12px;font-size:.82rem;color:{INK};margin-top:4px">'
            f'<strong>อาการ:</strong> {_SEV[severity][1]}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.severity = severity

    # ════════════════════════════════
    # ขวา — step 3
    # ════════════════════════════════

    with right:
        sh("ขั้นตอนที่ 3  —  สภาพอากาศและพยากรณ์")

        # ── auto-fetch จาก queue ─────────────────────────────────────────
        _prov_q = st.session_state.get("_pending_wx_prov")
        if _prov_q:
            st.session_state._pending_wx_prov = None
            _prov_lat, _prov_lon = core.THAI_LOCATIONS[_prov_q]
            with st.spinner(f"ดึงข้อมูลอากาศ {_prov_q}..."):
                _w = core.fetch_live_weather(_prov_lat, _prov_lon)
            if _w is None:
                st.warning(f"ดึงอากาศ {_prov_q} ไม่สำเร็จ — ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
            else:
                save_wx(_w, _prov_q, lat=_prov_lat, lon=_prov_lon)
                st.rerun()

        # ── เลือกจังหวัด (auto-fill + ปุ่มรีเฟรช) ─────────────────────
        with st.expander("เลือกจังหวัด"):
            st.caption("เปลี่ยนจังหวัดเพื่อดึงอากาศอัตโนมัติ หรือกดรีเฟรชเพื่อดึงซ้ำ")
            _pa, _pb = st.columns([2.6, 1])
            _pa.selectbox(
                "จังหวัด", list(core.THAI_LOCATIONS.keys()),
                key="_prov_sel",
                label_visibility="collapsed",
                on_change=_queue_prov_fetch,
            )
            if _pb.button("รีเฟรช", width='stretch'):
                st.session_state._pending_wx_prov = st.session_state.get(
                    "_prov_sel", list(core.THAI_LOCATIONS.keys())[0]
                )

        # ── ค้นหาสถานที่อื่น (Nominatim) ────────────────────────────────
        with st.expander("ค้นหาสถานที่เพิ่มเติม"):
            g1, g2 = st.columns([2.4, 1])
            query = g1.text_input(
                "สถานที่",
                placeholder="เช่น ตำบลในเมือง อุบล, ปากช่อง",
                label_visibility="collapsed",
            )
            if g2.button("ค้นหา", width='stretch'):
                with st.spinner("ค้นหาสถานที่..."):
                    st.session_state.geo_results = core.geocode_thailand(query)
                if not st.session_state.geo_results:
                    st.warning("ไม่พบสถานที่ — ลองพิมพ์ชื่ออำเภอหรือจังหวัด")

            if st.session_state.geo_results:
                labels = [r["label"] for r in st.session_state.geo_results]
                pick = st.selectbox(
                    "สถานที่ที่พบ", range(len(labels)),
                    format_func=lambda i: labels[i],
                    label_visibility="collapsed",
                )
                if st.button("ดึงสภาพอากาศตำแหน่งนี้", type="secondary",
                             width='stretch'):
                    place = st.session_state.geo_results[pick]
                    with st.spinner("ดึงข้อมูลอากาศ..."):
                        w = core.fetch_live_weather(place["lat"], place["lon"])
                    if w is None:
                        st.warning("ดึงอากาศไม่สำเร็จ — ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
                    else:
                        save_wx(w, place["name"], lat=place["lat"], lon=place["lon"])
                        st.rerun()

        if st.session_state.wx_data:
            wx_card(st.session_state.wx_data)

        # ปรับเอง
        hum  = st.session_state.humidity
        rain = st.session_state.rainfall
        temp = st.session_state.temperature
        wnd  = st.session_state.wind

        with st.expander("ปรับค่าอากาศด้วยตนเอง"):
            hum  = st.slider("ความชื้น (0–1)",       0.0, 1.0,  st.session_state.humidity,    step=0.05, key="sl_hum")
            rain = st.slider("ปริมาณฝน (0–3)",       0.0, 3.0,  st.session_state.rainfall,    step=0.10, key="sl_rain")
            temp = st.slider("อุณหภูมิ (°C)",         10.0,40.0, st.session_state.temperature, step=0.50, key="sl_temp")
            wnd  = st.slider("ความเร็วลม (0–2.5)",   0.0, 2.5,  st.session_state.wind,        step=0.10, key="sl_wind")
            st.session_state.update(humidity=hum, rainfall=rain, temperature=temp, wind=wnd)

        st.divider()

        scenario = core.match_weather_to_scenario(hum, rain, temp, wnd)
        sc_label = core.SCENARIO_LABELS_TH.get(scenario, scenario)
        st.markdown(
            f'<div style="font-size:.82rem;font-weight:600;color:{SOFT};margin-bottom:12px">'
            f'สถานการณ์อากาศ'
            f'<span class="sc-tag">{sc_label}</span></div>',
            unsafe_allow_html=True,
        )

        horizon = st.radio(
            "พยากรณ์ล่วงหน้า", [7, 14],
            format_func=lambda d: f"{d} วัน",
            horizontal=True,
        )

        run = st.button(
            "พยากรณ์การแพร่ระบาด",
            type="primary",
            width='stretch',
        )


# ── ผลพยากรณ์ ──────────────────────────────────────────────────────────────
st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)

if run:
    if not st.session_state.infected:
        st.warning("กรุณาเลือกต้นที่เป็นโรคอย่างน้อย 1 ต้น")
    else:
        try:
            use_h = horizon

            # ดึงอากาศพยากรณ์ช่วง horizon วันแทนอากาศปัจจุบัน
            # เพื่อให้ scenario ที่เลือกตรงกับสภาพอากาศจริงตลอดช่วงที่พยากรณ์
            fcast_scenario = scenario
            fcast_label    = sc_label
            _wx_lat = st.session_state.get("wx_lat")
            _wx_lon = st.session_state.get("wx_lon")
            if _wx_lat is not None:
                _fw = core.fetch_forecast_weather(_wx_lat, _wx_lon, use_h)
                if _fw:
                    fcast_scenario = core.match_weather_to_scenario(
                        _fw["humidity"], _fw["rainfall"],
                        _fw["temperature"], _fw["wind"],
                    )
                    fcast_label = core.SCENARIO_LABELS_TH.get(fcast_scenario, fcast_scenario)

            ei, ew = core.build_graph(positions, spacing)
            with st.spinner("กำลังพยากรณ์..."):
                feats = core.synthesize_features(
                    positions    = positions,
                    edge_index   = ei,
                    edge_weight  = ew,
                    infected_idx = st.session_state.infected,
                    severity     = severity,
                    scenario     = fcast_scenario,
                    variety      = variety,
                    age_years    = age_years,
                    health       = health,
                )
                horizons = core.list_available_horizons(fcast_scenario)
                use_h = use_h if use_h in horizons else (horizons[0] if horizons else None)
                if use_h is None:
                    st.error(f"ไม่พบโมเดลสำหรับ scenario: {fcast_scenario}")
                    st.stop()
                pred = core.predict(feats, ei, ew, fcast_scenario, use_h)
                pred = np.clip(pred, 0.0, 1.0)

            st.session_state.last_pred = pred.tolist()
            scroll_result()

            high = int((pred >= 0.66).sum())
            mid  = int(((pred >= 0.33) & (pred < 0.66)).sum())
            low  = int((pred < 0.33).sum())
            avg  = float(pred.mean())
            lvl, rc = risk_of(avg)

            st.markdown("---")
            _sc_tag = (
                f'<span class="sc-tag">{fcast_label} · พยากรณ์ {use_h} วัน</span>'
                if fcast_scenario != scenario else
                f'<span class="sc-tag">{fcast_label}</span>'
            )
            st.markdown(
                f'<h3 style="margin:0 0 12px">ผลพยากรณ์ในอีก {use_h} วัน  {_sc_tag}</h3>',
                unsafe_allow_html=True,
            )

            mc1, mc2 = st.columns([1.6, 1.0], gap="large")

            with mc1:
                legend()
                orchard_map(
                    positions, grid_index, pred,
                    f"แผนที่ความเสี่ยง  +{use_h} วัน",
                    key="rmap", arrow=True,
                )

            with mc2:
                stat_row(high, mid, low)

                st.markdown(
                    f'<div style="font-size:.75rem;font-weight:600;'
                    f'color:{SOFT};text-align:center;margin-bottom:2px;'
                    f'text-transform:uppercase;letter-spacing:.4px">'
                    f'ความเสี่ยงเฉลี่ย</div>',
                    unsafe_allow_html=True,
                )
                gauge(avg, key="g1")
                st.markdown(
                    f'<div style="text-align:center;margin-top:4px">'
                    f'<span class="risk-pill" style="background:{rc}">{lvl}</span></div>',
                    unsafe_allow_html=True,
                )

                # Export
                st.markdown("<br>", unsafe_allow_html=True)
                buf = io.StringIO()
                wr  = csv.DictWriter(
                    buf,
                    fieldnames=["ต้น", "แถว", "ต้นที่", "ความเสี่ยง_%", "ระดับ"],
                )
                wr.writeheader()
                for i in range(N):
                    wr.writerow({
                        "ต้น": i,
                        "แถว": grid_index[i][0] + 1,
                        "ต้นที่": grid_index[i][1] + 1,
                        "ความเสี่ยง_%": f"{pred[i]*100:.1f}",
                        "ระดับ": (
                            "สูง" if pred[i] >= 0.66
                            else "ปานกลาง" if pred[i] >= 0.33
                            else "ต่ำ"
                        ),
                    })
                st.download_button(
                    "ดาวน์โหลดผล CSV",
                    buf.getvalue().encode("utf-8-sig"),
                    file_name=f"risk_{fcast_scenario.lower()}_{use_h}d.csv",
                    mime="text/csv",
                    width='stretch',
                )

            # Animation
            st.markdown(
                f'<div class="sh" style="margin-top:18px">'
                f'การลุกลามของโรคแบบจำลองทีละวัน</div>',
                unsafe_allow_html=True,
            )
            sv = np.zeros(N)
            for i in st.session_state.infected:
                sv[i] = severity
            animated_map(positions, sv, pred, use_h, key="anim")
            st.caption("กด เล่น หรือลากสไลเดอร์เพื่อดูแนวโน้มทีละวัน")

            # คำแนะนำ
            if high > 0:
                top_n = np.argsort(pred)[::-1][:min(5, high)]
                items = "".join(
                    f'<li>ต้น #{i}  แถว {grid_index[i][0]+1}  '
                    f'ต้นที่ {grid_index[i][1]+1}  —  {pred[i]*100:.0f}%</li>'
                    for i in top_n
                )
                st.markdown(
                    f'<div class="rec">'
                    f'<div class="rec-title">ต้นที่ควรจัดการก่อน '
                    f'({min(5,high)} ต้นเสี่ยงสูงสุด)</div>'
                    f'<ul class="rec-list">{items}</ul>'
                    f'<div class="rec-note">พิจารณาตัดแต่งกิ่งและพ่นสารป้องกันโรค '
                    f'เฝ้าระวังต้นข้างเคียงในทิศทางลมตะวันออกเฉียงเหนือ</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="rec ok">'
                    f'<div class="rec-title">ความเสี่ยงโดยรวมยังต่ำ</div>'
                    f'<ul class="rec-list">'
                    f'<li>เฝ้าระวังตามปกติและรักษาสุขอนามัยในสวน</li>'
                    f'<li>ตรวจซ้ำในอีก 7 วัน</li>'
                    f'</ul></div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.exception(e)


# ── footer ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "ผลพยากรณ์เป็นการประมาณการจาก AI ควรใช้ร่วมกับการสำรวจแปลงจริง  ·  "
    "อากาศ: Open-Meteo  ·  พิกัด: OpenStreetMap Nominatim"
)
