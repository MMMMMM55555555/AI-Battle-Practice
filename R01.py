import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
from openai import OpenAI
import json
import base64
import math
import copy
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 0. ตั้งค่าหน้าจอ & PREMIUM ENTERPRISE CSS
# ==========================================
st.set_page_config(
    page_title="HydroConsult AI | ระบบตรวจรับสเปคเครื่องสูบน้ำและอุโมงค์ส่งน้ำ",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* พื้นหลังและฟอนต์ระดับพรีเมียม */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 60%, #000000 100%);
        border: 1px solid rgba(0, 180, 216, 0.35);
        border-radius: 16px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 180, 216, 0.18);
        color: #90E0EF;
        border: 1px solid rgba(144, 224, 239, 0.4);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 2.05rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
    }
    .hero-subtitle {
        color: #CBD5E1;
        font-size: 1.0rem;
        margin: 0;
        font-weight: 400;
    }

    /* Section Header Bar */
    .section-header {
        background: linear-gradient(90deg, rgba(0, 180, 216, 0.14) 0%, rgba(15, 23, 42, 0.0) 100%);
        border-left: 5px solid #00B4D8;
        padding: 12px 20px;
        border-radius: 6px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    .section-header h3 {
        margin: 0;
        color: #F8FAFC;
        font-size: 1.32rem;
        font-weight: 700;
    }
    .section-header p {
        margin: 4px 0 0 0;
        color: #94A3B8;
        font-size: 0.88rem;
    }

    /* ตกแต่งการ์ด Metric (st.metric) ให้เป็น Glassmorphism */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.82), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 180, 216, 0.55);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }

    /* Workflow Stepper Pills */
    .step-pill {
        text-align: center;
        padding: 10px 8px;
        border-radius: 10px;
        font-size: 0.84rem;
        font-weight: 600;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.7);
        color: #94A3B8;
    }
    .step-pill.active {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.18), rgba(0, 180, 216, 0.18));
        border-color: #06D6A0;
        color: #FFFFFF;
        box-shadow: 0 0 15px rgba(6, 214, 160, 0.2);
    }
    .step-pill.locked {
        background: rgba(239, 71, 111, 0.12);
        border-color: rgba(239, 71, 111, 0.45);
        color: #FDA4AF;
    }

    /* Verdict Banner สำหรับ Phase 5 */
    .verdict-box-pass {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.20) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #06D6A0;
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(6, 214, 160, 0.15);
    }
    .verdict-box-warn {
        background: linear-gradient(135deg, rgba(255, 209, 102, 0.20) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #FFD166;
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(255, 209, 102, 0.15);
    }
    .verdict-box-fail {
        background: linear-gradient(135deg, rgba(239, 71, 111, 0.22) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #EF476F;
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(239, 71, 111, 0.18);
    }
    .verdict-badge-pass {
        background: #06D6A0;
        color: #052E16;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    .verdict-badge-warn {
        background: #FFD166;
        color: #422006;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    .verdict-badge-fail {
        background: #EF476F;
        color: #FFFFFF;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# แสดงแบนเนอร์หัวเว็บระดับองค์กร
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🏛️ MECHANICAL & HYDRAULIC QA/QC DECISION SUPPORT SYSTEM</div>
    <h1 class="hero-title">ระบบวิศวกรรมอัจฉริยะตรวจสอบความเหมาะสมเครื่องสูบน้ำและระบบอุโมงค์ส่งน้ำ</h1>
    <p class="hero-subtitle">
        เครื่องมือช่วยวิศวกรผู้ควบคุมงานตรวจรับเอกสารขออนุมัติวัสดุ (Contractor Submittal Validation): 
        สกัดพารามิเตอร์อัตโนมัติ → คำนวณ System Curve → วิเคราะห์จุดตัด Pump Curve → ประเมินความเสี่ยงทางกลและสรุปผลแบบ Integrated
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def render_section_header(step_num, title_th, subtitle_th):
    """สร้างแถบหัวข้อแต่ละส่วนให้ดูเป็นระบบ Enterprise"""
    st.markdown(f"""
    <div class="section-header">
        <h3>ส่วนที่ {step_num}: {title_th}</h3>
        <p>{subtitle_th}</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 1. ฟังก์ชันช่วยเหลือ & แปลชื่อตัวแปรเป็นภาษาไทย
# ==========================================
FIELD_THAI_MAP = {
    "flow_rate_q_m3s": "อัตราการไหลออกแบบรวมของโครงการ (Design Flow Rate, Q)",
    "static_head_m": "ระยะยกน้ำสถิต (Static Head, Hst) หรือผลต่างระดับน้ำต้นทาง–ปลายทาง",
    "main_pipe_length_m": "ความยาวรวมของท่อส่งน้ำหรืออุโมงค์ (Conduit / Tunnel Length, L)",
    "main_pipe_diameter_m": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์ (Internal Diameter, D)",
    "pipe_material_or_roughness": "ชนิดวัสดุท่อ หรือสัมประสิทธิ์ความขรุขระ (Manning's n / Friction Factor f)",
    "proposed_pump_head_m": "เฮดของเครื่องสูบน้ำที่เสนอ (Proposed Pump Rated Head หรือพิกัดกราฟปั๊ม)",
    "proposed_pump_eff_pct": "ประสิทธิภาพเครื่องสูบน้ำ ณ จุดใช้งาน (Pump Efficiency, ηp %)",
    "proposed_gear_eff_pct": "ประสิทธิภาพชุดเกียร์ทดรอบ (Gearbox Transmission Efficiency, ηg %)",
    "proposed_motor_kw": "พิกัดกำลังมอเตอร์ไฟฟ้าที่ผู้รับจ้างเสนอ (Proposed Motor Nameplate Rating, kW)",
    "proposed_npshr_m": "ค่า NPSHr ของเครื่องสูบน้ำ (Net Positive Suction Head Required, m)"
}


def format_metric(val, unit="", decimals=2):
    """แสดงหน่วยวัดเฉพาะเมื่อมีตัวเลขจริงเท่านั้น หากไม่มีข้อมูลจะแสดง 'ไม่พบข้อมูล' ล้วนๆ"""
    if val is None or val == "" or str(val).strip().lower() in ["none", "null", "n/a", "ไม่พบข้อมูล"]:
        return "ไม่พบข้อมูล"
    try:
        num = float(val)
        if 0 < abs(num) < 0.1 and decimals == 2:
            formatted = f"{num:.4f}"
        elif num.is_integer():
            formatted = f"{int(num):,}"
        else:
            formatted = f"{num:,.{decimals}f}"
        return f"{formatted} {unit}".strip()
    except (ValueError, TypeError):
        return f"{val} {unit}".strip() if unit else str(val)


def post_process_extracted_data(extracted):
    """เติมเต็มข้อมูลที่เชื่อมโยงกันอัตโนมัติ เช่น ดึง Duty Head และ Efficiency จากตารางพิกัดกราฟปั๊ม"""
    q_total = float(extracted.get("flow_rate_q_m3s") or 0.0)
    num_pumps = int(extracted.get("num_parallel_pumps") or 1)
    q_unit_target = q_total / max(num_pumps, 1) if q_total > 0 else 0.0

    segments = extracted.get("pipe_segments") or []
    if segments:
        if not extracted.get("main_pipe_diameter_m"):
            for s in segments:
                if s.get("diameter_m"):
                    extracted["main_pipe_diameter_m"] = s["diameter_m"]
                    break
        if not extracted.get("main_pipe_length_m"):
            total_l = sum(float(s.get("length_m") or 0.0) for s in segments)
            if total_l > 0:
                extracted["main_pipe_length_m"] = total_l
        if not extracted.get("manning_n"):
            for s in segments:
                if s.get("manning_n"):
                    extracted["manning_n"] = s["manning_n"]
                    break

    if extracted.get("total_minor_k") and not extracted.get("minor_loss_fittings"):
        extracted["minor_loss_fittings"] = [
            f"อุปกรณ์ประกอบในระบบมีค่าสัมประสิทธิ์การสูญเสียรองรวม (ΣK) = {extracted['total_minor_k']}"
        ]

    pts = extracted.get("pump_curve_points") or []
    if pts:
        positive_pts = [p for p in pts if float(p.get("q_single_m3s") or 0.0) > 0]
        chosen_pt = None
        if positive_pts and q_unit_target > 0:
            chosen_pt = min(positive_pts, key=lambda p: abs(float(p.get("q_single_m3s", 0.0)) - q_unit_target))
        elif positive_pts:
            chosen_pt = positive_pts[-1]

        if chosen_pt:
            if extracted.get("proposed_pump_head_m") is None and chosen_pt.get("head_m") is not None:
                extracted["proposed_pump_head_m"] = float(chosen_pt["head_m"])
            if extracted.get("proposed_pump_eff_pct") is None and chosen_pt.get("eff_pct") is not None:
                if float(chosen_pt["eff_pct"]) > 0:
                    extracted["proposed_pump_eff_pct"] = float(chosen_pt["eff_pct"])

    return extracted


def check_step2_completeness(data):
    """ตรวจสอบข้อมูลสำหรับส่วนที่ 2 (ระบบท่อและ System Curve)"""
    missing = []
    q_val = data.get("flow_rate_q_m3s")
    h_st = data.get("static_head_m")
    main_l = data.get("main_pipe_length_m")
    main_d = data.get("main_pipe_diameter_m")
    roughness_str = data.get("pipe_material_or_roughness")
    manning_n = data.get("manning_n")
    darcy_f = data.get("darcy_f")
    segments = data.get("pipe_segments") or []

    has_valid_segment = False
    has_seg_roughness = False
    for seg in segments:
        seg_d = seg.get("diameter_m") or seg.get("area_m2")
        seg_l = float(seg.get("length_m") or 0.0)
        seg_k = float(seg.get("minor_loss_k_sum") or 0.0)
        seg_fixed = float(seg.get("fixed_minor_loss_m") or 0.0)
        if seg_d and (seg_l > 0 or seg_k > 0 or seg_fixed > 0):
            has_valid_segment = True
        if seg.get("manning_n") or seg.get("darcy_f"):
            has_seg_roughness = True

    has_valid_main = bool(main_d and float(main_d) > 0 and main_l and float(main_l) > 0)
    has_roughness = bool(roughness_str or manning_n or darcy_f or has_seg_roughness)

    if not q_val or float(q_val) <= 0:
        missing.append(FIELD_THAI_MAP["flow_rate_q_m3s"])
    if h_st is None:
        missing.append(FIELD_THAI_MAP["static_head_m"])
    if not (has_valid_segment or has_valid_main):
        if not main_d or float(main_d or 0) <= 0:
            missing.append(FIELD_THAI_MAP["main_pipe_diameter_m"])
        if not main_l or float(main_l or 0) <= 0:
            missing.append(FIELD_THAI_MAP["main_pipe_length_m"])
    if not has_roughness:
        missing.append(FIELD_THAI_MAP["pipe_material_or_roughness"])

    return (len(missing) == 0), missing


def check_step3_completeness(data):
    """ตรวจสอบข้อมูลสำหรับส่วนที่ 3 (ต้องมีค่า Head ปั๊ม หรือตารางพิกัดกราฟปั๊ม)"""
    pts = data.get("pump_curve_points") or []
    has_head = (data.get("proposed_pump_head_m") is not None and float(data.get("proposed_pump_head_m") or 0) > 0)
    has_pts = len(pts) > 0
    if not (has_head or has_pts):
        return False, [FIELD_THAI_MAP["proposed_pump_head_m"]]
    return True, []


def check_step4_completeness(data):
    """ตรวจสอบข้อมูลสำหรับส่วนที่ 4 (ต้องมีขนาดมอเตอร์ kW, ประสิทธิภาพปั๊ม, เกียร์ และ NPSHr)"""
    missing = []
    has_motor = (data.get("proposed_motor_kw") is not None and float(data.get("proposed_motor_kw") or 0) > 0)
    has_eff = (data.get("proposed_pump_eff_pct") is not None and float(data.get("proposed_pump_eff_pct") or 0) > 0)
    has_gear = (data.get("proposed_gear_eff_pct") is not None and float(data.get("proposed_gear_eff_pct") or 0) > 0)
    has_npshr = (data.get("proposed_npshr_m") is not None and float(data.get("proposed_npshr_m") or 0) > 0)

    if not has_motor:
        missing.append(FIELD_THAI_MAP["proposed_motor_kw"])
    if not has_eff:
        missing.append(FIELD_THAI_MAP["proposed_pump_eff_pct"])
    if not has_gear:
        missing.append(FIELD_THAI_MAP["proposed_gear_eff_pct"])
    if not has_npshr:
        missing.append(FIELD_THAI_MAP["proposed_npshr_m"])

    return (len(missing) == 0), missing


def extract_text_from_file(uploaded_file):
    """ดึงข้อความจากไฟล์ PDF, CSV, Excel"""
    text = ""
    fname = uploaded_file.name.lower()
    if fname.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
    elif fname.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        text = df.to_string()
    elif fname.endswith(('.xls', '.xlsx')):
        excel_file = pd.ExcelFile(uploaded_file)
        for sheet in excel_file.sheet_names:
            df = excel_file.parse(sheet)
            text += f"\n--- Sheet: {sheet} ---\n" + df.to_string()
    else:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return text


def encode_image(uploaded_file):
    """แปลงไฟล์ภาพเป็น Base64"""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


# ==========================================
# 2. ฟังก์ชันคำนวณชลศาสตร์ (PHASE 2: HYDRAULIC ENGINE)
# ==========================================
def run_hydraulic_engine(params, formula_override=None):
    g = 9.81
    q_total = float(params.get("flow_rate_q_m3s") or 0.0)
    h_static = float(params.get("static_head_m") or 0.0)
    num_pumps = int(params.get("num_parallel_pumps") or 1)

    segments = params.get("pipe_segments") or []
    valid_segments = [
        s for s in segments
        if (s.get("diameter_m") or s.get("area_m2")) and (
            float(s.get("length_m") or 0) > 0 or
            float(s.get("minor_loss_k_sum") or 0) > 0 or
            float(s.get("fixed_minor_loss_m") or 0) > 0
        )
    ]

    if not valid_segments:
        valid_segments = [{
            "segment_name": "ท่อส่งน้ำหลัก / อุโมงค์ระบายน้ำ (Primary Conduit)",
            "carries_single_pump_flow": False,
            "diameter_m": float(params.get("main_pipe_diameter_m") or 1.0),
            "area_m2": None,
            "length_m": float(params.get("main_pipe_length_m") or 100.0),
            "manning_n": float(params.get("manning_n") or 0.014),
            "darcy_f": float(params.get("darcy_f") or 0.015),
            "minor_loss_k_sum": float(params.get("total_minor_k") or 0.0),
            "fixed_minor_loss_m": float(params.get("fixed_minor_loss_m") or 0.0)
        }]
    elif len(valid_segments) == 1 and params.get("total_minor_k"):
        if float(valid_segments[0].get("minor_loss_k_sum") or 0.0) == 0.0:
            valid_segments[0]["minor_loss_k_sum"] = float(params["total_minor_k"])

    breakdown_rows = []
    total_major_loss = 0.0
    total_minor_loss = 0.0

    for idx, seg in enumerate(valid_segments, start=1):
        name = seg.get("segment_name", f"ช่วงท่อที่ {idx}")
        is_unit_flow = bool(seg.get("carries_single_pump_flow", False))
        q_seg = (q_total / max(num_pumps, 1)) if is_unit_flow else q_total

        d_m = seg.get("diameter_m")
        area = seg.get("area_m2")
        w_perim = seg.get("wetted_perimeter_m")

        if d_m and float(d_m) > 0:
            d_m = float(d_m)
            area = (math.pi * (d_m ** 2)) / 4.0 if not area else float(area)
            r_h = d_m / 4.0
        elif area and float(area) > 0:
            area = float(area)
            d_m = math.sqrt((4.0 * area) / math.pi)
            r_h = (area / float(w_perim)) if (w_perim and float(w_perim) > 0) else (d_m / 4.0)
        else:
            area = 1.0
            d_m = 1.0
            r_h = 0.25

        velocity = q_seg / area if area > 0 else 0.0
        length = float(seg.get("length_m") or 0.0)
        n_val = float(seg.get("manning_n") or params.get("manning_n") or 0.014)
        f_val = float(seg.get("darcy_f") or params.get("darcy_f") or 0.015)
        k_sum = float(seg.get("minor_loss_k_sum") or 0.0)
        fixed_minor = float(seg.get("fixed_minor_loss_m") or 0.0)

        calc_method = formula_override or seg.get("preferred_formula") or "Manning"
        if "darcy" in calc_method.lower():
            h_major = f_val * (length / d_m) * ((velocity ** 2) / (2.0 * g)) if length > 0 else 0.0
            method_label = f"Darcy-Weisbach (f={f_val:.4f})"
        else:
            h_major = ((n_val ** 2) * (velocity ** 2) * length) / (r_h ** (4.0 / 3.0)) if length > 0 else 0.0
            method_label = f"Manning (n={n_val:.3f})"

        h_minor = (k_sum * ((velocity ** 2) / (2.0 * g))) + fixed_minor
        h_seg_total = h_major + h_minor

        total_major_loss += h_major
        total_minor_loss += h_minor

        breakdown_rows.append({
            "ลำดับ": idx,
            "ช่วงระบบท่อ / อุปกรณ์": name,
            "Q ในช่วงท่อ (m³/s)": round(q_seg, 2),
            "พื้นที่หน้าตัด A (m²)": round(area, 2),
            "ความเร็ว V (m/s)": round(velocity, 2),
            "ความยาว L (m)": round(length, 1),
            "สมการ Major Loss": method_label if length > 0 else "-",
            "Major Loss (m)": round(h_major, 4),
            "รวมค่า K (Minor)": round(k_sum, 3),
            "Minor Loss (m)": round(h_minor, 4),
            "รวม Head Loss (m)": round(h_seg_total, 4)
        })

    total_dynamic_loss = total_major_loss + total_minor_loss
    total_system_head = h_static + total_dynamic_loss
    k_system_m3s = (total_dynamic_loss / (q_total ** 2)) if q_total > 0 else 0.0

    return {
        "q_total_m3s": q_total,
        "q_total_m3h": q_total * 3600.0,
        "h_static_m": h_static,
        "total_major_loss_m": total_major_loss,
        "total_minor_loss_m": total_minor_loss,
        "total_dynamic_loss_m": total_dynamic_loss,
        "total_system_head_m": total_system_head,
        "k_system_m3s": k_system_m3s,
        "breakdown_df": pd.DataFrame(breakdown_rows)
    }


# ==========================================
# 3. ฟังก์ชันคำนวณกราฟปั๊มและหาจุดตัด (PHASE 3: PUMP CURVE ENGINE)
# ==========================================
def evaluate_pump_vs_system(df_points, num_pumps, k_sys_m3s, h_static_m, q_target_total_m3s, use_m3h=False):
    scale = 3600.0 if use_m3h else 1.0
    k_sys = k_sys_m3s / (scale ** 2)
    q_target_plot = q_target_total_m3s * scale

    clean_df = df_points.dropna(subset=["Q_single_m3s", "Head_m"]).copy()
    clean_df = clean_df[clean_df["Head_m"] > 0].sort_values("Q_single_m3s")

    is_synthesized = len(clean_df) < 3

    if not is_synthesized:
        q_single_arr = clean_df["Q_single_m3s"].values.astype(float)
        q_total_arr = q_single_arr * num_pumps * scale
        h_arr = clean_df["Head_m"].values.astype(float)

        coeffs = np.polyfit(q_total_arr, h_arr, 2)
        a_p, b_p, c_p = coeffs[0], coeffs[1], coeffs[2]
        curve_type_label = "Exact Model Curve (Polynomial Fit จากพิกัดผู้ผลิต)"

        if "Eff_pct" in clean_df.columns and clean_df["Eff_pct"].dropna().count() >= 3:
            eff_coef = np.polyfit(q_total_arr, clean_df["Eff_pct"].fillna(0).values.astype(float), 2)
        else:
            eff_coef = None
    else:
        if len(clean_df) >= 1:
            q_duty_single = float(clean_df["Q_single_m3s"].iloc[-1])
            h_duty = float(clean_df["Head_m"].iloc[-1])
            eff_col = clean_df["Eff_pct"] if "Eff_pct" in clean_df.columns else pd.Series([85.0])
            eff_duty = float(eff_col.iloc[-1]) if pd.notna(eff_col.iloc[-1]) else 85.0
        else:
            q_duty_single = q_target_total_m3s / max(num_pumps, 1)
            h_duty = h_static_m * 1.5
            eff_duty = 85.0

        if q_duty_single <= 0:
            q_duty_single = (q_target_total_m3s / max(num_pumps, 1)) if q_target_total_m3s > 0 else 10.0

        q_duty_total_plot = q_duty_single * num_pumps * scale
        h_shutoff = 1.30 * h_duty
        a_p = -0.30 * h_duty / (q_duty_total_plot ** 2)
        b_p = 0.0
        c_p = h_shutoff
        curve_type_label = "Synthesized Curve (กราฟจำลองทางวิศวกรรมจากจุด Duty Point)"
        eff_coef = [-eff_duty / (q_duty_total_plot ** 2), (2.0 * eff_duty) / q_duty_total_plot, 0.0]

    A_eq = a_p - k_sys
    B_eq = b_p
    C_eq = c_p - h_static_m

    q_op_plot = None
    h_op = None
    eff_op = None

    if abs(A_eq) > 1e-12:
        disc = (B_eq ** 2) - (4.0 * A_eq * C_eq)
        if disc >= 0:
            r1 = (-B_eq + math.sqrt(disc)) / (2.0 * A_eq)
            r2 = (-B_eq - math.sqrt(disc)) / (2.0 * A_eq)
            pos_roots = [r for r in (r1, r2) if r > 0]
            if pos_roots:
                q_op_plot = min(pos_roots, key=lambda r: abs(r - q_target_plot)) if len(pos_roots) > 1 else pos_roots[0]
    elif abs(B_eq) > 1e-12:
        r = -C_eq / B_eq
        if r > 0:
            q_op_plot = r

    if q_op_plot is not None:
        h_op = (k_sys * (q_op_plot ** 2)) + h_static_m
        if eff_coef is not None:
            eff_op = float(np.polyval(eff_coef, q_op_plot))
            eff_op = max(0.0, min(95.0, eff_op))

    return {
        "is_synthesized": is_synthesized,
        "curve_type_label": curve_type_label,
        "poly_coeffs": (a_p, b_p, c_p),
        "eff_coeffs": eff_coef,
        "q_op_plot": q_op_plot,
        "q_op_m3s": (q_op_plot / scale) if q_op_plot is not None else None,
        "h_op_m": h_op,
        "eff_op_pct": eff_op,
        "clean_points_df": clean_df
    }


# ==========================================
# 4. ฟังก์ชันพิจารณาความเหมาะสมและสร้างข้อเสนอแนะ (PHASE 4 & 5 ENGINE)
# ==========================================
def select_standard_motor_kw(min_req_kw):
    standard_ratings = [
        1.5, 2.2, 3.0, 4.0, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90,
        110, 132, 160, 200, 250, 315, 355, 400, 450, 500, 560, 630, 710, 800,
        900, 1000, 1250, 1500, 1800, 2000, 2200, 2400, 2500, 2800, 3000, 3200,
        3500, 4000, 4500, 5000, 6000
    ]
    for rating in standard_ratings:
        if rating >= min_req_kw:
            return float(rating)
    return math.ceil(min_req_kw / 100.0) * 100.0


def evaluate_pipe_velocity_only(calc_res):
    df_bd = calc_res["breakdown_df"]
    vel_warnings = []
    vel_details = []

    for _, row in df_bd.iterrows():
        seg_name = row["ช่วงระบบท่อ / อุปกรณ์"]
        v_val = float(row["ความเร็ว V (m/s)"])
        area_val = float(row["พื้นที่หน้าตัด A (m²)"])
        is_siphon_or_culvert = any(k in seg_name.lower() for k in ["ไซฟอน", "siphon", "culvert", "ช่องรับน้ำ"])
        v_max_limit = 3.0 if is_siphon_or_culvert else 3.5

        if v_val > v_max_limit:
            vel_warnings.append(
                f"🔴 **{seg_name}:** ความเร็วน้ำสูงเกินเกณฑ์มาตรฐาน ($V = {v_val:.2f}\\text{{ m/s}} > {v_max_limit:.1f}\\text{{ m/s}}$ บนพื้นที่หน้าตัด ${area_val:.2f}\\text{{ m}}^2$) "
                f"→ **ผลกระทบเชิงวิศวกรรม:** ทำให้สูญเสียเฮด (Dynamic Loss) สูงผิดปกติ เสี่ยงต่อการกัดเซาะผนังท่อ/คอนกรีต (Erosion) เกิดการสั่นสะเทือนทางชลศาสตร์ และแรงกระแทกค้อนน้ำ (Water Hammer)"
            )
        elif 0 < v_val < 0.6 and "ปล่อง" not in seg_name and "shaft" not in seg_name.lower():
            vel_warnings.append(
                f"🟡 **{seg_name}:** ความเร็วน้ำต่ำกว่าเกณฑ์ทำความสะอาดตัวเอง ($V = {v_val:.2f}\\text{{ m/s}} < 0.60\\text{{ m/s}}$) "
                f"→ **ผลกระทบเชิงวิศวกรรม:** เสี่ยงต่อการตกตะกอนสะสมของทรายและตะกอนท้องท่อ (Sedimentation / Siltation) ทำให้ประสิทธิภาพการลำเลียงน้ำลดลงในระยะยาว"
            )
        else:
            vel_details.append(
                f"🟢 **{seg_name}:** $V = {v_val:.2f}\\text{{ m/s}}$ (พื้นที่หน้าตัด ${area_val:.2f}\\text{{ m}}^2$) — *อยู่ในเกณฑ์มาตรฐานวิศวกรรม ($0.6 - {v_max_limit:.1f}\\text{{ m/s}}$)*"
            )

    pipe_pass = len(vel_warnings) == 0
    if pipe_pass:
        pipe_status = "✅ ขนาดท่อเหมาะสม (PASS)"
        pipe_narrative = "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ในทุกช่วงของแนวท่อและอุโมงค์ พบว่าขนาดหน้าตัดท่อมีความเหมาะสม ความเร็วน้ำอยู่ในช่วงมาตรฐานวิศวกรรม ไม่พบจุดเสี่ยงต่อการกัดเซาะหรือค้อนน้ำรุนแรง"
    else:
        pipe_status = f"⚠️ พบจุดเสี่ยงความเร็วน้ำ ({len(vel_warnings)} จุด)"
        pipe_narrative = "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ตามขนาดหน้าตัดท่อที่กำหนด **พบช่วงท่อที่มีความเร็วน้ำอยู่นอกเกณฑ์แนะนำทางวิศวกรรม** ซึ่งส่งผลกระทบโดยตรงต่อความมั่นคงของระบบดังนี้:"

    return {
        "pipe_pass": pipe_pass,
        "pipe_status": pipe_status,
        "pipe_narrative": pipe_narrative,
        "vel_warnings": vel_warnings,
        "vel_details": vel_details
    }


def evaluate_phase4_and_5(
    working_data, calc_res, pump_eval, num_active_pumps,
    pump_eff_pct, gear_eff_pct, safety_margin_pct, npshr_m,
    user_motor_kw, is_force_unlocked=False, missing_step4_list=None
):
    rho = 1000.0
    g = 9.81

    q_target_total = calc_res["q_total_m3s"]
    q_target_unit = q_target_total / max(num_active_pumps, 1)
    h_sys_req = calc_res["total_system_head_m"]
    h_static = calc_res["h_static_m"]
    h_dyn_loss = calc_res["total_dynamic_loss_m"]

    q_op_total = pump_eval["q_op_m3s"]
    h_op = pump_eval["h_op_m"]
    a_p, b_p, c_p = pump_eval["poly_coeffs"]

    pump_h_at_target = float(working_data.get("proposed_pump_head_m") or 0.0)
    if pump_h_at_target <= 0 and q_target_total > 0:
        pump_h_at_target = float((a_p * (q_target_total ** 2)) + (b_p * q_target_total) + c_p)

    # 1. รูปแบบการเดินเครื่อง
    if num_active_pumps > 1:
        mode_title = f"การเดินเครื่องสูบน้ำแบบขนาน (Parallel Operation จำนวน {num_active_pumps} เครื่อง)"
        graph_narrative = (
            f"จากกราฟเปรียบเทียบ **System Curve vs. Pump Curve** ระบบถูกกำหนดให้ **{mode_title}** "
            f"ซึ่งในทางชลศาสตร์ การต่อปั๊มขนานกันจะทำให้อัตราการไหลรวม ($Q_{{total}}$) เพิ่มขึ้นเป็น {num_active_pumps} เท่าของปั๊มเดี่ยว ณ ระดับ Head เดียวกัน "
            f"ในขณะที่เส้นสีฟ้า (**System Curve**) เริ่มต้นที่ระยะยกน้ำสถิต $H_{{st}} = {h_static:.2f}\\text{{ m}}$ และโค้งพาราโบลาชันขึ้นตามความเสียดทานในแนวท่อและอุปกรณ์ ($H_{{loss}} = KQ^2$)"
        )
    else:
        mode_title = "การเดินเครื่องสูบน้ำเดี่ยว (Single Pump Operation จำนวน 1 เครื่อง)"
        graph_narrative = (
            f"จากกราฟเปรียบเทียบ **System Curve vs. Pump Curve** ระบบทำงานแบบ **{mode_title}** "
            f"โดยเส้นสีฟ้า (**System Curve**) เริ่มต้นที่ระยะยกน้ำสถิต $H_{{st}} = {h_static:.2f}\\text{{ m}}$ และโค้งชันขึ้นตามอัตราการไหลยกกำลังสอง ($H = KQ^2 + H_{{st}}$) "
            f"ตัดกับเส้นกราฟสมรรถนะของปั๊ม (เส้นสีชมพู) ซึ่งมีจุด Shutoff Head (ที่ $Q=0$) อยู่ที่ประมาณ ${c_p:.2f}\\text{{ m}}$"
        )

    # 2. ความเพียงพอของ Q และ H
    if q_op_total is not None and h_op is not None:
        q_op_unit = q_op_total / max(num_active_pumps, 1)
        q_diff = q_op_total - q_target_total
        h_margin = pump_h_at_target - h_sys_req
        qh_pass = (q_diff >= -0.05) and (h_op >= h_sys_req - 0.15)

        if qh_pass:
            qh_status = "✅ เพียงพอตามเกณฑ์ (PASS)"
            qh_narrative = (
                f"**ผ่านเกณฑ์ด้านอัตราการไหลและเฮด:** ที่อัตราการไหลเป้าหมายของโครงการ $Q_{{target}} = {q_target_total:,.2f}\\text{{ m}}^3/\\text{{s}}$ "
                f"({q_target_unit:,.2f} $\\text{{m}}^3/\\text{{s}}$ ต่อเครื่อง) ระบบต้องการเฮดรวม (Total System Head) เท่ากับ **${h_sys_req:.3f}\\text{{ m}}$** "
                f"เมื่อพล็อตกราฟตัดกัน จุดทำงานจริง (**Operating Point**) ไปตัดกันที่ **$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** "
                f"และ **$H_{{op}} = {h_op:.2f}\\text{{ m}}$** ซึ่งให้ปริมาณน้ำเกินกว่าเป้าหมายอยู่ $+{max(q_diff, 0.0):.2f}\\text{{ m}}^3/\\text{{s}}$ สามารถสูบระบายน้ำเอาชนะความเสียดทานในระบบได้อย่างสมบูรณ์"
            )
        else:
            qh_status = "❌ ไม่เพียงพอ (FLOW/HEAD DEFICIT)"
            deficit_pct = (abs(q_diff) / q_target_total * 100.0) if q_target_total > 0 else 0.0
            qh_narrative = (
                f"**ไม่ผ่านเกณฑ์ด้านอัตราการไหลและเฮด:** ระบบมีความต้านทานรวมสูงถึง **${h_sys_req:.3f}\\text{{ m}}$** แต่กราฟของปั๊มที่เสนอให้เฮดไม่เพียงพอ "
                f"ทำให้บนกราฟ เส้นปั๊มตกลงมาตัดกับเส้น System Curve ก่อนถึงเป้าหมาย โดยจุดตัด **Operating Point** จริงทำอัตราการไหลได้เพียง "
                f"**$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** ที่เฮด **${h_op:.2f}\\text{{ m}}$** "
                f"**ส่งผลให้อัตราการไหลสูบน้ำขาดหายไปถึง ${abs(q_diff):,.2f}\\text{{ m}}^3/\\text{{s}}$ (คิดเป็น $-{deficit_pct:.1f}\\%$)** ไม่สามารถระบายน้ำได้ทันตามเกณฑ์โครงการ"
            )
    else:
        qh_pass = False
        q_op_unit = 0.0
        q_diff = -q_target_total
        qh_status = "❌ กราฟไม่ตัดกัน (SHUTOFF FAIL)"
        qh_narrative = (
            f"**วิกฤต (ปั๊มไม่สามารถส่งน้ำได้):** เฮดสูงสุดของปั๊ม (Shutoff Head = ${c_p:.2f}\\text{{ m}}$) มีค่าต่ำกว่าระยะยกน้ำสถิตของระบบ ($H_{{st}} = {h_static:.2f}\\text{{ m}}$) "
            f"ทำให้เส้นกราฟปั๊มไม่ตัดกับ System Curve เลย น้ำไม่สามารถไหลผ่านระบบได้"
        )

    # 3. Efficiency / Power / Motor Margin
    eff_pump = float(pump_eff_pct or 85.0)
    if eff_pump <= 0:
        eff_pump = 80.0
    eff_gear = float(gear_eff_pct or 98.0)

    q_calc_unit = max(q_op_unit, q_target_unit)
    h_calc_power = max(h_op or 0.0, pump_h_at_target, h_sys_req)

    p_hydraulic_kw = (rho * g * q_calc_unit * h_calc_power) / 1000.0
    p_pump_shaft_kw = p_hydraulic_kw / (eff_pump / 100.0)
    p_gear_shaft_kw = p_pump_shaft_kw / (eff_gear / 100.0)
    p_min_motor_kw = p_gear_shaft_kw * (safety_margin_pct / 100.0)
    rec_standard_motor_kw = select_standard_motor_kw(p_min_motor_kw)

    has_motor = (user_motor_kw is not None) and (float(user_motor_kw or 0) > 0)
    eff_good = 78.0 <= eff_pump <= 92.0
    eff_desc = f"อยู่ในเกณฑ์ดีและใกล้เคียงจุด BEP ({eff_pump:.1f}%)" if eff_good else f"ค่อนข้างต่ำและตกอยู่นอกช่วง BEP ({eff_pump:.1f}%) ทำให้สูญเสียพลังงานสูง"

    if has_motor:
        prop_motor_kw = float(user_motor_kw)
        motor_pass = prop_motor_kw >= p_min_motor_kw
        actual_margin_pct = (prop_motor_kw / p_gear_shaft_kw) * 100.0 if p_gear_shaft_kw > 0 else 0.0

        if motor_pass and eff_good:
            power_status = f"✅ เหมาะสม ({prop_motor_kw:,.0f} kW)"
            power_narrative = (
                f"**ผ่านเกณฑ์ด้านประสิทธิภาพและพิกัดกำลังมอเตอร์:** ณ จุดพิจารณา ประสิทธิภาพเครื่องสูบน้ำอยู่ที่ **{eff_pump:.1f}%** ({eff_desc}) "
                f"และประสิทธิภาพชุดเกียร์ทดรอบอยู่ที่ **{eff_gear:.1f}%** คำนวณกำลังเพลาเกียร์ (Gear Input Power) ได้ **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคูณเผื่อ Safety Margin ที่ **{safety_margin_pct:.0f}%** จะต้องการขนาดมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"ดังนั้น มอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** ที่ใช้พิจารณาจึงมีความเหมาะสม (คิดเป็น Safety Margin จริง **{actual_margin_pct:.1f}%**) ไม่มีความเสี่ยงเรื่อง Motor Overload"
            )
        elif not motor_pass:
            power_status = f"❌ เสี่ยง Overload ({prop_motor_kw:,.0f} kW < {p_min_motor_kw:,.0f} kW)"
            overload_deficit = p_min_motor_kw - prop_motor_kw
            power_narrative = (
                f"**ไม่ผ่านเกณฑ์พิกัดกำลังมอเตอร์ (เสี่ยงเกิด Motor Overload):** ปั๊มต้องใช้กำลังเพลาเกียร์สูงถึง **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** จะต้องใช้มอเตอร์ขนาดอย่างน้อย **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"แต่มอเตอร์ที่เสนอมีขนาดเพียง **${prop_motor_kw:,.0f}\\text{{ kW}}$** (ขาดไป **${overload_deficit:,.1f}\\text{{ kW}}$**) "
                f"**ผลกระทบ:** มอเตอร์จะเกิดความร้อนสะสม รันเกินพิกัด และเบรกเกอร์ทริปตัดการทำงานเมื่อเดินเครื่องเต็มกำลัง **แนะนำให้เปลี่ยนไปใช้มอเตอร์พิกัดมาตรฐาน ${rec_standard_motor_kw:,.0f}\\text{{ kW}}$ ต่อเครื่อง**"
            )
        else:
            power_status = f"⚠️ มอเตอร์พอแต่ Eff ต่ำ ({eff_pump:.1f}%)"
            power_narrative = (
                f"**ข้อควรระวังด้านประสิทธิภาพ (Off-BEP):** แม้ว่ามอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** จะสูงกว่าค่าขั้นต่ำที่ต้องการ "
                f"แต่ประสิทธิภาพของปั๊มต่ำเพียง **{eff_pump:.1f}%** แสดงว่าปั๊มทำงานห่างจากจุด Best Efficiency Point (BEP) ทำให้สิ้นเปลืองพลังงานไฟฟ้าสูง"
            )
    else:
        motor_pass = True
        prop_motor_kw = None
        actual_margin_pct = safety_margin_pct
        power_status = f"💡 แนะนำพิกัด {rec_standard_motor_kw:,.0f} kW"
        power_narrative = (
            f"**การคำนวณและกำหนดพิกัดมอเตอร์แนะนำ:** ระบบต้องการกำลังเพลาเกียร์ (Gear Input Power) **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
            f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** จะต้องการกำลังมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
            f"👉 **ขอแนะนำให้กำหนดสเปคมอเตอร์ไฟฟ้าขนาดพิกัดมาตรฐาน (Standard Commercial Motor Rating) ที่ขนาด `${rec_standard_motor_kw:,.0f} kW` ต่อเครื่อง**"
        )

    # 4. ความเหมาะสมของขนาดท่อและความเร็วน้ำ
    pipe_eval = evaluate_pipe_velocity_only(calc_res)

    # 5. จุดเสี่ยง Off-BEP, ระดับน้ำบ่อสูบ และ NPSH / Cavitation Flags
    risk_flags = []
    suction_wl = working_data.get("suction_wl_m")
    if suction_wl is not None:
        suction_wl = float(suction_wl)
        sump_wl = suction_wl - calc_res["total_major_loss_m"] - (calc_res["total_minor_loss_m"] * 0.85)
    else:
        suction_wl = -3.00
        sump_wl = -3.00 - (h_dyn_loss * 0.88)

    npsh_pass = npshr_m <= 6.5
    if not npsh_pass:
        risk_flags.append(
            f"🚨 **ความเสี่ยงเกิดโพรงอากาศรุนแรง (High NPSHr / Cavitation Flag):** ปั๊มมีค่า $\\text{{NPSHr}}$ สูงถึง **${npshr_m:.1f}\\text{{ m}}$** "
            f"ในขณะที่ระดับน้ำในบ่อสูบขณะเดินเครื่อง (Operating Sump WL) ลดต่ำลงไปอยู่ที่ประมาณ **${sump_wl:.2f}\\text{{ m.MSL}}$** ทำให้ค่า $\\text{{NPSHa}}$ ลดต่ำลง เสี่ยงต่อการเกิด Cavitation กัดกร่อนใบพัดและเกิดการสั่นสะเทือนรุนแรง"
        )
    else:
        risk_flags.append(
            f"✅ **การตรวจสอบ NPSH และระดับน้ำบ่อสูบ:** ค่า $\\text{{NPSHr}}$ อยู่ที่ **${npshr_m:.1f}\\text{{ m}}$** อยู่ในเกณฑ์ปลอดภัย "
            f"ทั้งนี้ผู้ควบคุมงานควรกำหนดค่าระดับน้ำหยุดปั๊มอัตโนมัติ (Low Water Level Interlock) ให้สอดคล้องกับระดับน้ำในบ่อสูบขณะเดินเครื่องเต็มพิกัด (**${sump_wl:.2f}\\text{{ m.MSL}}$**)"
        )

    if eff_pump < 78.0:
        risk_flags.append(
            f"⚠️ **ความเสี่ยงการทำงานนอกจุดออกแบบ (Off-BEP / Runout Zone):** ประสิทธิภาพปั๊มลดต่ำลงเหลือเพียง **{eff_pump:.1f}%** บ่งชี้ว่าจุดตัดการทำงานเลื่อนไปอยู่ปลายกราฟด้านขวา เสี่ยงต่อการเกิด Radial Thrust บนเพลาและซีลเสื่อมสภาพเร็ว"
        )

    if pump_eval["is_synthesized"]:
        risk_flags.append(
            "ℹ️ **ข้อจำกัดด้านข้อมูลกราฟปั๊ม (Synthesized Curve Assumption):** เนื่องจากข้อมูลนำเข้ามีพิกัดปั๊มน้อยกว่า 3 จุด เส้นโค้งสมรรถนะจึงถูกสร้างขึ้นด้วยสมการจำลอง ($H_{{shutoff}} = 1.30 H_{{duty}}$) ผู้ควบคุมงานต้องเรียกขอ Factory Test Curve ฉบับจริงจากผู้รับจ้างเพื่อยืนยันก่อนอนุมัติสั่งซื้อ"
        )

    if is_force_unlocked and missing_step4_list:
        risk_flags.append(
            f"ℹ️ **หมายเหตุการกำหนดพารามิเตอร์จำลองโดยผู้ใช้งาน (Engineering Override Active):** การประเมินนี้ใช้ตัวเลขจำลองที่ผู้ใช้งานกำหนดแทนข้อมูลที่ไม่ได้ระบุในเอกสารต้นฉบับ ได้แก่: "
            f"**{', '.join(missing_step4_list)}**"
        )

    # ------------------------------------------------------------------
    # PHASE 5: สร้างข้อเสนอแนะการปรับปรุงระบบ (Actionable Recommendations) & สรุปสถานะ
    # ------------------------------------------------------------------
    recommendations = []
    if not qh_pass:
        req_duty_h = math.ceil((h_sys_req + 0.5) * 2.0) / 2.0
        recommendations.append(
            f"**ปรับปรุงสเปคเฮดเครื่องสูบน้ำ (Pump Head Up-rating):** ให้แจ้งผู้รับจ้างเปลี่ยนรุ่นเครื่องสูบน้ำหรือขยายขนาดเส้นผ่านศูนย์กลางใบพัด (Impeller Diameter) "
            f"เพื่อให้สามารถทำ Rated Head ได้ไม่น้อยกว่า **{req_duty_h:.1f} m** ณ อัตราการไหล **{q_target_unit:.2f} m³/s ต่อเครื่อง** (เพื่อให้ชนะ Total System Head {h_sys_req:.3f} m)"
        )
    if has_motor and not motor_pass:
        recommendations.append(
            f"**เพิ่มพิกัดกำลังมอเตอร์ไฟฟ้า (Motor Up-sizing):** ปฏิเสธการใช้มอเตอร์ขนาด {prop_motor_kw:,.0f} kW เนื่องจากไม่ผ่านเกณฑ์ Safety Margin {safety_margin_pct:.0f}% "
            f"และสั่งการให้ผู้รับจ้างปรับเพิ่มขนาดมอเตอร์เป็นรุ่นมาตรฐาน **{rec_standard_motor_kw:,.0f} kW ต่อเครื่อง** (ความต้องการขั้นต่ำคือ {p_min_motor_kw:,.1f} kW)"
        )
    elif not has_motor:
        recommendations.append(
            f"**กำหนดพิกัดมอเตอร์ในใบสั่งซื้อ (Motor Specification):** ให้ระบุขนาดมอเตอร์ไฟฟ้าในเอกสารอนุมัติที่พิกัดมาตรฐาน **{rec_standard_motor_kw:,.0f} kW ต่อเครื่อง** "
            f"(รองรับกำลังเพลาเกียร์ {p_gear_shaft_kw:,.1f} kW พร้อม Safety Margin {safety_margin_pct:.0f}%)"
        )
    if not eff_good:
        recommendations.append(
            f"**คัดเลือกปั๊มที่มีจุด BEP ตรงกับจุดทำงานจริง (BEP Re-selection):** ประสิทธิภาพปัจจุบัน ({eff_pump:.1f}%) ต่ำกว่าเกณฑ์มาตรฐาน (≥ 78–80%) "
            f"ควรให้ผู้รับจ้างเสนอปั๊มรุ่นที่มีจุด Best Efficiency Point อยู่ใกล้พิกัด Q = {q_target_unit:.2f} m³/s, H = {h_sys_req:.2f} m หรือพิจารณาติดตั้งระบบปรับความเร็วรอบ (VFD/VSD)"
        )
    if not pipe_eval["pipe_pass"]:
        recommendations.append(
            "**ปรับปรุงขนาดหน้าตัดท่อหรือลดการสูญเสียรอง (Hydraulic Conduit Optimization):** เนื่องจากพบช่วงท่อที่มีความเร็วน้ำเกินเกณฑ์แนะนำ "
            "ควรพิจารณาขยายขนาดเส้นผ่านศูนย์กลางท่อในจุดที่เกิดคอขวด หรือเลือกใช้อุปกรณ์ข้องอ/วาล์วที่มีค่าสัมประสิทธิ์การสูญเสีย (K) ต่ำลงเพื่อลด Dynamic Head Loss"
        )
    if not npsh_pass:
        recommendations.append(
            f"**ป้องกันปัญหาโพรงอากาศ (Cavitation Mitigation):** ค่า NPSHr ({npshr_m:.1f} m) สูงเกินเกณฑ์ปลอดภัยเมื่อเทียบกับระดับน้ำบ่อสูบ ({sump_wl:.2f} m.MSL) "
            "ให้ผู้รับจ้างเปลี่ยนไปใช้ปั๊มที่มีความเร็วจำเพาะฝั่งดูดดีขึ้น (NPSHr ≤ 5.0 m) หรือลดระดับการติดตั้งใบพัดปั๊ม (Submergence Depth) ให้ลึกลง"
        )
    if pump_eval["is_synthesized"]:
        recommendations.append(
            "**เงื่อนไขการตรวจรับเอกสาร (Submittal Condition):** ให้ผู้รับจ้างแนบกราฟทดสอบสมรรถนะจากโรงงานผู้ผลิต (Official ISO 9906 Pump Performance Curve) ที่แสดงเส้น Q-H, Efficiency, Power และ NPSHr ครบตลอดย่านการทำงาน"
        )
    if len(recommendations) == 0:
        recommendations.append(
            "**สเปคมีความสมบูรณ์พร้อมอนุมัติ (Ready for Approval):** พารามิเตอร์ทางชลศาสตร์และทางกลทั้งหมดผ่านเกณฑ์มาตรฐานวิศวกรรม แนะนำให้อนุมัติเอกสาร Submittal และกำหนดให้มีการทดสอบสมรรถนะที่โรงงาน (FAT ตามมาตรฐาน ISO 9906 Grade 1B/2B) ก่อนจัดส่งเข้าพื้นที่โครงการ"
        )

    # กำหนดป้ายตัดสินสถานะรวม (Verdict Status)
    hard_pass = qh_pass and motor_pass and npsh_pass
    soft_warnings = (not eff_good) or (not pipe_eval["pipe_pass"]) or pump_eval["is_synthesized"] or is_force_unlocked

    if hard_pass and not soft_warnings:
        verdict_code = "APPROVED"
        verdict_badge = "✅ เหมาะสม (APPROVED FOR CONSTRUCTION)"
        verdict_title = "ผลการตรวจรับ: เครื่องสูบน้ำและระบบขับเคลื่อนมีความเหมาะสมตามหลักวิศวกรรม"
        verdict_sub = "ผ่านเกณฑ์ทั้งด้านอัตราการไหล (Q), เฮดรวมของระบบ (Total Head), ประสิทธิภาพ (Efficiency), พิกัดกำลังมอเตอร์ (105% Margin) และเกณฑ์ความเร็วน้ำในท่อ"
    elif hard_pass and soft_warnings:
        verdict_code = "CONDITIONAL"
        verdict_badge = "⚠️ เหมาะสมแบบมีเงื่อนไข (CONDITIONALLY APPROVED)"
        verdict_title = "ผลการตรวจรับ: ผ่านเกณฑ์หลักทางชลศาสตร์และมอเตอร์ แต่มีข้อสังเกตที่ต้องยืนยันเพิ่มเติม"
        verdict_sub = "เครื่องสูบน้ำสามารถส่งอัตราการไหลและเฮดได้ตามต้องการ แต่ผู้รับจ้างต้องดำเนินการตามข้อเสนอแนะด้านล่างก่อนการอนุมัติขั้นสุดท้าย"
    else:
        verdict_code = "REJECTED"
        verdict_badge = "❌ ไม่เหมาะสม (REJECTED / REVISION REQUIRED)"
        verdict_title = "ผลการตรวจรับ: ไม่ผ่านเกณฑ์มาตรฐานวิศวกรรม (ไม่อนุมัติสเปคฉบับนี้)"
        verdict_sub = "พบข้อบกพร่องร้ายแรงด้านความสามารถในการส่งน้ำ (Flow/Head Deficit), พิกัดกำลังมอเตอร์เสี่ยงเกิด Overload หรือมีความเสี่ยงเกิดโพรงอากาศ (Cavitation) ต้องส่งคืนให้ผู้รับจ้างแก้ไข"

    # ตารางสรุปเปรียบเทียบสำหรับแสดงใน Phase 5
    summary_table_rows = [
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": "1. อัตราการไหลรวมของระบบ (Total Flow Rate, Q)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": f"{q_target_total:,.2f} m³/s ({num_active_pumps} เครื่อง)",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"{q_op_total:,.2f} m³/s" if q_op_total else "ไม่ตัดกัน",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": f"{q_diff:+,.2f} m³/s" if q_op_total else "N/A",
            "ผลการประเมิน (Status)": "✅ ผ่านเกณฑ์" if qh_pass else "❌ ไม่ผ่านเกณฑ์"
        },
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": "2. เฮดรวมของระบบ (Total System Head, H)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": f"{h_sys_req:,.3f} m (Hst {h_static:.2f} + Loss {h_dyn_loss:.3f})",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"{h_op:,.2f} m (Duty Spec: {pump_h_at_target:.2f} m)" if h_op else "ไม่พอ",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": f"{(pump_h_at_target - h_sys_req):+,.3f} m",
            "ผลการประเมิน (Status)": "✅ ผ่านเกณฑ์" if qh_pass else "❌ เฮดไม่พอ"
        },
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": "3. ประสิทธิภาพเครื่องสูบน้ำ (Pump Efficiency, ηp)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": "≥ 78.0% (ใกล้จุด BEP)",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"{eff_pump:.1f}%",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": f"{(eff_pump - 78.0):+.1f}%",
            "ผลการประเมิน (Status)": "✅ เหมาะสม" if eff_good else "⚠️ ต่ำกว่าเกณฑ์/Off-BEP"
        },
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": f"4. พิกัดกำลังมอเตอร์ต่อเครื่อง (เผื่อ Margin {safety_margin_pct:.0f}%)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": f"≥ {p_min_motor_kw:,.1f} kW (กำลังเพลาเกียร์ {p_gear_shaft_kw:,.1f} kW)",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"{prop_motor_kw:,.0f} kW" if has_motor else f"แนะนำ {rec_standard_motor_kw:,.0f} kW",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": f"{(prop_motor_kw - p_min_motor_kw):+,.1f} kW ({actual_margin_pct:.1f}%)" if has_motor else "ตามพิกัดแนะนำ",
            "ผลการประเมิน (Status)": "✅ ผ่านเกณฑ์" if motor_pass else "❌ เสี่ยง Overload"
        },
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": "5. ความเร็วน้ำในแนวท่อและอุโมงค์ (Flow Velocity, V)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": "0.60 – 3.50 m/s (ไซฟอน ≤ 3.0 m/s)",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"สูงสุด {calc_res['breakdown_df']['ความเร็ว V (m/s)'].max():.2f} m/s",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": "อยู่ในเกณฑ์" if pipe_eval["pipe_pass"] else f"เกินเกณฑ์ {len(pipe_eval['vel_warnings'])} จุด",
            "ผลการประเมิน (Status)": "✅ เหมาะสม" if pipe_eval["pipe_pass"] else "⚠️ พบจุดเสี่ยง"
        },
        {
            "พารามิเตอร์หลัก (Key Engineering Parameter)": "6. เกณฑ์ป้องกันโพรงอากาศ (NPSHr & Sump Level)",
            "เกณฑ์ความต้องการโครงการ (Project Requirement)": "NPSHr ≤ 6.5 m",
            "ค่าที่ทำได้จริง ณ จุดทำงาน (Operating Value)": f"NPSHr = {npshr_m:.1f} m (Sump WL {sump_wl:.2f} m.MSL)",
            "ส่วนต่าง / ระยะเผื่อ (Margin)": f"{(6.5 - npshr_m):+.1f} m",
            "ผลการประเมิน (Status)": "✅ ปลอดภัย" if npsh_pass else "🚨 เสี่ยง Cavitation"
        }
    ]

    return {
        "overall_suitable": hard_pass and not soft_warnings,
        "verdict_code": verdict_code,
        "verdict_badge": verdict_badge,
        "verdict_title": verdict_title,
        "verdict_sub": verdict_sub,
        "summary_table_df": pd.DataFrame(summary_table_rows),
        "recommendations": recommendations,
        "is_approx_unlocked": is_force_unlocked,
        "mode_title": mode_title,
        "graph_narrative": graph_narrative,
        "qh_pass": qh_pass,
        "qh_status": qh_status,
        "qh_narrative": qh_narrative,
        "power_pass": motor_pass and eff_good,
        "power_status": power_status,
        "power_narrative": power_narrative,
        "p_hydraulic_kw": p_hydraulic_kw,
        "p_pump_shaft_kw": p_pump_shaft_kw,
        "p_gear_shaft_kw": p_gear_shaft_kw,
        "p_min_motor_kw": p_min_motor_kw,
        "rec_standard_motor_kw": rec_standard_motor_kw,
        "proposed_motor_kw": prop_motor_kw,
        "eff_pump_pct": eff_pump,
        "eff_gear_pct": eff_gear,
        "pipe_pass": pipe_eval["pipe_pass"],
        "pipe_status": pipe_eval["pipe_status"],
        "pipe_narrative": pipe_eval["pipe_narrative"],
        "vel_warnings": pipe_eval["vel_warnings"],
        "vel_details": pipe_eval["vel_details"],
        "risk_flags": risk_flags,
        "sump_wl_m": sump_wl
    }


def generate_ai_official_memo(client, working_data, calc_res, pump_eval, phase4_res):
    """สร้างหนังสือบันทึกข้อความตรวจรับงานเครื่องกลอย่างเป็นทางการด้วย OpenAI"""
    prompt_data = {
        "โครงการ": working_data.get("project_name") or "โครงการระบบสูบน้ำและอุโมงค์ระบายน้ำ",
        "ผลการตัดสินรวม": phase4_res["verdict_badge"],
        "อัตราการไหลเป้าหมาย_m3s": calc_res["q_total_m3s"],
        "จุดตัดอัตราการไหลจริง_m3s": pump_eval["q_op_m3s"],
        "เฮดรวมของระบบ_m": round(calc_res["total_system_head_m"], 3),
        "จุดตัดเฮดทำงานจริง_m": round(pump_eval["h_op_m"] or 0, 2),
        "Major_Loss_m": round(calc_res["total_major_loss_m"], 3),
        "Minor_Loss_m": round(calc_res["total_minor_loss_m"], 3),
        "ประสิทธิภาพปั๊ม_pct": round(phase4_res["eff_pump_pct"], 1),
        "กำลังเพลาเกียร์_kW": round(phase4_res["p_gear_shaft_kw"], 1),
        "มอเตอร์ขั้นต่ำที่เผื่อ105pct_kW": round(phase4_res["p_min_motor_kw"], 1),
        "มอเตอร์ที่ผู้รับจ้างเสนอ_kW": phase4_res["proposed_motor_kw"],
        "มอเตอร์มาตรฐานแนะนำ_kW": phase4_res["rec_standard_motor_kw"],
        "ความเสี่ยงที่พบ": phase4_res["risk_flags"],
        "ข้อเสนอแนะวิศวกรรม": phase4_res["recommendations"]
    }

    memo_sys_prompt = """
    คุณคือหัวหน้าวิศวกรที่ปรึกษาอาวุโส (Chief Mechanical & Hydraulic Inspector) ของบริษัท TEAM Consulting Engineering and Management PCL.
    จงเขียน "บันทึกข้อความตรวจรับงานเครื่องกล (Official Technical Submittal Review Memo)" เป็นภาษาไทยที่สละสลวย เป็นทางการ และแม่นยำตามหลักวิศวกรรม
    โครงสร้างเอกสารประกอบด้วย:
    1. ส่วนหัวบันทึกข้อความ (เรื่อง, เรียน, วันที่, อ้างอิงโครงการ)
    2. สรุปผลการพิจารณาอนุมัติ (ระบุชัดเจนว่า อนุมัติ / อนุมัติแบบมีเงื่อนไข / ไม่อนุมัติและให้แก้ไข)
    3. ผลการตรวจสอบทางชลศาสตร์และจุดทำงาน (System Curve vs. Operating Point)
    4. ผลการตรวจสอบทางกลและพิกัดกำลังมอเตอร์ไฟฟ้า (Efficiency, Shaft Power, 105% Safety Margin, NPSH)
    5. ข้อสั่งการและข้อเสนอแนะสำหรับผู้รับจ้าง (Action Items)
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": memo_sys_prompt},
            {"role": "user", "content": f"สร้างบันทึกข้อความตรวจรับงานจากข้อมูลการคำนวณต่อไปนี้:\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"}
        ]
    )
    return resp.choices[0].message.content


# ==========================================
# 5. โครงสร้างคำสั่ง AI (OPENAI FUNCTION CALLING SCHEMA)
# ==========================================
extraction_tool = [{
    "type": "function",
    "function": {
        "name": "extract_hydraulic_parameters",
        "description": "สกัดพารามิเตอร์ระบบท่อ อุโมงค์ ท่อไซฟอน สเปคปั๊ม มอเตอร์ NPSHr และพิกัดจุดบนกราฟ Pump Curve",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "flow_rate_q_m3s": {"type": "number"},
                "flow_rate_original_text": {"type": "string"},
                "num_parallel_pumps": {"type": "integer"},
                "static_head_m": {"type": "number"},
                "suction_wl_m": {"type": "number"},
                "discharge_wl_m": {"type": "number"},
                "main_pipe_length_m": {"type": "number"},
                "main_pipe_diameter_m": {"type": "number"},
                "pipe_material_or_roughness": {"type": "string"},
                "manning_n": {"type": "number"},
                "darcy_f": {"type": "number"},
                "total_minor_k": {"type": "number"},
                "system_type": {"type": "string"},
                "proposed_pump_head_m": {"type": "number"},
                "proposed_pump_shutoff_head_m": {"type": "number"},
                "proposed_pump_eff_pct": {"type": "number"},
                "proposed_gear_eff_pct": {"type": "number"},
                "proposed_motor_kw": {"type": "number"},
                "proposed_npshr_m": {"type": "number"},
                "minor_loss_fittings": {"type": "array", "items": {"type": "string"}},
                "pipe_segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_name": {"type": "string"},
                            "carries_single_pump_flow": {"type": "boolean"},
                            "diameter_m": {"type": "number"},
                            "area_m2": {"type": "number"},
                            "length_m": {"type": "number"},
                            "manning_n": {"type": "number"},
                            "darcy_f": {"type": "number"},
                            "minor_loss_k_sum": {"type": "number"},
                            "fixed_minor_loss_m": {"type": "number"}
                        },
                        "required": ["segment_name"]
                    }
                },
                "pump_curve_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "q_single_m3s": {"type": "number"},
                            "head_m": {"type": "number"},
                            "eff_pct": {"type": "number"}
                        },
                        "required": ["q_single_m3s", "head_m"]
                    }
                },
                "pump_curve_confidence": {
                    "type": "string",
                    "enum": ["high_multi_point", "single_duty_point_only", "uncertain_from_pdf_or_blurry", "no_pump_data"]
                },
                "ai_curve_advice_th": {"type": "string"}
            },
            "required": ["pump_curve_confidence"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, ตารางคำนวณ, รูปภาพกราฟ หรือข้อความจากหน้างาน และสกัดพารามิเตอร์ทางชลศาสตร์ออกมา
กฎเหล็ก:
1. สื่อสารเป็นภาษาไทยเสมอ ห้ามเดาตัวเลขใดๆ ที่ผู้ใช้งานไม่ได้ระบุ หากไม่มีให้ใส่ค่าเป็น null
2. หากผู้ใช้งานระบุค่า Minor Loss รวม (เช่น รวม k = 13.5) ให้ใส่ลงในทั้ง total_minor_k, ใน minor_loss_k_sum ของ pipe_segments และอธิบายใน minor_loss_fittings
3. หากผู้ใช้งานให้ตารางพิกัดกราฟปั๊ม (pump_curve_points) มาหลายจุด ให้ดึงทุกจุดใส่ใน pump_curve_points และให้ดึงค่า Head และ Efficiency ณ จุดออกแบบมาใส่ใน proposed_pump_head_m และ proposed_pump_eff_pct ด้วย
"""

# ==========================================
# 6. ส่วนแสดงผล STEP 1: นำเข้าและสกัดข้อมูลทางวิศวกรรม
# ==========================================
render_section_header(
    1,
    "การนำเข้าและสกัดพารามิเตอร์ทางวิศวกรรม (Engineering Data Acquisition & AI Parsing)",
    "รองรับเอกสารขออนุมัติวัสดุ (Submittal PDF), ตารางรายการคำนวณ (Excel/CSV), ภาพถ่ายกราฟสมรรถนะปั๊ม (Vision AI) หรือข้อความระบุสเปคโดยตรง"
)

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "📂 อัปโหลดเอกสารโครงการ / ตารางคำนวณ / รูปภาพกราฟ Pump Curve",
        accept_multiple_files=True,
        type=['pdf', 'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'txt']
    )
    with st.expander("📸 เปิดกล้องถ่ายภาพเอกสารหน้างานหรือกราฟแคตตาล็อก (Camera Capture)", expanded=False):
        camera_photo = st.camera_input("ถ่ายภาพตารางสเปคหรือกราฟเครื่องสูบน้ำ")

with col2:
    manual_input = st.text_area(
        "✍️ ระบุรายละเอียดสเปคเครื่องสูบน้ำ แนวท่อส่งน้ำ หรือพิกัดกราฟ (Engineering Specification Input):",
        height=165,
        placeholder="ตัวอย่าง: โครงการอุโมงค์ระบายน้ำ ต้องการสูบน้ำรวม Q = 60 m3/s (เดินขนาน 6 เครื่อง) Static Head = 5.50 m ท่อคอนกรีต Dia 5.0 m ยาว 8,435 m (n = 0.014, ΣK = 13.5)..."
    )

if st.button("⚡ ประมวลผลเอกสารและสกัดพารามิเตอร์วิศวกรรม (Execute AI Parameter Extraction)", type="primary", use_container_width=True):
    if uploaded_files or camera_photo or manual_input:
        with st.spinner("🔄 ระบบ AI กำลังวิเคราะห์เอกสาร สกัดพารามิเตอร์ชลศาสตร์ และตรวจสอบความครบถ้วนของข้อมูล..."):
            text_content = manual_input
            image_base64_list = []

            for file in (uploaded_files or []):
                if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_base64_list.append(encode_image(file))
                else:
                    text_content += f"\n--- ข้อมูลจากไฟล์ {file.name} ---\n"
                    text_content += extract_text_from_file(file)

            if camera_photo:
                image_base64_list.append(encode_image(camera_photo))

            user_content = [{"type": "text", "text": f"กรุณาสกัดพารามิเตอร์ชลศาสตร์และพิกัดกราฟปั๊มจากข้อมูลต่อไปนี้:\n\n{text_content}"}]
            for img_b64 in image_base64_list:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                tools=extraction_tool,
                tool_choice={"type": "function", "function": {"name": "extract_hydraulic_parameters"}}
            )

            tool_call = response.choices[0].message.tool_calls[0]
            extracted = json.loads(tool_call.function.arguments)
            extracted = post_process_extracted_data(extracted)

            st.session_state['extracted_params'] = extracted
            if 'custom_pump_points_df' in st.session_state:
                del st.session_state['custom_pump_points_df']
            if 'official_ai_memo' in st.session_state:
                del st.session_state['official_ai_memo']
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์เอกสาร ถ่ายภาพ หรือระบุรายละเอียดพารามิเตอร์ก่อนกดปุ่มประมวลผล")

# ==========================================
# แสดงผลการ์ดสรุปพารามิเตอร์ & แถบสถานะ WORKFLOW
# ==========================================
if 'extracted_params' in st.session_state:
    raw_data = st.session_state['extracted_params']

    step2_ok, missing_s2 = check_step2_completeness(raw_data)
    step3_ok, missing_s3 = check_step3_completeness(raw_data)
    step4_ok, missing_s4 = check_step4_completeness(raw_data)

    # แสดงแถบ Workflow Progress Tracker 5 ขั้นตอน
    st.markdown("#### 🧭 สถานะกระบวนการตรวจสอบทางวิศวกรรม (Engineering Verification Workflow Status)")
    st_c1, st_c2, st_c3, st_c4, st_c5 = st.columns(5)
    st_c1.markdown('<div class="step-pill active">1️⃣ สกัดพารามิเตอร์<br><small>✅ เสร็จสมบูรณ์</small></div>', unsafe_allow_html=True)
    st_c2.markdown(f'<div class="step-pill {"active" if step2_ok else "locked"}">2️⃣ คำนวณ System Curve<br><small>{"✅ ข้อมูลท่อครบ" if step2_ok else "🔒 รอข้อมูลท่อ/Override"}</small></div>', unsafe_allow_html=True)
    st_c3.markdown(f'<div class="step-pill {"active" if (step2_ok and step3_ok) else "locked"}">3️⃣ เทียบกราฟปั๊ม (Q-H)<br><small>{"✅ พบข้อมูลปั๊ม" if step3_ok else "🔒 รอพิกัดปั๊ม/Override"}</small></div>', unsafe_allow_html=True)
    st_c4.markdown(f'<div class="step-pill {"active" if (step2_ok and step3_ok and step4_ok) else "locked"}">4️⃣ ประเมินมอเตอร์ & ท่อ<br><small>{"✅ ข้อมูลกลครบ" if step4_ok else "🔒 รอขนาดมอเตอร์/Override"}</small></div>', unsafe_allow_html=True)
    st_c5.markdown(f'<div class="step-pill {"active" if (step2_ok and step3_ok and step4_ok) else "locked"}">5️⃣ สรุปผล Integrated<br><small>{"✅ พร้อมออกรายงาน" if (step2_ok and step3_ok and step4_ok) else "⏳ รอผลส่วนที่ 4"}</small></div>', unsafe_allow_html=True)

    st.markdown("#### 📋 สรุปพารามิเตอร์ที่ระบบตรวจพบจากข้อมูลนำเข้า (Extracted Baseline Parameters)")

    c1, c2, c3 = st.columns(3)
    q_display = raw_data.get('flow_rate_original_text') or format_metric(raw_data.get('flow_rate_q_m3s'), "m³/s")
    c1.metric("อัตราการไหลออกแบบรวม (Design Flow, Q)", q_display)
    c2.metric("ระยะยกน้ำสถิต (Static Head, Hst)", format_metric(raw_data.get('static_head_m'), "m"))
    c3.metric("เฮดเครื่องสูบน้ำที่เสนอ (Proposed Duty Head)", format_metric(raw_data.get('proposed_pump_head_m'), "m"))

    c4, c5, c6 = st.columns(3)
    c4.metric("ความยาวท่อส่งน้ำหลัก (Conduit Length, L)", format_metric(raw_data.get('main_pipe_length_m'), "m"))
    c5.metric("ขนาดเส้นผ่านศูนย์กลางท่อ (Diameter, D)", format_metric(raw_data.get('main_pipe_diameter_m'), "m"))
    c6.metric("วัสดุท่อ / ความขรุขระ (Roughness Factor)", format_metric(raw_data.get('pipe_material_or_roughness'), ""))

    with st.expander("🔧 รายละเอียดอุปกรณ์ประกอบในระบบท่อ (Minor Loss Fittings) และข้อมูลระบบขับเคลื่อน", expanded=True):
        sys_t = format_metric(raw_data.get('system_type'), '')
        n_pumps = format_metric(raw_data.get('num_parallel_pumps'), 'เครื่อง', 0)
        mot_kw = format_metric(raw_data.get('proposed_motor_kw'), 'kW', 0)
        p_eff = format_metric(raw_data.get('proposed_pump_eff_pct'), '%', 1)
        st.write(f"**ลักษณะระบบ:** {sys_t} | **จำนวนเครื่องสูบน้ำเดินขนาน:** {n_pumps} | **ประสิทธิภาพปั๊ม:** {p_eff} | **พิกัดมอเตอร์ที่เสนอ:** {mot_kw}")

        fittings = raw_data.get('minor_loss_fittings') or []
        if fittings:
            for item in fittings:
                st.write(f"- {item}")
        else:
            st.write("- ไม่พบรายละเอียดอุปกรณ์ประกอบย่อยในเอกสาร")

    working_data = copy.deepcopy(raw_data)

    # ==========================================
    # 7. ส่วนที่ 2: คำนวณ HEAD LOSS & SYSTEM CURVE
    # ==========================================
    render_section_header(
        2,
        "การวิเคราะห์ความสูญเสียทางชลศาสตร์และสมการความต้านทานระบบ (Hydraulic Loss & System Curve)",
        "คำนวณความเสียดทานท่อตรง (Major Loss) และการสูญเสียตามอุปกรณ์ (Minor Loss) เพื่อสร้างสมการพาราโบลา H = KQ² + Hst"
    )

    allow_step_2 = False
    if step2_ok:
        st.success("✅ **การตรวจสอบความครบถ้วนส่วนที่ 2 (Data Integrity Verified):** ข้อมูลระบบท่อส่งน้ำครบถ้วนสมบูรณ์ คำนวณสมการ System Curve อัตโนมัติ")
        allow_step_2 = True
        edit_pipe_manual = st.toggle("🛠️ เปิดโหมดปรับแต่งพารามิเตอร์ระบบท่อเพิ่มเติม (Customize Conduit Parameters)", value=False)
    else:
        st.warning("🔒 **ระงับการประมวลผลส่วนที่ 2 ชั่วคราว (Data Integrity Gate Locked):** ข้อมูลระบบท่อไม่เพียงพอสำหรับการคำนวณชลศาสตร์")
        for m in missing_s2:
            st.error(f"❌ ข้อมูลที่ต้องระบุเพิ่มเติม: {m}")
        edit_pipe_manual = st.toggle("🔓 **เปิดโหมดกำหนดพารามิเตอร์ระบบท่อจำลองด้วยตนเอง (Enable Hydraulic Parameter Override - Part 2)**", value=False)
        if edit_pipe_manual:
            allow_step_2 = True

    if edit_pipe_manual:
        st.info("✍️ **แผงกำหนดค่าพารามิเตอร์ระบบท่อส่งน้ำ (Engineering Conduit Parameter Override):**")
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            sim_q = st.number_input("1. อัตราการไหลรวม Q (m³/s):", min_value=0.01, value=float(raw_data.get("flow_rate_q_m3s") or 60.0), step=1.0)
            sim_hst = st.number_input("2. ระยะยกน้ำสถิต Static Head (m):", value=float(raw_data.get("static_head_m") if raw_data.get("static_head_m") is not None else 5.50), step=0.5)
        with sb2:
            sim_d = st.number_input("3. ขนาดเส้นผ่านศูนย์กลางท่อ D (m):", min_value=0.05, value=float(raw_data.get("main_pipe_diameter_m") or 5.0), step=0.1)
            sim_l = st.number_input("4. ความยาวท่อ/อุโมงค์ L (m):", min_value=1.0, value=float(raw_data.get("main_pipe_length_m") or 8435.0), step=50.0)
        with sb3:
            sim_n = st.number_input("5. สัมประสิทธิ์ความขรุขระ Manning's n:", min_value=0.008, max_value=0.050, value=float(raw_data.get("manning_n") or 0.014), step=0.001, format="%.4f")
            sim_k = st.number_input("6. ผลรวมสัมประสิทธิ์การสูญเสียรอง (ΣK):", min_value=0.0, value=float(raw_data.get("total_minor_k") or 13.5), step=0.5)

        working_data["flow_rate_q_m3s"] = sim_q
        working_data["static_head_m"] = sim_hst
        working_data["main_pipe_diameter_m"] = sim_d
        working_data["main_pipe_length_m"] = sim_l
        working_data["manning_n"] = sim_n
        working_data["total_minor_k"] = sim_k
        working_data["pipe_segments"] = [{
            "segment_name": "ท่อส่งน้ำหลัก / อุโมงค์ระบายน้ำ (Custom Engineering Input)",
            "carries_single_pump_flow": False,
            "diameter_m": sim_d,
            "length_m": sim_l,
            "manning_n": sim_n,
            "darcy_f": float(raw_data.get("darcy_f") or 0.015),
            "minor_loss_k_sum": sim_k,
            "fixed_minor_loss_m": 0.0
        }]

    if allow_step_2:
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            formula_choice = st.selectbox(
                "📐 เลือกสมการคำนวณความเสียดทานหลัก (Friction Loss Equation):",
                ["Manning's Equation (มาตรฐานงานอุโมงค์และท่อระบายน้ำขนาดใหญ่)", "Darcy-Weisbach Equation (มาตรฐานงานท่อรับแรงดัน)"]
            )
        with cfg2:
            flow_unit_plot = st.selectbox(
                "📊 หน่วยแสดงผลอัตราการไหลบนกราฟ (Flow Rate Unit):",
                ["m³/s (ลูกบาศก์เมตรต่อวินาที)", "m³/h (ลูกบาศก์เมตรต่อชั่วโมง)"]
            )

        selected_method = "Darcy" if "Darcy" in formula_choice else "Manning"
        calc_res = run_hydraulic_engine(working_data, formula_override=selected_method)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ระยะยกน้ำสถิต (Static Head, Hst)", f"{calc_res['h_static_m']:.2f} m")
        m2.metric("การสูญเสียหลักในท่อตรง (Major Loss)", f"{calc_res['total_major_loss_m']:.3f} m")
        m3.metric("การสูญเสียรองตามอุปกรณ์ (Minor Loss)", f"{calc_res['total_minor_loss_m']:.3f} m")
        m4.metric("เฮดความต้องการรวมของระบบ (Total System Head)", f"{calc_res['total_system_head_m']:.3f} m")

        use_m3h = "m³/h" in flow_unit_plot
        scale_factor = 3600.0 if use_m3h else 1.0
        k_display = calc_res['k_system_m3s'] / (scale_factor ** 2)
        q_label = "Q (m³/h)" if use_m3h else "Q (m³/s)"
        q_design = calc_res['q_total_m3h'] if use_m3h else calc_res['q_total_m3s']

        st.info(
            f"🧮 **สมการพฤติกรรมความต้านทานของระบบ (System Curve Equation):**  "
            f"`H = ({k_display:.6e}) · Q² + {calc_res['h_static_m']:.2f}`  "
            f"*(โดยมี Dynamic Head Loss รวม = `{calc_res['total_dynamic_loss_m']:.3f} m` ณ อัตราการไหลออกแบบ `{q_design:,.1f} {q_label.split()[1].strip('()')}`)*"
        )

        with st.expander("📑 ตารางแจกแจงการคำนวณความเร็วน้ำ (Velocity) และความสูญเสียเฮดรายช่วงท่อ (Hydraulic Loss Schedule)", expanded=True):
            st.dataframe(calc_res["breakdown_df"], use_container_width=True, hide_index=True)

        # ==========================================
        # 8. ส่วนที่ 3: เปรียบเทียบกับ PUMP CURVE
        # ==========================================
        render_section_header(
            3,
            "การวิเคราะห์จุดตัดการทำงานของเครื่องสูบน้ำและระบบท่อ (Pump vs. System Curve & Operating Point)",
            "เปรียบเทียบสมรรถนะเครื่องสูบน้ำของผู้รับจ้าง (Exact Polynomial Fit หรือ Synthesized Curve) กับสมการความต้านทานของระบบ"
        )

        allow_step_3 = False
        if step3_ok:
            st.success("✅ **การตรวจสอบความครบถ้วนส่วนที่ 3 (Pump Data Verified):** พบข้อมูลสเปคเฮดปั๊ม/พิกัดกราฟปั๊ม พร้อมคำนวณจุดตัด Operating Point")
            allow_step_3 = True
        else:
            st.warning("🔒 **ระงับการประมวลผลส่วนที่ 3 ชั่วคราว (Data Integrity Gate Locked):** ไม่พบข้อมูลเฮดเครื่องสูบน้ำที่เสนอ หรือตารางพิกัดกราฟปั๊ม")
            for m3 in missing_s3:
                st.error(f"❌ ข้อมูลที่ต้องระบุเพิ่มเติม: {m3}")
            force_step_3 = st.toggle("🔓 **เปิดโหมดกำหนดพิกัดกราฟสมรรถนะเครื่องสูบน้ำด้วยตนเอง (Enable Pump Curve Override - Part 3)**", value=False)
            if force_step_3:
                allow_step_3 = True

        if allow_step_3:
            num_pumps_default = int(working_data.get("num_parallel_pumps") or 1)
            q_unit_default = calc_res["q_total_m3s"] / max(num_pumps_default, 1)
            h_duty_default = float(working_data.get("proposed_pump_head_m") or calc_res["total_system_head_m"] or 23.0)
            eff_default = float(working_data.get("proposed_pump_eff_pct") or 85.0)

            if 'custom_pump_points_df' not in st.session_state:
                ai_points = working_data.get("pump_curve_points") or []
                if len(ai_points) >= 1:
                    init_rows = [
                        {
                            "Q_single_m3s": float(pt.get("q_single_m3s", 0.0)),
                            "Head_m": float(pt.get("head_m", 0.0)),
                            "Eff_pct": float(pt.get("eff_pct") if pt.get("eff_pct") is not None else eff_default)
                        }
                        for pt in ai_points
                    ]
                else:
                    init_rows = [{"Q_single_m3s": round(q_unit_default, 2), "Head_m": round(h_duty_default, 2), "Eff_pct": round(eff_default, 1)}]
                st.session_state['custom_pump_points_df'] = pd.DataFrame(init_rows)

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown("##### ⚙️ การจัดรูปแบบการเดินเครื่อง (Pump Configuration)")
                operation_mode = st.radio(
                    "เลือกรูปแบบการเดินเครื่องสูบน้ำ:",
                    ["เดินเครื่องสูบน้ำแบบขนาน (Parallel Operation)", "เดินเครื่องสูบน้ำเดี่ยว (Single Unit Operation)"],
                    index=0 if num_pumps_default > 1 else 1
                )
                num_active_pumps = st.number_input(
                    "จำนวนเครื่องสูบน้ำที่เดินขนานพร้อมกัน (เครื่อง):",
                    min_value=1, max_value=20, value=max(num_pumps_default, 2), step=1
                ) if "ขนาน" in operation_mode else 1

                if st.button("✨ โหลดชุดพิกัดมาตรฐานผู้ผลิต 5 จุด (Load Reference 5-Point Curve)", use_container_width=True):
                    st.session_state['custom_pump_points_df'] = pd.DataFrame([
                        {"Q_single_m3s": 0.0, "Head_m": round(h_duty_default * 1.34, 2), "Eff_pct": 0.0},
                        {"Q_single_m3s": round(q_unit_default * 0.6, 2), "Head_m": round(h_duty_default * 1.23, 2), "Eff_pct": 74.0},
                        {"Q_single_m3s": round(q_unit_default * 0.8, 2), "Head_m": round(h_duty_default * 1.15, 2), "Eff_pct": 82.0},
                        {"Q_single_m3s": round(q_unit_default * 1.0, 2), "Head_m": round(h_duty_default * 1.00, 2), "Eff_pct": eff_default},
                        {"Q_single_m3s": round(q_unit_default * 1.2, 2), "Head_m": round(h_duty_default * 0.79, 2), "Eff_pct": 76.0},
                    ])
                    st.rerun()

            with p_col2:
                st.markdown("##### 📝 ตารางพิกัดสมรรถนะเครื่องสูบน้ำต่อ 1 เครื่อง (ระบุ ≥ 3 จุด เพื่อสร้างสมการ Exact Polynomial Fit)")
                edited_pump_df = st.data_editor(
                    st.session_state['custom_pump_points_df'],
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Q_single_m3s": st.column_config.NumberColumn("อัตราไหลต่อเครื่อง Q (m³/s)", min_value=0.0, step=0.5, format="%.2f"),
                        "Head_m": st.column_config.NumberColumn("เฮดปั๊ม H (m)", min_value=0.0, step=0.5, format="%.2f"),
                        "Eff_pct": st.column_config.NumberColumn("ประสิทธิภาพ ηp (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
                    }
                )

            pump_eval = evaluate_pump_vs_system(
                df_points=edited_pump_df,
                num_pumps=num_active_pumps,
                k_sys_m3s=calc_res["k_system_m3s"],
                h_static_m=calc_res["h_static_m"],
                q_target_total_m3s=calc_res["q_total_m3s"],
                use_m3h=use_m3h
            )

            if pump_eval["is_synthesized"]:
                st.warning(
                    "⚠️ **ประกาศแจ้งเตือนทางวิศวกรรม (Synthesized Pump Curve Active):** เนื่องจากมีข้อมูลพิกัดปั๊มน้อยกว่า 3 จุด "
                    "ระบบจึงจำลองเส้นโค้งสมรรถนะตามทฤษฎีปั๊ม ($H_{shutoff} \\approx 1.30 H_{duty}$) "
                    "เพื่อความแม่นยำในการตรวจรับงานจริง แนะนำให้อัปโหลดภาพกราฟผู้ผลิต หรือระบุพิกัดอย่างน้อย 3 จุด (Shutoff, Design, Runout) ลงในตารางด้านบน"
                )
            else:
                st.success(
                    f"✅ **Exact Model Curve Active:** สร้างสมการโพลีโนเมียลกำลังสอง (2nd-Degree Polynomial Regression) จากพิกัดจริงจำนวน {len(pump_eval['clean_points_df'])} จุดเรียบร้อยแล้ว"
                )

            op1, op2, op3, op4 = st.columns(4)
            if pump_eval["q_op_plot"] is not None:
                q_op_disp = f"{pump_eval['q_op_plot']:,.2f} {'m³/h' if use_m3h else 'm³/s'}"
                h_op_disp = f"{pump_eval['h_op_m']:,.2f} m"
                eff_op_disp = f"{pump_eval['eff_op_pct']:.1f} %" if pump_eval["eff_op_pct"] is not None else "N/A"
                q_diff_m3s = pump_eval["q_op_m3s"] - calc_res["q_total_m3s"]
                flow_status = "✅ เพียงพอ (PASS)" if q_diff_m3s >= -0.05 else f"❌ ขาดไป {abs(q_diff_m3s):.2f} m³/s"
            else:
                q_op_disp, h_op_disp, eff_op_disp, flow_status = "ไม่ตัดกัน", "ไม่ตัดกัน", "N/A", "❌ เฮดไม่พอส่งน้ำ"

            op1.metric("อัตราการไหล ณ จุดทำงาน (Operating Q)", q_op_disp)
            op2.metric("เฮด ณ จุดทำงานจริง (Operating H)", h_op_disp)
            op3.metric("ประสิทธิภาพ ณ จุดทำงาน (Operating ηp)", eff_op_disp)
            op4.metric("สถานะเทียบเป้าหมายโครงการ", flow_status)

            # สร้างกราฟ Plotly หลัก (สำหรับแสดงทั้งในส่วนที่ 3 และสามารถนำไปแสดงซ้ำในแดชบอร์ดส่วนที่ 5)
            a_p, b_p, c_p = pump_eval["poly_coeffs"]
            q_max_candidates = [q_design * 1.35]
            if pump_eval["q_op_plot"]:
                q_max_candidates.append(pump_eval["q_op_plot"] * 1.25)
            q_max_plot = max(q_max_candidates) if max(q_max_candidates) > 0 else 100.0

            q_vals = np.linspace(0, q_max_plot, 120)
            h_sys_vals = (k_display * (q_vals ** 2)) + calc_res["h_static_m"]
            h_pump_parallel_vals = (a_p * (q_vals ** 2)) + (b_p * q_vals) + c_p
            h_pump_parallel_vals = np.where(h_pump_parallel_vals >= 0, h_pump_parallel_vals, np.nan)

            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=q_vals, y=h_sys_vals, mode='lines',
                name='System Curve (H = KQ² + Hst)',
                line=dict(color='#00B4D8', width=4)
            ))

            pump_curve_name = f"Combined Pump Curve ({num_active_pumps} เครื่อง) - {'[Synthesized]' if pump_eval['is_synthesized'] else '[Exact Fit]'}"
            fig_main.add_trace(go.Scatter(
                x=q_vals, y=h_pump_parallel_vals, mode='lines',
                name=pump_curve_name,
                line=dict(color='#FF006E', width=4, dash='dashdot' if pump_eval['is_synthesized'] else 'solid')
            ))

            if num_active_pumps > 1:
                q_single_vals = q_vals / num_active_pumps
                h_single_vals = (a_p * ((q_single_vals * num_active_pumps) ** 2)) + (b_p * (q_single_vals * num_active_pumps)) + c_p
                h_single_vals = np.where(h_single_vals >= 0, h_single_vals, np.nan)
                fig_main.add_trace(go.Scatter(
                    x=q_single_vals, y=h_single_vals, mode='lines',
                    name='Single Pump Curve (สมรรถนะปั๊มเดี่ยว 1 เครื่อง)',
                    line=dict(color='#FB5607', width=2, dash='dot')
                ))

            clean_pts = pump_eval["clean_points_df"]
            if not clean_pts.empty:
                fig_main.add_trace(go.Scatter(
                    x=clean_pts["Q_single_m3s"] * num_active_pumps * scale_factor,
                    y=clean_pts["Head_m"], mode='markers',
                    name='พิกัดสมรรถนะที่ระบุ (Submittal Data Points)',
                    marker=dict(color='#FFBE0B', size=9, symbol='diamond')
                ))

            fig_main.add_trace(go.Scatter(
                x=[q_design], y=[calc_res["total_system_head_m"]],
                mode='markers+text',
                name='จุดความต้องการโครงการ (Project Design Duty)',
                marker=dict(color='#FFD166', size=12, symbol='circle-open', line=dict(width=3)),
                text=[f"Design Target: ({q_design:,.1f}, {calc_res['total_system_head_m']:.2f} m)"],
                textposition="bottom right"
            ))

            if pump_eval["q_op_plot"] is not None and pump_eval["h_op_m"] is not None:
                is_pass_op = pump_eval["q_op_m3s"] >= (calc_res["q_total_m3s"] - 0.05)
                op_color = "#06D6A0" if is_pass_op else "#EF476F"
                fig_main.add_trace(go.Scatter(
                    x=[pump_eval["q_op_plot"]], y=[pump_eval["h_op_m"]],
                    mode='markers+text',
                    name='จุดตัดการทำงานจริง (Actual Operating Point)',
                    marker=dict(color=op_color, size=16, symbol='star', line=dict(color='white', width=1.5)),
                    text=[f"Operating Point: ({pump_eval['q_op_plot']:,.1f}, {pump_eval['h_op_m']:.2f} m)"],
                    textposition="top left"
                ))

            fig_main.add_hline(
                y=calc_res["h_static_m"], line_dash="dash", line_color="#90E0EF",
                annotation_text=f"Static Head (Hst = {calc_res['h_static_m']:.2f} m)",
                annotation_position="bottom right"
            )

            fig_main.update_layout(
                title=f"📈 กราฟวิเคราะห์จุดตัดการทำงาน System Curve vs. Pump Curve ({pump_eval['curve_type_label']})",
                xaxis_title=f"อัตราการไหลรวม {q_label}",
                yaxis_title="เฮดรวม Total Dynamic Head (m)",
                template="plotly_dark",
                hovermode="x unified",
                height=540,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_main, use_container_width=True, key="chart_step3")

            # ==========================================
            # 9. ส่วนที่ 4: พิจารณาเลือกปั๊ม กำลังมอเตอร์ และความเสี่ยงเชิงวิศวกรรม
            # ==========================================
            render_section_header(
                4,
                "การประเมินความเหมาะสมทางกล พิกัดกำลังมอเตอร์ และความเสี่ยงทางชลศาสตร์ (Mechanical & Risk Evaluation)",
                "ตรวจสอบกำลังเพลา (Shaft Power), พิกัดมอเตอร์เผื่อ 105% Margin, ความเร็วน้ำในท่อ (Erosion/Siltation) และเกณฑ์ป้องกันโพรงอากาศ (NPSH)"
            )

            est_eff_pump = float(pump_eval.get("eff_op_pct") or working_data.get("proposed_pump_eff_pct") or 85.0)
            if est_eff_pump <= 0:
                est_eff_pump = 85.0
            est_eff_gear = float(working_data.get("proposed_gear_eff_pct") or 98.0)
            est_q_unit = max((pump_eval.get("q_op_m3s") or 0.0) / max(num_active_pumps, 1), q_unit_default)
            est_h = max(pump_eval.get("h_op_m") or 0.0, h_duty_default, calc_res["total_system_head_m"])
            est_p_gear = ((1000.0 * 9.81 * est_q_unit * est_h) / 1000.0) / (est_eff_pump / 100.0) / (est_eff_gear / 100.0)
            est_min_motor = est_p_gear * 1.05
            est_rec_motor = select_standard_motor_kw(est_min_motor)

            allow_step_4 = False
            if step4_ok:
                st.success("✅ **การตรวจสอบความครบถ้วนส่วนที่ 4 (Mechanical Data Verified):** ข้อมูลพิกัดกำลังมอเตอร์และพารามิเตอร์ทางกลครบถ้วนสมบูรณ์")
                allow_step_4 = True
                edit_step4_manual = st.toggle("🛠️ เปิดโหมดปรับแต่งพิกัดกำลังมอเตอร์และพารามิเตอร์ทางกล (Customize Mechanical Parameters)", value=False)
            else:
                st.warning("🔒 **ระงับการประมวลผลส่วนที่ 4–5 ชั่วคราว (Data Integrity Gate Locked):** ขาดข้อมูลทางกลสำหรับประเมินพิกัดมอเตอร์และความเสี่ยง")
                for m4_item in missing_s4:
                    st.error(f"❌ ข้อมูลที่ต้องระบุเพิ่มเติมสำหรับส่วนที่ 4: {m4_item}")

                st.info(
                    f"💡 **ข้อมูลช่วยตัดสินใจเบื้องต้นจากจุดตัดกราฟ (Pre-Calculated Reference):** ณ จุดทำงานจริง ระบบต้องการกำลังเพลาเกียร์ประมาณ **`{est_p_gear:,.1f} kW/เครื่อง`** "
                    f"เมื่อคูณเผื่อ Safety Margin 105% จะต้องการพิกัดมอเตอร์ขั้นต่ำ **`{est_min_motor:,.1f} kW/เครื่อง`** (พิกัดมอเตอร์มาตรฐานที่แนะนำคือ **`{est_rec_motor:,.0f} kW`**)"
                )

                edit_step4_manual = st.toggle(
                    "🔓 **เปิดโหมดกำหนดพิกัดกำลังมอเตอร์ (kW) และพารามิเตอร์ทางกลด้วยตนเอง (Enable Mechanical Parameter Override - Part 4 & 5)**",
                    value=False
                )
                if edit_step4_manual:
                    allow_step_4 = True

            default_motor_val = float(working_data.get("proposed_motor_kw") or est_rec_motor)
            default_pump_eff = float(pump_eval.get("eff_op_pct") or working_data.get("proposed_pump_eff_pct") or 85.0)
            default_gear_eff = float(working_data.get("proposed_gear_eff_pct") or 98.0)
            default_npshr = float(working_data.get("proposed_npshr_m") or 4.5)
            margin_input = 105.0

            if edit_step4_manual:
                st.info("✍️ **แผงกำหนดค่าพิกัดกำลังมอเตอร์และพารามิเตอร์ทางกล (Custom Mechanical & NPSH Inputs):**")
                g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
                with g_col1:
                    motor_kw_input = st.number_input("1. พิกัดมอเตอร์ที่เสนอ (kW):", min_value=1.0, max_value=20000.0, value=default_motor_val, step=50.0)
                with g_col2:
                    pump_eff_input = st.number_input("2. ประสิทธิภาพปั๊ม ηp (%):", min_value=10.0, max_value=98.0, value=min(max(default_pump_eff, 10.0), 98.0), step=1.0)
                with g_col3:
                    gear_eff_input = st.number_input("3. ประสิทธิภาพเกียร์ ηg (%):", min_value=50.0, max_value=100.0, value=default_gear_eff, step=0.5)
                with g_col4:
                    margin_input = st.number_input("4. Safety Margin มอเตอร์ (%):", min_value=100.0, max_value=150.0, value=105.0, step=1.0)
                with g_col5:
                    npshr_input = st.number_input("5. ค่า NPSHr ของปั๊ม (m):", min_value=0.5, max_value=25.0, value=default_npshr, step=0.5)
            else:
                motor_kw_input = working_data.get("proposed_motor_kw")
                pump_eff_input = default_pump_eff
                gear_eff_input = default_gear_eff
                npshr_input = default_npshr

            if allow_step_4:
                phase4_res = evaluate_phase4_and_5(
                    working_data=working_data,
                    calc_res=calc_res,
                    pump_eval=pump_eval,
                    num_active_pumps=num_active_pumps,
                    pump_eff_pct=pump_eff_input,
                    gear_eff_pct=gear_eff_input,
                    safety_margin_pct=margin_input,
                    npshr_m=npshr_input,
                    user_motor_kw=motor_kw_input,
                    is_force_unlocked=(not step4_ok and edit_step4_manual),
                    missing_step4_list=missing_s4
                )

                p4_c1, p4_c2, p4_c3, p4_c4 = st.columns(4)
                p4_c1.metric("กำลังเพลาเครื่องสูบน้ำ (Pump Shaft Power)", f"{phase4_res['p_pump_shaft_kw']:,.1f} kW/เครื่อง")
                p4_c2.metric("กำลังเพลาชุดเกียร์ (Gear Input Power)", f"{phase4_res['p_gear_shaft_kw']:,.1f} kW/เครื่อง")
                p4_c3.metric(f"พิกัดมอเตอร์ขั้นต่ำ (เผื่อ {margin_input:.0f}%)", f"{phase4_res['p_min_motor_kw']:,.1f} kW/เครื่อง")
                p4_c4.metric("ผลการประเมินพิกัดมอเตอร์", phase4_res["power_status"])

                with st.container(border=True):
                    st.markdown(f"**1. การวิเคราะห์พฤติกรรมกราฟและโหมดการเดินเครื่อง ({phase4_res['mode_title']}):**")
                    st.write(phase4_res["graph_narrative"])

                    st.markdown("---")
                    st.markdown(f"**2. ความเพียงพอของอัตราการไหล (Q) และเฮด (H) — [{phase4_res['qh_status']}]:**")
                    if phase4_res["qh_pass"]:
                        st.success(phase4_res["qh_narrative"])
                    else:
                        st.error(phase4_res["qh_narrative"])

                    st.markdown("---")
                    st.markdown(f"**3. ประสิทธิภาพ กำลังเพลา และความเหมาะสมของขนาดมอเตอร์ (Efficiency / Power / Margin) — [{phase4_res['power_status']}]:**")
                    if phase4_res["proposed_motor_kw"] is None:
                        st.info(phase4_res["power_narrative"])
                    elif phase4_res["power_pass"]:
                        st.success(phase4_res["power_narrative"])
                    else:
                        st.error(phase4_res["power_narrative"])

                    st.markdown("---")
                    st.markdown(f"**4. ความเหมาะสมของขนาดท่อ แนวท่อ และความเร็วน้ำ (Conduit Sizing & Velocity Assessment) — [{phase4_res['pipe_status']}]:**")
                    st.write(phase4_res["pipe_narrative"])
                    for w in phase4_res["vel_warnings"]:
                        st.warning(w)
                    with st.expander("🔍 ดูรายละเอียดความเร็วน้ำรายช่วงท่อที่ผ่านเกณฑ์มาตรฐาน", expanded=False):
                        for d_item in phase4_res["vel_details"]:
                            st.write(d_item)

                    st.markdown("---")
                    st.markdown("**5. การแจ้งเตือนจุดเสี่ยง ผลกระทบทางชลศาสตร์ และเกณฑ์ป้องกันโพรงอากาศ (NPSH & Risk Flags):**")
                    for flag in phase4_res["risk_flags"]:
                        if flag.startswith("🚨"):
                            st.error(flag)
                        elif flag.startswith("⚠️"):
                            st.warning(flag)
                        elif flag.startswith("ℹ️"):
                            st.info(flag)
                        else:
                            st.success(flag)

                # ==========================================
                # 10. ส่วนที่ 5: แสดงผลแบบ INTEGRATED EXECUTIVE DASHBOARD (PHASE 5)
                # ==========================================
                render_section_header(
                    5,
                    "สรุปผลการพิจารณาตรวจรับงานเครื่องกลแบบบูรณาการ (Integrated Engineering Verdict & Inspection Report)",
                    "สรุปความเห็นวิศวกรผู้ควบคุมงาน (เหมาะสม / ไม่เหมาะสม) ตารางเปรียบเทียบพารามิเตอร์หลัก กราฟประกอบ และข้อเสนอแนะการปรับปรุงระบบ"
                )

                # 5.1 ป้ายฟันธงสถานะขนาดใหญ่ (Executive Verdict Banner)
                v_code = phase4_res["verdict_code"]
                box_cls = "verdict-box-pass" if v_code == "APPROVED" else ("verdict-box-warn" if v_code == "CONDITIONAL" else "verdict-box-fail")
                badge_cls = "verdict-badge-pass" if v_code == "APPROVED" else ("verdict-badge-warn" if v_code == "CONDITIONAL" else "verdict-badge-fail")

                st.markdown(f"""
                <div class="{box_cls}">
                    <div class="{badge_cls}">{phase4_res['verdict_badge']}</div>
                    <h2 style="margin: 0 0 8px 0; color: #FFFFFF; font-size: 1.65rem;">{phase4_res['verdict_title']}</h2>
                    <p style="margin: 0; color: #E2E8F0; font-size: 1.02rem;">{phase4_res['verdict_sub']}</p>
                </div>
                """, unsafe_allow_html=True)

                # 5.2 การ์ดสรุปค่าที่สำคัญทั้ง 5 ตัวตาม Flow Chart ส่วนที่ 5 (Q, Head, Head Loss, Efficiency, Operating Point)
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("1. อัตราการไหลเป้าหมาย (Q)", f"{calc_res['q_total_m3s']:,.2f} m³/s")
                k2.metric("2. เฮดรวมของระบบ (System Head)", f"{calc_res['total_system_head_m']:,.2f} m")
                k3.metric("3. ความสูญเสียเฮดรวม (Head Loss)", f"{calc_res['total_dynamic_loss_m']:,.2f} m")
                k4.metric("4. ประสิทธิภาพปั๊ม (Efficiency)", f"{phase4_res['eff_pump_pct']:.1f} %")
                op_summary_str = f"{pump_eval['q_op_m3s']:,.1f} m³/s @ {pump_eval['h_op_m']:.1f} m" if pump_eval["q_op_m3s"] else "ไม่ตัดกัน"
                k5.metric("5. จุดทำงานจริง (Operating Point)", op_summary_str)

                # 5.3 แดชบอร์ดคู่ขนาน: ตารางสรุปผลการตรวจรับ vs. กราฟประกอบแบบ Integrated
                int_col1, int_col2 = st.columns([1.15, 1.0])
                with int_col1:
                    st.markdown("##### 📊 ตารางสรุปผลการตรวจสอบเทียบเกณฑ์มาตรฐานโครงการ (QA/QC Compliance Matrix)")
                    st.dataframe(phase4_res["summary_table_df"], use_container_width=True, hide_index=True)

                    st.markdown("##### 🛠️ ข้อเสนอแนะการปรับปรุงระบบและแนวทางปฏิบัติสำหรับผู้ควบคุมงาน (Engineering Recommendations)")
                    with st.container(border=True):
                        for idx_r, rec_text in enumerate(phase4_res["recommendations"], start=1):
                            st.markdown(f"**{idx_r}.** {rec_text}")

                with int_col2:
                    st.markdown("##### 📈 กราฟประกอบการพิจารณาตรวจรับงาน (Integrated Visual Verification)")
                    tab_curve, tab_bar = st.tabs(["กราฟจุดตัด System vs. Pump Curve", "กราฟเปรียบเทียบกำลังมอเตอร์และองค์ประกอบเฮด"])
                    with tab_curve:
                        fig_mini = copy.deepcopy(fig_main)
                        fig_mini.update_layout(height=410, title="จุดตัดการทำงานจริง (Operating Point Verification)")
                        st.plotly_chart(fig_mini, use_container_width=True, key="chart_step5_curve")

                    with tab_bar:
                        motor_compare_val = phase4_res["proposed_motor_kw"] or phase4_res["rec_standard_motor_kw"]
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=["กำลังน้ำ (Hydraulic)", "กำลังเพลาปั๊ม (Shaft)", "กำลังเพลาเกียร์ (Gear)", "มอเตอร์ขั้นต่ำ (105%)", "พิกัดมอเตอร์ที่เสนอ/แนะนำ"],
                            y=[
                                phase4_res["p_hydraulic_kw"],
                                phase4_res["p_pump_shaft_kw"],
                                phase4_res["p_gear_shaft_kw"],
                                phase4_res["p_min_motor_kw"],
                                motor_compare_val
                            ],
                            marker_color=["#00B4D8", "#48CAE4", "#F77F00", "#FFD166", "#06D6A0" if phase4_res["power_pass"] else "#EF476F"],
                            text=[
                                f"{phase4_res['p_hydraulic_kw']:,.0f} kW",
                                f"{phase4_res['p_pump_shaft_kw']:,.0f} kW",
                                f"{phase4_res['p_gear_shaft_kw']:,.0f} kW",
                                f"{phase4_res['p_min_motor_kw']:,.0f} kW",
                                f"{motor_compare_val:,.0f} kW"
                            ],
                            textposition="auto"
                        ))
                        fig_bar.update_layout(
                            title="การตรวจสอบลำดับกำลังไฟฟ้าต่อเครื่อง (Power Cascade & Motor Margin Check)",
                            yaxis_title="กำลังไฟฟ้า (kW / เครื่อง)",
                            template="plotly_dark",
                            height=410
                        )
                        st.plotly_chart(fig_bar, use_container_width=True, key="chart_step5_bar")

                # 5.4 เครื่องมือช่วยออกบันทึกข้อความตรวจรับงานเครื่องกลจากผู้รับจ้าง (AI Official Memo Generator)
                st.markdown("---")
                st.markdown("##### 📄 เครื่องมือสร้างบันทึกข้อความตรวจรับงานเครื่องกลสำหรับผู้ควบคุมงาน (Official Submittal Inspection Memo)")
                memo_col1, memo_col2 = st.columns([1, 2])
                with memo_col1:
                    st.write(
                        "สร้างร่างบันทึกข้อความรายงานผลการตรวจสอบสเปคเครื่องสูบน้ำอย่างเป็นทางการ "
                        "เพื่อใช้แนบเป็นความเห็นวิศวกรที่ปรึกษาในการอนุมัติหรือส่งคืนเอกสาร Submittal ให้แก่ผู้รับจ้าง"
                    )
                    if st.button("📝 สร้างบันทึกข้อความตรวจรับงานด้วย AI (Generate Official QA/QC Memo)", type="primary", use_container_width=True):
                        with st.spinner("🤖 AI กำลังเรียบเรียงบันทึกข้อความตรวจรับงานเครื่องกลอย่างเป็นทางการ..."):
                            st.session_state['official_ai_memo'] = generate_ai_official_memo(
                                client, working_data, calc_res, pump_eval, phase4_res
                            )

                with memo_col2:
                    if 'official_ai_memo' in st.session_state:
                        with st.container(border=True):
                            st.markdown(st.session_state['official_ai_memo'])
                        st.download_button(
                            label="📥 ดาวน์โหลดบันทึกข้อความตรวจรับงาน (.txt / Markdown)",
                            data=st.session_state['official_ai_memo'].encode("utf-8"),
                            file_name=f"Submittal_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    else:
                        st.info("💡 กดปุ่ม **'📝 สร้างบันทึกข้อความตรวจรับงานด้วย AI'** ทางด้านซ้าย เพื่อออกรายงานสรุปผลการตรวจรับงานฉบับสมบูรณ์")
