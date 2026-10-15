import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
from openai import OpenAI
import json
import base64
import math
import copy
import plotly.graph_objects as go

# --- ตั้งค่าหน้าจอ (PAGE CONFIG) ---
st.set_page_config(page_title="AI Pump & System Curve Validator", layout="wide")
st.title("🤖 ระบบ AI ตรวจสอบความเหมาะสมเครื่องสูบน้ำและระบบอุโมงค์")
st.caption("จาก Input Parameters → คำนวณ Head Loss & System Curve → เปรียบเทียบ Pump Curve → พิจารณาความเหมาะสมและจุดเสี่ยงเชิงวิศวกรรม")

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================================
# 1. ฟังก์ชันช่วยเหลือ & แปลชื่อตัวแปรเป็นภาษาไทย (HELPERS & TRANSLATOR)
# ==========================================
FIELD_THAI_MAP = {
    "flow_rate_q_m3s": "อัตราการไหลที่ต้องการ (Flow Rate Q)",
    "static_head_m": "ระยะยกน้ำสถิต (Static Head) หรือระดับน้ำต้นทาง-ปลายทาง",
    "main_pipe_length_m": "ความยาวรวมของท่อหรืออุโมงค์ (Pipe/Tunnel Length)",
    "main_pipe_diameter_m": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์ (Pipe Diameter)",
    "pipe_material_or_roughness": "ชนิดวัสดุท่อ หรือค่าความขรุขระ (Manning's n / Friction Factor f)",
    "manning_n": "ค่าสัมประสิทธิ์ความขรุขระ Manning's n",
    "darcy_f": "ค่าสัมประสิทธิ์ความเสียดทาน Darcy Friction Factor (f)",
    "minor_loss_fittings": "รายการอุปกรณ์ประกอบในระบบท่อ (Minor Loss Fittings)",
    "proposed_pump_head_m": "เฮดของเครื่องสูบน้ำที่ผู้รับจ้างเสนอ (Proposed Pump Duty Head)",
    "proposed_pump_shutoff_head_m": "เฮดสูงสุดที่อัตราการไหลเป็นศูนย์ (Pump Shutoff Head)",
    "proposed_pump_eff_pct": "ประสิทธิภาพเครื่องสูบน้ำ (Pump Efficiency %)",
    "proposed_gear_eff_pct": "ประสิทธิภาพชุดเกียร์ทดรอบ (Gearbox Efficiency %)",
    "proposed_motor_kw": "ขนาดกำลังมอเตอร์ไฟฟ้าที่เสนอ (Proposed Motor Power kW)",
    "proposed_npshr_m": "ค่า NPSHr ของเครื่องสูบน้ำ (Net Positive Suction Head Required)"
}

def translate_field_name(raw_name):
    """แปลงชื่อตัวแปรภาษาอังกฤษที่ AI อาจส่งติดมา ให้เป็นภาษาไทยที่สวยงาม"""
    clean_key = str(raw_name).strip()
    if clean_key in FIELD_THAI_MAP:
        return FIELD_THAI_MAP[clean_key]
    for eng_k, th_v in FIELD_THAI_MAP.items():
        if eng_k in clean_key.lower():
            return th_v
    return clean_key

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

def check_step2_system_completeness(data):
    """
    ตรวจสอบเฉพาะข้อมูลที่จำเป็นสำหรับส่วนที่ 2 (คำนวณ Head Loss และสร้าง System Curve) เท่านั้น!
    ไม่นำข้อมูลปั๊มหรือมอเตอร์มาล็อกส่วนที่ 2 เด็ดขาด
    """
    missing_step2 = []

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
        missing_step2.append(FIELD_THAI_MAP["flow_rate_q_m3s"])
    if h_st is None:
        missing_step2.append(FIELD_THAI_MAP["static_head_m"])
    if not (has_valid_segment or has_valid_main):
        if not main_d or float(main_d or 0) <= 0:
            missing_step2.append(FIELD_THAI_MAP["main_pipe_diameter_m"])
        if not main_l or float(main_l or 0) <= 0:
            missing_step2.append(FIELD_THAI_MAP["main_pipe_length_m"])
    if not has_roughness:
        missing_step2.append(FIELD_THAI_MAP["pipe_material_or_roughness"])

    return (len(missing_step2) == 0), missing_step2

def check_step4_pump_completeness(data, edited_pump_df):
    """
    ตรวจสอบข้อมูลที่จำเป็นสำหรับส่วนที่ 4 (พิจารณาเลือกปั๊ม กำลังมอเตอร์ และ NPSH)
    แยกออกเป็นรายการที่ขาด เพื่อให้ล็อกเฉพาะส่วนที่คำนวณไม่ได้ หรือให้ผู้ใช้กดใช้ค่าแนะนำแทนได้
    """
    missing_step4 = []

    has_pump_head = (data.get("proposed_pump_head_m") is not None) or (
        edited_pump_df is not None and not edited_pump_df.dropna(subset=["Head_m"]).empty
    )
    has_pump_eff = (data.get("proposed_pump_eff_pct") is not None)
    has_gear_eff = (data.get("proposed_gear_eff_pct") is not None)
    has_motor_kw = (data.get("proposed_motor_kw") is not None and float(data.get("proposed_motor_kw") or 0) > 0)
    has_npshr = (data.get("proposed_npshr_m") is not None)

    if not has_pump_head:
        missing_step4.append(FIELD_THAI_MAP["proposed_pump_head_m"])
    if not has_pump_eff:
        missing_step4.append(FIELD_THAI_MAP["proposed_pump_eff_pct"])
    if not has_gear_eff:
        missing_step4.append(FIELD_THAI_MAP["proposed_gear_eff_pct"])
    if not has_motor_kw:
        missing_step4.append(FIELD_THAI_MAP["proposed_motor_kw"])
    if not has_npshr:
        missing_step4.append(FIELD_THAI_MAP["proposed_npshr_m"])

    return (len(missing_step4) == 0), missing_step4, {
        "has_pump_head": has_pump_head,
        "has_pump_eff": has_pump_eff,
        "has_gear_eff": has_gear_eff,
        "has_motor_kw": has_motor_kw,
        "has_npshr": has_npshr
    }

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
    """แปลงไฟล์ภาพเป็น Base64 สำหรับ OpenAI Vision"""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# ==========================================
# 2. ฟังก์ชันคำนวณชลศาสตร์ (PHASE 2: HYDRAULIC ENGINE)
# ==========================================
def run_hydraulic_engine(params, formula_override=None):
    """คำนวณ Major Loss, Minor Loss, ความเร็ว V และสร้างสมการ H = K*Q^2 + Hst"""
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
            "segment_name": "ท่อหลัก / อุโมงค์ส่งน้ำ (Main Conduit)",
            "carries_single_pump_flow": False,
            "diameter_m": float(params.get("main_pipe_diameter_m") or 1.0),
            "area_m2": None,
            "length_m": float(params.get("main_pipe_length_m") or 100.0),
            "manning_n": float(params.get("manning_n") or 0.014),
            "darcy_f": float(params.get("darcy_f") or 0.015),
            "minor_loss_k_sum": float(params.get("total_minor_k") or 0.0),
            "fixed_minor_loss_m": float(params.get("fixed_minor_loss_m") or 0.0)
        }]

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
    """สร้างสมการ Pump Curve (Exact Polynomial Fit หรือ Synthesized Curve) และหาจุดตัด Operating Point"""
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
        curve_type_label = "Exact Model Curve (Polynomial Fit จากข้อมูลจริง)"

        if "Eff_pct" in clean_df.columns and clean_df["Eff_pct"].dropna().count() >= 3:
            eff_Coef = np.polyfit(q_total_arr, clean_df["Eff_pct"].fillna(0).values.astype(float), 2)
        else:
            eff_Coef = None
    else:
        if len(clean_df) >= 1:
            q_duty_single = float(clean_df["Q_single_m3s"].iloc[-1])
            h_duty = float(clean_df["Head_m"].iloc[-1])
            eff_duty = float(clean_df["Eff_pct"].iloc[-1]) if ("Eff_pct" in clean_df.columns and pd.notna(clean_df["Eff_pct"].iloc[-1])) else 85.0
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
        curve_type_label = "Synthesized / Approximated Curve (กราฟจำลองประมาณการ)"
        eff_Coef = [-eff_duty / (q_duty_total_plot ** 2), (2.0 * eff_duty) / q_duty_total_plot, 0.0]

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
        if eff_Coef is not None:
            eff_op = float(np.polyval(eff_Coef, q_op_plot))
            eff_op = max(0.0, min(95.0, eff_op))

    return {
        "is_synthesized": is_synthesized,
        "curve_type_label": curve_type_label,
        "poly_coeffs": (a_p, b_p, c_p),
        "eff_coeffs": eff_Coef,
        "q_op_plot": q_op_plot,
        "q_op_m3s": (q_op_plot / scale) if q_op_plot is not None else None,
        "h_op_m": h_op,
        "eff_op_pct": eff_op,
        "clean_points_df": clean_df
    }

# ==========================================
# 4. ฟังก์ชันพิจารณาความเหมาะสมและอธิบายผลเชิงวิศวกรรม (PHASE 4 ENGINE)
# ==========================================
def select_standard_motor_kw(min_req_kw):
    """เทียบขนาดมอเตอร์มาตรฐานอุตสาหกรรม (IEC / NEMA Standard Ratings)"""
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
    """
    ตรวจสอบเฉพาะขนาดท่อและความเร็วน้ำ (สามารถทำงานและแสดงผลได้ทันทีโดยไม่ต้องรอสเปคปั๊ม!)
    """
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
                f"🔴 **{seg_name}:** ความเร็วน้ำสูงเกินเกณฑ์ ($V = {v_val:.2f}\\text{{ m/s}} > {v_max_limit:.1f}\\text{{ m/s}}$ บนพื้นที่หน้าตัด ${area_val:.2f}\\text{{ m}}^2$) "
                f"→ **ผลกระทบ:** เกิดการสูญเสียเฮด (Head Loss) พุ่งสูงขึ้นอย่างรวดเร็ว เสี่ยงต่อการกัดเซาะผนังท่อ/คอนกรีต (Erosion) เกิดการสั่นสะเทือน (Hydraulic Vibration) และแรงกระแทกน้ำ (Water Hammer)"
            )
        elif 0 < v_val < 0.6 and "ปล่อง" not in seg_name and "shaft" not in seg_name.lower():
            vel_warnings.append(
                f"🟡 **{seg_name}:** ความเร็วน้ำต่ำกว่าเกณฑ์ทำความสะอาดตัวเอง ($V = {v_val:.2f}\\text{{ m/s}} < 0.60\\text{{ m/s}}$) "
                f"→ **ผลกระทบ:** เสี่ยงต่อการตกตะกอนของทรายและโคลนตม (Sedimentation / Siltation) สะสมในท้องท่อหรือรางน้ำ ทำให้พื้นที่หน้าตัดการไหลลดลงในระยะยาว"
            )
        else:
            vel_details.append(f"🟢 **{seg_name}:** $V = {v_val:.2f}\\text{{ m/s}}$ (พื้นที่หน้าตัด ${area_val:.2f}\\text{{ m}}^2$) — *อยู่ในเกณฑ์มาตรฐานวิศวกรรม ($0.6 - {v_max_limit:.1f}\\text{{ m/s}}$)*")

    pipe_pass = len(vel_warnings) == 0
    if pipe_pass:
        pipe_status = "✅ ขนาดท่อเหมาะสม (PASS)"
        pipe_narrative = (
            "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ในทุกช่วงของแนวท่อและอุโมงค์ พบว่าขนาดหน้าตัดท่อมีความเหมาะสม "
            "ความเร็วน้ำอยู่ในช่วงมาตรฐานวิศวกรรม ($0.6 - 3.5\\text{{ m/s}}$) ไม่พบจุดที่เสี่ยงต่อการกัดเซาะผนังท่อหรือการเกิดค้อนน้ำรุนแรง"
        )
    else:
        pipe_status = f"⚠️ พบจุดเสี่ยงความเร็วน้ำ ({len(vel_warnings)} จุด)"
        pipe_narrative = (
            "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ตามขนาดหน้าตัดท่อที่กำหนด **พบช่วงท่อที่มีความเร็วน้ำอยู่นอกเกณฑ์แนะนำทางวิศวกรรม** "
            "ซึ่งส่งผลกระทบโดยตรงต่อพฤติกรรมชลศาสตร์และความปลอดภัยของโครงสร้างดังนี้:"
        )

    return {
        "pipe_pass": pipe_pass,
        "pipe_status": pipe_status,
        "pipe_narrative": pipe_narrative,
        "vel_warnings": vel_warnings,
        "vel_details": vel_details
    }

def evaluate_phase4_full(
    working_data, calc_res, pump_eval, num_active_pumps,
    pump_eff_pct, gear_eff_pct, safety_margin_pct, npshr_m,
    proposed_motor_kw, is_approx_unlocked=False, missing_step4_list=None
):
    """วิเคราะห์ความเหมาะสมเชิงวิศวกรรมครบทั้ง 5 ด้านสำหรับส่วนที่ 4"""
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

    # 1. อธิบายสิ่งที่เห็นบนกราฟ & โหมดการเดินเครื่อง
    if num_active_pumps > 1:
        mode_title = f"การเดินเครื่องสูบน้ำแบบขนาน (Parallel Operation จำนวน {num_active_pumps} เครื่อง)"
        graph_narrative = (
            f"จากกราฟเปรียบเทียบ **System Curve vs. Pump Curve** ระบบถูกกำหนดให้ **{mode_title}** "
            f"ซึ่งในทางชลศาสตร์ การต่อปั๊มขนานกันจะทำให้อัตราการไหลรวม ($Q_{{total}}$) เพิ่มขึ้นเป็น {num_active_pumps} เท่าของปั๊มเดี่ยว ณ ระดับ Head เดียวกัน "
            f"(สังเกตได้จากเส้นกราฟปั๊มรวมสีชมพูที่ยืดขยายออกไปทางขวาเมื่อเทียบกับเส้นประสีส้มของปั๊มเดี่ยว 1 เครื่อง) "
            f"ในขณะที่เส้นสีฟ้า (**System Curve**) เริ่มต้นที่ระยะยกน้ำสถิต $H_{{st}} = {h_static:.2f}\\text{{ m}}$ และโค้งพาราโบลาชันขึ้นตามความเสียดทานในแนวท่อและอุปกรณ์ ($H_{{loss}} = KQ^2$)"
        )
    else:
        mode_title = "การเดินเครื่องสูบน้ำเดี่ยว (Single Pump Operation จำนวน 1 เครื่อง)"
        graph_narrative = (
            f"จากกราฟเปรียบเทียบ **System Curve vs. Pump Curve** ระบบทำงานแบบ **{mode_title}** "
            f"โดยเส้นสีฟ้า (**System Curve**) เริ่มต้นที่ระยะยกน้ำสถิต $H_{{st}} = {h_static:.2f}\\text{{ m}}$ และโค้งชันขึ้นตามอัตราการไหลยกกำลังสอง ($H = KQ^2 + H_{{st}}$) "
            f"ตัดกับเส้นกราฟสมรรถนะของปั๊ม (เส้นสีชมพู) ซึ่งมีจุด Shutoff Head (ที่ $Q=0$) อยู่ที่ประมาณ ${c_p:.2f}\\text{{ m}}$ และลดระดับลงเมื่ออัตราการไหลเพิ่มขึ้น"
        )

    # 2. ความเพียงพอของ Q และ H
    if q_op_total is not None and h_op is not None:
        q_op_unit = q_op_total / max(num_active_pumps, 1)
        q_diff = q_op_total - q_target_total
        h_margin = pump_h_at_target - h_sys_req
        qh_pass = (q_diff >= -0.05) and (h_op >= h_sys_req - 0.15)

        if qh_pass:
            qh_status = "✅ เพียงพอ (PASS)"
            qh_narrative = (
                f"**ผ่านเกณฑ์ด้านอัตราการไหลและเฮด:** ที่อัตราการไหลเป้าหมายของโครงการ $Q_{{target}} = {q_target_total:,.2f}\\text{{ m}}^3/\\text{{s}}$ "
                f"({q_target_unit:,.2f} $\\text{{m}}^3/\\text{{s}}$ ต่อเครื่อง) ระบบต้องการเฮดรวม (Total System Head) เท่ากับ **${h_sys_req:.3f}\\text{{ m}}$** "
                f"(แบ่งเป็น Static Head ${h_static:.2f}\\text{{ m}}$ และ Dynamic Head Loss รวม ${h_dyn_loss:.3f}\\text{{ m}}$) "
                f"เมื่อพล็อตกราฟตัดกัน จุดทำงานจริง (**Operating Point**) ไปตัดกันที่ **$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** "
                f"(${q_op_unit:,.2f}\\text{{ m}}^3/\\text{{s}}$ ต่อเครื่อง) และ **$H_{{op}} = {h_op:.2f}\\text{{ m}}$** "
                f"ซึ่งให้ปริมาณน้ำเกินกว่าเป้าหมายอยู่ $+{max(q_diff, 0.0):.2f}\\text{{ m}}^3/\\text{{s}}$ และมีเฮดเผื่อ (Head Margin) ที่จุดออกแบบ $+{max(h_margin, 0.0):.3f}\\text{{ m}}$ สามารถสูบระบายน้ำเอาชนะแรงเสียดทานในระบบได้อย่างสมบูรณ์"
            )
        else:
            qh_status = "❌ ไม่เพียงพอ (DEFICIT / FAIL)"
            deficit_pct = (abs(q_diff) / q_target_total * 100.0) if q_target_total > 0 else 0.0
            qh_narrative = (
                f"**ไม่ผ่านเกณฑ์ด้านอัตราการไหลและเฮด:** ที่อัตราการไหลเป้าหมาย $Q_{{target}} = {q_target_total:,.2f}\\text{{ m}}^3/\\text{{s}}$ "
                f"ระบบมีความต้านทานรวมสูงถึง **${h_sys_req:.3f}\\text{{ m}}$** แต่กราฟของปั๊มที่เสนอให้เฮดไม่เพียงพอ (ขาดไป ${abs(h_margin):.3f}\\text{{ m}}$ ณ อัตราไหลออกแบบ) "
                f"ทำให้บนกราฟ เส้นปั๊มตกลงมาตัดกับเส้น System Curve ก่อนถึงเป้าหมาย โดยจุดตัด **Operating Point** จริงทำอัตราการไหลได้เพียง "
                f"**$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** (${q_op_unit:,.2f}\\text{{ m}}^3/\\text{{s}}$ ต่อเครื่อง) ที่เฮด **${h_op:.2f}\\text{{ m}}$** "
                f"**ส่งผลให้อัตราการไหลสูบน้ำขาดหายไปถึง ${abs(q_diff):,.2f}\\text{{ m}}^3/\\text{{s}}$ (คิดเป็น $-{deficit_pct:.1f}\\%$)** ไม่สามารถระบายน้ำได้ทันตามเกณฑ์ที่โครงการกำหนด"
            )
    else:
        qh_pass = False
        q_op_unit = 0.0
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

    has_proposed_motor = (proposed_motor_kw is not None) and (float(proposed_motor_kw or 0) > 0)
    eff_good = 78.0 <= eff_pump <= 92.0
    eff_desc = f"อยู่ในเกณฑ์ดีและใกล้เคียงจุด BEP ({eff_pump:.1f}%)" if eff_good else f"ค่อนข้างต่ำและตกอยู่นอกช่วง BEP ({eff_pump:.1f}%) ซึ่งทำให้สิ้นเปลืองพลังงานสูงกว่าปกติ"

    if has_proposed_motor:
        prop_motor_kw = float(proposed_motor_kw)
        motor_pass = prop_motor_kw >= p_min_motor_kw
        actual_margin_pct = (prop_motor_kw / p_gear_shaft_kw) * 100.0 if p_gear_shaft_kw > 0 else 0.0

        if motor_pass and eff_good:
            power_status = f"✅ เหมาะสม ({prop_motor_kw:,.0f} kW)"
            power_narrative = (
                f"**ผ่านเกณฑ์ด้านประสิทธิภาพและกำลังมอเตอร์:** ณ จุดพิจารณา ประสิทธิภาพเครื่องสูบน้ำอยู่ที่ **{eff_pump:.1f}%** ({eff_desc}) "
                f"และประสิทธิภาพชุดเกียร์ทดรอบอยู่ที่ **{eff_gear:.1f}%** คำนวณกำลังน้ำ (Hydraulic Power) ต่อเครื่องได้ **${p_hydraulic_kw:,.1f}\\text{{ kW}}$** "
                f"คิดเป็นกำลังเพลาปั๊ม (Pump Shaft Power) **${p_pump_shaft_kw:,.1f}\\text{{ kW}}$** และกำลังเพลาเกียร์ (Gear Input Power) **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคูณเผื่อ Safety Margin ที่ **{safety_margin_pct:.0f}%** จะต้องการขนาดมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"ดังนั้น มอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** ที่ระบุจึงมีความเหมาะสม (คิดเป็น Safety Margin จริง **{actual_margin_pct:.1f}%**) ไม่มีความเสี่ยงเรื่องการเกิด Overload"
            )
        elif not motor_pass:
            power_status = f"❌ เสี่ยง Overload ({prop_motor_kw:,.0f} kW < {p_min_motor_kw:,.0f} kW)"
            overload_deficit = p_min_motor_kw - prop_motor_kw
            power_narrative = (
                f"**ไม่ผ่านเกณฑ์ขนาดมอเตอร์ (เสี่ยงเกิด Motor Overload):** ที่ประสิทธิภาพปั๊ม **{eff_pump:.1f}%** ({eff_desc}) และประสิทธิภาพเกียร์ **{eff_gear:.1f}%** "
                f"ปั๊มต้องใช้กำลังเพลาปั๊มสูงถึง **${p_pump_shaft_kw:,.1f}\\text{{ kW}}$** และกำลังเพลาเกียร์ **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** ตามมาตรฐานวิศวกรรม จะต้องใช้มอเตอร์ขนาดอย่างน้อย **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"แต่มอเตอร์ที่เสนอมีขนาดเพียง **${prop_motor_kw:,.0f}\\text{{ kW}}$** (ขาดไป **${overload_deficit:,.1f}\\text{{ kW}}$** หรือมี Margin เพียง {actual_margin_pct:.1f}%) "
                f"**ผลกระทบ:** มอเตอร์จะเกิดความร้อนสะสม รันเกินพิกัด (Overload) และเบรกเกอร์ทริปตัดการทำงานเมื่อเดินเครื่องเต็มกำลัง **แนะนำให้ปรับเพิ่มขนาดมอเตอร์เป็นรุ่นมาตรฐาน ${rec_standard_motor_kw:,.0f}\\text{{ kW}}$ ต่อเครื่อง**"
            )
        else:
            power_status = f"⚠️ มอเตอร์พอแต่ Eff ต่ำ ({eff_pump:.1f}%)"
            power_narrative = (
                f"**ข้อควรระวังด้านประสิทธิภาพ (Off-BEP):** แม้ว่ามอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** จะสูงกว่าค่าขั้นต่ำที่ต้องการ (${p_min_motor_kw:,.1f}\\text{{ kW}}$) "
                f"แต่ประสิทธิภาพของปั๊ม ณ จุดทำงานต่ำเพียง **{eff_pump:.1f}%** แสดงว่าปั๊มทำงานห่างจากจุด Best Efficiency Point (BEP) ทำให้เกิดการสูญเสียพลังงานและเกิดแรงดันปั่นป่วนในเรือนปั๊ม"
            )
    else:
        motor_pass = True
        prop_motor_kw = None
        power_status = f"💡 แนะนำมอเตอร์ {rec_standard_motor_kw:,.0f} kW"
        power_narrative = (
            f"**การคำนวณและแนะนำขนาดมอเตอร์ (กรณีไม่ได้ระบุขนาดมอเตอร์ในเอกสาร):** ที่ประสิทธิภาพเครื่องสูบน้ำ **{eff_pump:.1f}%** ({eff_desc}) "
            f"และประสิทธิภาพชุดเกียร์ **{eff_gear:.1f}%** ระบบต้องการกำลังน้ำต่อเครื่อง **${p_hydraulic_kw:,.1f}\\text{{ kW}}$**, "
            f"กำลังเพลาปั๊ม (Pump Shaft Power) **${p_pump_shaft_kw:,.1f}\\text{{ kW}}$**, และกำลังเพลาเกียร์ (Gear Input Power) **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
            f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** จะต้องการกำลังมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
            f"👉 **ดังนั้น ขอแนะนำให้เลือกใช้มอเตอร์ไฟฟ้าขนาดพิกัดมาตรฐาน (Recommended Standard Motor Rating) ที่ขนาด `${rec_standard_motor_kw:,.0f} kW` ต่อเครื่อง**"
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

    if npshr_m > 6.5:
        risk_flags.append(
            f"🚨 **ความเสี่ยงเกิดโพรงอากาศรุนแรง (High NPSHr / Cavitation Flag):** ปั๊มมีค่า $\\text{{NPSHr}}$ สูงถึง **${npshr_m:.1f}\\text{{ m}}$** "
            f"ในขณะที่เมื่อเดินเครื่องสูบน้ำเต็มพิกัด ระดับน้ำในบ่อสูบ (Suction Sump Water Level) จะลดต่ำลงไปอยู่ที่ประมาณ **${sump_wl:.2f}\\text{{ m.MSL}}$** "
            f"(เนื่องจากสูญเสียเฮดในอุโมงค์ฝั่งดูด) ทำให้ค่า $\\text{{NPSHa}}$ หน้าใบพัดลดต่ำลง เสี่ยงต่อการเกิด **Cavitation** กัดกร่อนใบพัดปั๊ม เกิดเสียงดังและการสั่นสะเทือนรุนแรง"
        )
    else:
        risk_flags.append(
            f"✅ **การตรวจสอบ NPSH และระดับน้ำบ่อสูบ:** ค่า $\\text{{NPSHr}}$ อยู่ที่ **${npshr_m:.1f}\\text{{ m}}$** อยู่ในเกณฑ์มาตรฐาน "
            f"ทั้งนี้ผู้ควบคุมงานต้องตั้งค่าระดับน้ำต่ำสุดในบ่อสูบ (Low Water Level Interlock) ให้สอดคล้องกับระดับน้ำขณะเดินเครื่องเต็มพิกัด (ประมาณ **${sump_wl:.2f}\\text{{ m.MSL}}$**) เพื่อป้องกันอากาศถูกดูดเข้าท่อดูด (Vortexing)"
        )

    if eff_pump < 78.0:
        risk_flags.append(
            f"⚠️ **ความเสี่ยงการทำงานนอกจุด BEP (Off-BEP / Runout Operation):** ประสิทธิภาพปั๊ม ณ จุดทำงานลดต่ำลงเหลือเพียง **{eff_pump:.1f}%** "
            f"บ่งชี้ว่าจุดตัดการทำงานลื่นไหลไปอยู่ทางขวาของกราฟมากเกินไป (Runout Zone) ซึ่งจะทำให้เกิดแรงกระทำในแนวรัศมีต่อเพลาปั๊ม (Radial Thrust) สูงขึ้น ซีลและแบริ่งเสื่อมสภาพเร็ว"
        )

    if pump_eval["is_synthesized"]:
        risk_flags.append(
            "ℹ️ **ความไม่แน่นอนจากกราฟปั๊มจำลอง (Synthesized Curve Uncertainty):** เนื่องจากไม่ได้ระบุจุดพิกัดกราฟปั๊มครบ 3 จุดขึ้นไป "
            "การหาจุดตัดจึงอ้างอิงจากเส้นโค้งจำลอง ($H_{{shutoff}} = 1.30 H_{{duty}}$) ควรเรียกขอเอกสาร Performance Curve จริงจากโรงงานผู้ผลิตเพื่อยืนยันก่อนอนุมัติขั้นสุดท้าย"
        )

    if is_approx_unlocked and missing_step4_list:
        risk_flags.append(
            f"⚠️ **หมายเหตุการปลดล็อกด้วยค่าประมาณการ (Approximated / Recommended Parameters Used):** การประเมินในส่วนที่ 4 นี้ใช้ค่าแนะนำ/ค่าจำลองแทนข้อมูลที่ขาดหายไป ได้แก่: "
            f"**{', '.join(missing_step4_list)}** ผลลัพธ์นี้ใช้เพื่อการจำลองและวางแผนเบื้องต้นเท่านั้น ผู้ควบคุมงานต้องตรวจสอบกับเอกสาร Submittal จริงอีกครั้ง"
        )

    overall_suitable = qh_pass and motor_pass and eff_good and pipe_eval["pipe_pass"] and (npshr_m <= 6.5)

    return {
        "overall_suitable": overall_suitable,
        "is_approx_unlocked": is_approx_unlocked,
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

# ==========================================
# 5. โครงสร้างคำสั่ง AI (แยกข้อมูลท่อ กับ ข้อมูลปั๊ม ออกจากกันชัดเจน)
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
                "flow_rate_q_m3s": {"type": "number", "description": "อัตราการไหลรวมที่ต้องการของระบบในหน่วย m3/s"},
                "flow_rate_original_text": {"type": "string"},
                "num_parallel_pumps": {"type": "integer", "description": "จำนวนเครื่องสูบน้ำที่เดินขนานกัน (ถ้าเป็นปั๊มเดี่ยวใส่ 1)"},
                "static_head_m": {"type": "number", "description": "ระยะยกน้ำสถิต (Static Head / Hst) หน่วยเมตร ห้ามเดาเอง"},
                "suction_wl_m": {"type": "number"},
                "discharge_wl_m": {"type": "number"},
                "main_pipe_length_m": {"type": "number", "description": "ความยาวรวมของท่อหรืออุโมงค์หลัก (m) ห้ามเดาเอง"},
                "main_pipe_diameter_m": {"type": "number", "description": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์หลัก (m)"},
                "pipe_material_or_roughness": {"type": "string"},
                "manning_n": {"type": "number"},
                "darcy_f": {"type": "number"},
                "system_type": {"type": "string"},
                "proposed_pump_head_m": {"type": "number", "description": "ค่า Operating Head ของปั๊มที่ผู้รับจ้างเสนอ (m) ถ้าไม่ระบุให้ใส่ null"},
                "proposed_pump_shutoff_head_m": {"type": "number"},
                "proposed_pump_eff_pct": {"type": "number", "description": "ประสิทธิภาพปั๊ม (%) ถ้าไม่ระบุให้ใส่ null"},
                "proposed_gear_eff_pct": {"type": "number", "description": "ประสิทธิภาพเกียร์ทดรอบ (%) ถ้าไม่ระบุให้ใส่ null"},
                "proposed_motor_kw": {"type": "number", "description": "ขนาดกำลังมอเตอร์ที่เสนอ (kW) ถ้าไม่ระบุให้ใส่ null"},
                "proposed_npshr_m": {"type": "number", "description": "ค่า NPSHr ของปั๊ม (m) ถ้าไม่ระบุให้ใส่ null"},
                "minor_loss_fittings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
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
                "ai_curve_advice_th": {"type": "string"},
                "missing_system_curve_data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ระบุเฉพาะข้อมูลของ 'ระบบท่อและระดับน้ำ' ที่ขาดหายไปสำหรับคำนวณ System Curve (ได้แก่ อัตราการไหล Q, Static Head, ขนาดท่อ, ความยาวท่อ, ความขรุขระ) ห้ามใส่ข้อมูลปั๊มหรือมอเตอร์ลงในนี้ และต้องเขียนเป็นภาษาไทยเท่านั้น"
                },
                "missing_pump_selection_data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ระบุเฉพาะข้อมูลของ 'เครื่องสูบน้ำและมอเตอร์' ที่ไม่ได้ระบุมาสำหรับประเมินในส่วนที่ 4 (ได้แก่ เฮดปั๊มที่เสนอ, ประสิทธิภาพปั๊ม, ขนาดมอเตอร์ kW, ค่า NPSHr) เขียนเป็นภาษาไทยเท่านั้น"
                }
            },
            "required": ["missing_system_curve_data", "missing_pump_selection_data", "pump_curve_confidence"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, ตารางคำนวณ, รูปภาพกราฟ หรือข้อความจากหน้างาน และสกัดพารามิเตอร์ทางชลศาสตร์ออกมา

กฎเหล็ก (Operational Rules):
1. สื่อสารเป็น "ภาษาไทย" เสมอ ห้ามส่งชื่อตัวแปรภาษาอังกฤษดิบๆ (เช่น proposed_pump_head_m) ออกมาในรายการข้อมูลที่ขาดเด็ดขาด
2. ห้ามเดาตัวเลขใดๆ ที่ผู้ใช้งานไม่ได้ระบุ หากไม่มีให้ใส่ค่าเป็น null
3. แยกการตรวจสอบข้อมูลที่ขาดออกเป็น 2 ส่วนอย่างชัดเจน:
   - `missing_system_curve_data`: เฉพาะข้อมูลระบบท่อที่จำเป็นต้องใช้คำนวณ Head Loss และสร้าง System Curve (อัตราการไหล Q, ระยะยกน้ำ Static Head, ขนาดท่อ D, ความยาวท่อ L, ค่าความขรุขระ n/f)
   - `missing_pump_selection_data`: เฉพาะข้อมูลสเปคเครื่องสูบน้ำและมอเตอร์สำหรับใช้ตัดสินใจในส่วนที่ 4 (เฮดปั๊มที่เสนอ, ประสิทธิภาพปั๊ม, ขนาดมอเตอร์ kW, ค่า NPSHr)
"""

# ==========================================
# 6. ส่วนแสดงผล STEP 1: นำเข้าและสกัดข้อมูล
# ==========================================
st.subheader("1️⃣ ส่วนที่ 1: รับข้อมูลนำเข้า (Input Parameters & AI Extraction)")
st.write("อัปโหลดเอกสาร Submittal (PDF/Excel/CSV), **รูปภาพกราฟ Pump Curve (JPG/PNG)**, ถ่ายรูปจากกล้อง หรือพิมพ์ข้อความ")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "📂 อัปโหลดไฟล์เอกสารหรือรูปภาพกราฟ Pump Curve",
        accept_multiple_files=True,
        type=['pdf', 'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'txt']
    )
    camera_photo = st.camera_input("📸 หรือถ่ายรูปกราฟ Pump Curve / ตารางสเปค")

with col2:
    manual_input = st.text_area(
        "✍️ หรือกรอกข้อมูลสเปคปั๊มและระบบท่อด้วยตนเอง:",
        height=180,
        placeholder="พิมพ์รายละเอียดสเปคปั๊ม แนวท่อ หรือจุดพิกัดกราฟปั๊ม (Q, H) ที่นี่..."
    )

if st.button("🚀 สกัดข้อมูลและวิเคราะห์พารามิเตอร์ (Extract & Parse Data)", type="primary", use_container_width=True):
    if uploaded_files or camera_photo or manual_input:
        with st.spinner("🤖 AI กำลังอ่านข้อมูล ตรวจสอบพารามิเตอร์ระบบท่อ และสกัดพิกัด Pump Curve..."):
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

            user_content = [{"type": "text", "text": f"กรุณาสกัดพารามิเตอร์ทางชลศาสตร์และพิกัดกราฟปั๊มจากข้อมูลต่อไปนี้:\n\n{text_content}"}]
            for img_b64 in image_base64_list:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                tools=extraction_tool,
                tool_choice={"type": "function", "function": {"name": "extract_hydraulic_parameters"}}
            )

            tool_call = response.choices[0].message.tool_calls[0]
            extracted = json.loads(tool_call.function.arguments)

            if not extracted.get("main_pipe_diameter_m") and extracted.get("pipe_segments"):
                first_d = extracted["pipe_segments"][0].get("diameter_m")
                if first_d:
                    extracted["main_pipe_diameter_m"] = first_d

            st.session_state['extracted_params'] = extracted
            if 'custom_pump_points_df' in st.session_state:
                del st.session_state['custom_pump_points_df']
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์, ถ่ายรูป, หรือพิมพ์ข้อความก่อนกดปุ่มวิเคราะห์ครับ")

# ==========================================
# แสดงผลการ์ดพารามิเตอร์ & ระบบล็อก GATEKEEPER แยกส่วน (STEP 2 vs STEP 4)
# ==========================================
if 'extracted_params' in st.session_state:
    raw_data = st.session_state['extracted_params']
    st.divider()
    st.markdown("### 📋 สรุปพารามิเตอร์ที่ระบบอ่านได้จากเอกสารและรูปภาพ")

    c1, c2, c3 = st.columns(3)
    q_display = raw_data.get('flow_rate_original_text') or format_metric(raw_data.get('flow_rate_q_m3s'), "m³/s")
    c1.metric("อัตราการไหลรวม (Q)", q_display)
    c2.metric("ระยะยกน้ำ (Static Head)", format_metric(raw_data.get('static_head_m'), "m"))
    c3.metric("Head ปั๊มที่เสนอ (Duty H)", format_metric(raw_data.get('proposed_pump_head_m'), "m"))

    c4, c5, c6 = st.columns(3)
    c4.metric("ความยาวท่อหลัก (Pipe Length)", format_metric(raw_data.get('main_pipe_length_m'), "m"))
    c5.metric("ขนาดท่อหลัก (Pipe Diameter)", format_metric(raw_data.get('main_pipe_diameter_m'), "m"))
    c6.metric("วัสดุ/ความขรุขระ (Roughness)", format_metric(raw_data.get('pipe_material_or_roughness'), ""))

   with st.expander("🔧 ดูรายการอุปกรณ์ประกอบ (Minor Loss Fittings) และสเปคมอเตอร์", expanded=False):
        st.write(f"**ลักษณะระบบ:** {format_metric(raw_data.get('system_type'), '')} | **จำนวนปั๊มเดินขนาน:** {format_metric(raw_data.get('num_parallel_pumps'), 'เครื่อง', 0)} | **ขนาดมอเตอร์ที่เสนอ:** {format_metric(raw_data.get('proposed_motor_kw'), 'kW', 0)}")
        
        fittings = raw_data.get('minor_loss_fittings') or []
        if fittings:
            for item in fittings:
                st.write(f"- {item}")
        else:
            st.write("- ไม่พบข้อมูลอุปกรณ์ประกอบ")
