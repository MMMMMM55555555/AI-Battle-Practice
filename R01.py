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
# 1. ฟังก์ชันช่วยเหลือ & แปลชื่อตัวแปรเป็นภาษาไทย
# ==========================================
FIELD_THAI_MAP = {
    "flow_rate_q_m3s": "อัตราการไหลที่ต้องการ (Flow Rate Q)",
    "static_head_m": "ระยะยกน้ำสถิต (Static Head) หรือระดับน้ำต้นทาง-ปลายทาง",
    "main_pipe_length_m": "ความยาวรวมของท่อหรืออุโมงค์ (Pipe/Tunnel Length)",
    "main_pipe_diameter_m": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์ (Pipe Diameter)",
    "pipe_material_or_roughness": "ชนิดวัสดุท่อ หรือค่าความขรุขระ (Manning's n / Friction Factor f)",
    "proposed_pump_head_m": "เฮดของเครื่องสูบน้ำที่เสนอ (Proposed Pump Head หรือพิกัดกราฟปั๊ม)",
    "proposed_pump_eff_pct": "ประสิทธิภาพเครื่องสูบน้ำ (Pump Efficiency %)",
    "proposed_gear_eff_pct": "ประสิทธิภาพชุดเกียร์ทดรอบ (Gearbox Efficiency %)",
    "proposed_motor_kw": "ขนาดกำลังมอเตอร์ไฟฟ้าที่เสนอ (Proposed Motor Power kW)",
    "proposed_npshr_m": "ค่า NPSHr ของเครื่องสูบน้ำ (Net Positive Suction Head Required)"
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
            f"อุปกรณ์ประกอบในระบบมีค่าสัมประสิทธิ์การสูญเสียรองรวม (Σk) = {extracted['total_minor_k']}"
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
    """ตรวจสอบข้อมูลสำหรับส่วนที่ 4 (ต้องมีขนาดมอเตอร์ kW และประสิทธิภาพปั๊ม)"""
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
        curve_type_label = "Exact Model Curve (Polynomial Fit จากข้อมูลจริง)"

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
        curve_type_label = "Synthesized / Approximated Curve (กราฟจำลองประมาณการ)"
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
# 4. ฟังก์ชันพิจารณาความเหมาะสมและอธิบายผลเชิงวิศวกรรม (PHASE 4 ENGINE)
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
        pipe_narrative = "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ในทุกช่วงของแนวท่อและอุโมงค์ พบว่าขนาดหน้าตัดท่อมีความเหมาะสม ความเร็วน้ำอยู่ในช่วงมาตรฐานวิศวกรรม ไม่พบจุดที่เสี่ยงต่อการกัดเซาะผนังท่อหรือการเกิดค้อนน้ำรุนแรง"
    else:
        pipe_status = f"⚠️ พบจุดเสี่ยงความเร็วน้ำ ({len(vel_warnings)} จุด)"
        pipe_narrative = "จากการตรวจสอบความเร็วการไหล ($V = Q/A$) ตามขนาดหน้าตัดท่อที่กำหนด **พบช่วงท่อที่มีความเร็วน้ำอยู่นอกเกณฑ์แนะนำทางวิศวกรรม** ซึ่งส่งผลกระทบโดยตรงต่อพฤติกรรมชลศาสตร์และความปลอดภัยของโครงสร้างดังนี้:"

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
            f"ตัดกับเส้นกราฟสมรรถนะของปั๊ม (เส้นสีชมพู) ซึ่งมีจุด Shutoff Head (ที่ $Q=0$) อยู่ที่ประมาณ ${c_p:.2f}\\text{{ m}}$ และลดระดับลงเมื่ออัตราการไหลเพิ่มขึ้น"
        )

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
                f"เมื่อพล็อตกราฟตัดกัน จุดทำงานจริง (**Operating Point**) ไปตัดกันที่ **$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** "
                f"และ **$H_{{op}} = {h_op:.2f}\\text{{ m}}$** ซึ่งให้ปริมาณน้ำเกินกว่าเป้าหมายอยู่ $+{max(q_diff, 0.0):.2f}\\text{{ m}}^3/\\text{{s}}$ สามารถสูบระบายน้ำได้ทันเกณฑ์"
            )
        else:
            qh_status = "❌ ไม่เพียงพอ (DEFICIT / FAIL)"
            deficit_pct = (abs(q_diff) / q_target_total * 100.0) if q_target_total > 0 else 0.0
            qh_narrative = (
                f"**ไม่ผ่านเกณฑ์ด้านอัตราการไหลและเฮด:** ระบบมีความต้านทานรวมสูงถึง **${h_sys_req:.3f}\\text{{ m}}$** แต่กราฟของปั๊มที่เสนอให้เฮดไม่เพียงพอ "
                f"ทำให้บนกราฟ เส้นปั๊มตกลงมาตัดกับเส้น System Curve ก่อนถึงเป้าหมาย โดยจุดตัด **Operating Point** จริงทำอัตราการไหลได้เพียง "
                f"**$Q_{{op}} = {q_op_total:,.2f}\\text{{ m}}^3/\\text{{s}}$** ที่เฮด **${h_op:.2f}\\text{{ m}}$** "
                f"**ส่งผลให้อัตราการไหลสูบน้ำขาดหายไปถึง ${abs(q_diff):,.2f}\\text{{ m}}^3/\\text{{s}}$** ไม่สามารถระบายน้ำได้ทันเกณฑ์"
            )
    else:
        qh_pass = False
        q_op_unit = 0.0
        qh_status = "❌ กราฟไม่ตัดกัน (SHUTOFF FAIL)"
        qh_narrative = (
            f"**วิกฤต (ปั๊มไม่สามารถส่งน้ำได้):** เฮดสูงสุดของปั๊ม (Shutoff Head = ${c_p:.2f}\\text{{ m}}$) มีค่าต่ำกว่าระยะยกน้ำสถิตของระบบ ($H_{{st}} = {h_static:.2f}\\text{{ m}}$) "
            f"ทำให้เส้นกราฟปั๊มไม่ตัดกับ System Curve เลย น้ำไม่สามารถไหลผ่านระบบได้"
        )

    eff_pump = float(pump_eff_pct or 85.0)
    if eff_pump <= 0: eff_pump = 80.0
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
    eff_desc = f"อยู่ในเกณฑ์ดีและใกล้เคียงจุด BEP ({eff_pump:.1f}%)" if eff_good else f"ค่อนข้างต่ำและตกอยู่นอกช่วง BEP ({eff_pump:.1f}%) ซึ่งทำให้สิ้นเปลืองพลังงานสูงกว่าปกติ"

    if has_motor:
        prop_motor_kw = float(user_motor_kw)
        motor_pass = prop_motor_kw >= p_min_motor_kw
        actual_margin_pct = (prop_motor_kw / p_gear_shaft_kw) * 100.0 if p_gear_shaft_kw > 0 else 0.0

        if motor_pass and eff_good:
            power_status = f"✅ เหมาะสม ({prop_motor_kw:,.0f} kW)"
            power_narrative = (
                f"**ผ่านเกณฑ์ด้านประสิทธิภาพและกำลังมอเตอร์:** ณ จุดพิจารณา ประสิทธิภาพเครื่องสูบน้ำอยู่ที่ **{eff_pump:.1f}%** ({eff_desc}) "
                f"และประสิทธิภาพชุดเกียร์ทดรอบอยู่ที่ **{eff_gear:.1f}%** คำนวณกำลังเพลาเกียร์ (Gear Input Power) ได้ **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคูณเผื่อ Safety Margin ที่ **{safety_margin_pct:.0f}%** จะต้องการขนาดมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"ดังนั้น มอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** ที่ใช้พิจารณาจึงมีความเหมาะสม (มี Safety Margin จริง **{actual_margin_pct:.1f}%**) ไม่มีความเสี่ยงเรื่อง Overload"
            )
        elif not motor_pass:
            power_status = f"❌ เสี่ยง Overload ({prop_motor_kw:,.0f} kW < {p_min_motor_kw:,.0f} kW)"
            overload_deficit = p_min_motor_kw - prop_motor_kw
            power_narrative = (
                f"**ไม่ผ่านเกณฑ์ขนาดมอเตอร์ (เสี่ยงเกิด Motor Overload):** ปั๊มต้องใช้กำลังเพลาเกียร์รวม **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
                f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** จะต้องใช้มอเตอร์ขนาดอย่างน้อย **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
                f"แต่มอเตอร์ที่ระบุมีขนาดเพียง **${prop_motor_kw:,.0f}\\text{{ kW}}$** (ขาดไป **${overload_deficit:,.1f}\\text{{ kW}}$**) "
                f"**ผลกระทบ:** เสี่ยงต่อการเกิด Overload และเบรกเกอร์ทริปเมื่อเดินเครื่องเต็มกำลัง **แนะนำให้ปรับเพิ่มขนาดมอเตอร์เป็นรุ่นมาตรฐาน ${rec_standard_motor_kw:,.0f}\\text{{ kW}}$ ต่อเครื่อง**"
            )
        else:
            power_status = f"⚠️ มอเตอร์พอแต่ Eff ต่ำ ({eff_pump:.1f}%)"
            power_narrative = (
                f"**ข้อควรระวังด้านประสิทธิภาพ (Off-BEP):** แม้ว่ามอเตอร์ขนาด **${prop_motor_kw:,.0f}\\text{{ kW}}$** จะสูงกว่าค่าขั้นต่ำที่ต้องการ "
                f"แต่ประสิทธิภาพของปั๊มต่ำเพียง **{eff_pump:.1f}%** แสดงว่าปั๊มทำงานห่างจากจุด BEP ทำให้เกิดการสูญเสียพลังงานสูง"
            )
    else:
        motor_pass = True
        prop_motor_kw = None
        power_status = f"💡 แนะนำมอเตอร์ {rec_standard_motor_kw:,.0f} kW"
        power_narrative = (
            f"**การคำนวณและแนะนำขนาดมอเตอร์:** ระบบต้องการกำลังเพลาเกียร์ (Gear Input Power) **${p_gear_shaft_kw:,.1f}\\text{{ kW}}$** "
            f"เมื่อคิดเผื่อ Safety Margin **{safety_margin_pct:.0f}%** จะต้องการกำลังมอเตอร์ขั้นต่ำ **${p_min_motor_kw:,.1f}\\text{{ kW}}$** "
            f"👉 **ขอแนะนำให้เลือกใช้มอเตอร์ไฟฟ้าขนาดพิกัดมาตรฐาน (Recommended Standard Motor Rating) ที่ขนาด `${rec_standard_motor_kw:,.0f} kW` ต่อเครื่อง**"
        )

    pipe_eval = evaluate_pipe_velocity_only(calc_res)

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
            f"ในขณะที่ระดับน้ำในบ่อสูบขณะเดินเครื่อง (Sump WL) จะลดต่ำลงไปอยู่ที่ประมาณ **${sump_wl:.2f}\\text{{ m.MSL}}$** ทำให้เสี่ยงต่อการเกิด Cavitation รุนแรง"
        )
    else:
        risk_flags.append(
            f"✅ **การตรวจสอบ NPSH และระดับน้ำบ่อสูบ:** ค่า $\\text{{NPSHr}}$ อยู่ที่ **${npshr_m:.1f}\\text{{ m}}$** อยู่ในเกณฑ์มาตรฐาน "
            f"ควรตั้งค่า Low Water Level Interlock ให้สอดคล้องกับระดับน้ำขณะเดินเครื่อง (ประมาณ **${sump_wl:.2f}\\text{{ m.MSL}}$**)"
        )

    if eff_pump < 78.0:
        risk_flags.append(
            f"⚠️ **ความเสี่ยงการทำงานนอกจุด BEP (Off-BEP / Runout Operation):** ประสิทธิภาพปั๊มลดต่ำลงเหลือเพียง **{eff_pump:.1f}%** บ่งชี้ว่าจุดตัดการทำงานลื่นไหลไปอยู่ทางขวาของกราฟมากเกินไป"
        )

    if pump_eval["is_synthesized"]:
        risk_flags.append(
            "ℹ️ **ความไม่แน่นอนจากกราฟปั๊มจำลอง (Synthesized Curve Uncertainty):** เนื่องจากไม่ได้ระบุจุดพิกัดกราฟปั๊มครบ 3 จุดขึ้นไป การหาจุดตัดจึงอ้างอิงจากเส้นโค้งจำลอง ควรเรียกขอ Performance Curve จริงจากโรงงานผู้ผลิตเพื่อยืนยัน"
        )

    if is_force_unlocked and missing_step4_list:
        risk_flags.append(
            f"⚠️ **หมายเหตุการปลดล็อกด้วยค่าประมาณการ (Force Access / Approximated Parameters):** การประเมินส่วนที่ 4 ใช้ค่าที่ป้อนด้วยตนเอง/ค่าจำลองแทนข้อมูลที่ขาดหายไป ได้แก่: "
            f"**{', '.join(missing_step4_list)}**"
        )

    overall_suitable = qh_pass and motor_pass and eff_good and pipe_eval["pipe_pass"] and (npshr_m <= 6.5)

    return {
        "overall_suitable": overall_suitable,
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
                "pump_curve_confidence": {"type": "string", "enum": ["high_multi_point", "single_duty_point_only", "uncertain_from_pdf_or_blurry", "no_pump_data"]},
                "missing_system_curve_data": {"type": "array", "items": {"type": "string"}},
                "missing_pump_selection_data": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["missing_system_curve_data", "missing_pump_selection_data", "pump_curve_confidence"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, ตารางคำนวณ, รูปภาพกราฟ หรือข้อความจากหน้างาน และสกัดพารามิเตอร์ทางชลศาสตร์ออกมา
กฎเหล็ก:
1. สื่อสารเป็นภาษาไทยเสมอ ห้ามเดาตัวเลขใดๆ
2. แยกการตรวจสอบข้อมูลที่ขาดออกเป็น 2 ส่วนอย่างชัดเจน:
   - missing_system_curve_data: ข้อมูลระบบท่อ (อัตราการไหล Q, Static Head, ขนาดท่อ, ความยาวท่อ, ความขรุขระ)
   - missing_pump_selection_data: ข้อมูลปั๊มและมอเตอร์ (เฮดปั๊ม, ประสิทธิภาพปั๊ม, ขนาดมอเตอร์ kW, NPSHr)
"""

# ==========================================
# 6. ส่วนแสดงผล STEP 1: นำเข้าและสกัดข้อมูล
# ==========================================
st.subheader("1️⃣ ส่วนที่ 1: รับข้อมูลนำเข้า (Input Parameters & AI Extraction)")
st.write("อัปโหลดเอกสาร Submittal (PDF/Excel/CSV), **รูปภาพกราฟ Pump Curve (JPG/PNG)**, ถ่ายรูปจากกล้อง หรือพิมพ์ข้อความ")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("📂 อัปโหลดไฟล์เอกสารหรือรูปภาพ", accept_multiple_files=True, type=['pdf', 'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'txt'])
    camera_photo = st.camera_input("📸 หรือถ่ายรูปกราฟ / ตารางสเปค")

with col2:
    manual_input = st.text_area("✍️ หรือกรอกข้อมูลสเปคปั๊มและระบบท่อด้วยตนเอง:", height=180, placeholder="พิมพ์รายละเอียดสเปคปั๊ม แนวท่อ หรือพิกัดกราฟปั๊มที่นี่...")

if st.button("🚀 สกัดข้อมูลและวิเคราะห์พารามิเตอร์ (Extract & Parse Data)", type="primary", use_container_width=True):
    if uploaded_files or camera_photo or manual_input:
        with st.spinner("🤖 AI กำลังอ่านข้อมูลและสกัดพารามิเตอร์..."):
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
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์, ถ่ายรูป, หรือพิมพ์ข้อความก่อนกดปุ่มวิเคราะห์ครับ")

# ==========================================
# แสดงผลการ์ดสรุปพารามิเตอร์ในส่วนที่ 1
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

    step2_ok, missing_s2 = check_step2_completeness(raw_data)
    step3_ok, missing_s3 = check_step3_completeness(raw_data)
    step4_ok, missing_s4 = check_step4_completeness(raw_data)

    working_data = copy.deepcopy(raw_data)

    # ==========================================
    # 7. ส่วนที่ 2: คำนวณ HEAD LOSS & SYSTEM CURVE
    # ==========================================
    st.divider()
    st.subheader("2️⃣ ส่วนที่ 2: คำนวณ Head Loss และสร้างสมการ System Curve (H = KQ² + Hst)")

    allow_step_2 = False
    if step2_ok:
        st.success("✅ **พบข้อมูลระบบท่อครบถ้วน (Gatekeeper ส่วนที่ 2 ผ่านเกณฑ์):** พร้อมคำนวณ Head Loss อัตโนมัติ")
        allow_step_2 = True
        edit_pipe_manual = st.toggle("🛠️ ปรับแต่งพารามิเตอร์ระบบท่อด้วยตนเอง (Custom Pipe Parameters)", value=False)
    else:
        st.warning("🔒 **ส่วนที่ 2 ถูกล็อกชั่วคราว (Gatekeeper Locked):** ข้อมูลระบบท่อไม่เพียงพอสำหรับคำนวณ Head Loss")
        for m in missing_s2:
            st.error(f"❌ ขาดข้อมูลระบบท่อ: {m}")
        edit_pipe_manual = st.toggle("🔓 **บังคับเข้าถึงส่วนที่ 2 เพื่อกรอกพารามิเตอร์ระบบท่อเอง (Force Access Part 2)**", value=False)
        if edit_pipe_manual:
            allow_step_2 = True

    if edit_pipe_manual:
        st.info("✍️ **แผงป้อนพารามิเตอร์ระบบท่อ:**")
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            sim_q = st.number_input("1. อัตราการไหลรวม Q (m³/s):", min_value=0.01, value=float(raw_data.get("flow_rate_q_m3s") or 60.0), step=1.0)
            sim_hst = st.number_input("2. ระยะยกน้ำสถิต Static Head (m):", value=float(raw_data.get("static_head_m") if raw_data.get("static_head_m") is not None else 5.50), step=0.5)
        with sb2:
            sim_d = st.number_input("3. ขนาดเส้นผ่านศูนย์กลางท่อ D (m):", min_value=0.05, value=float(raw_data.get("main_pipe_diameter_m") or 5.0), step=0.1)
            sim_l = st.number_input("4. ความยาวท่อ/อุโมงค์ L (m):", min_value=1.0, value=float(raw_data.get("main_pipe_length_m") or 8435.0), step=50.0)
        with sb3:
            sim_n = st.number_input("5. ค่าความขรุขระ Manning's n:", min_value=0.008, max_value=0.050, value=float(raw_data.get("manning_n") or 0.014), step=0.001, format="%.4f")
            sim_k = st.number_input("6. ผลรวมสัมประสิทธิ์ Minor Loss (ΣK):", min_value=0.0, value=float(raw_data.get("total_minor_k") or 13.5), step=0.5)

        working_data["flow_rate_q_m3s"] = sim_q
        working_data["static_head_m"] = sim_hst
        working_data["main_pipe_diameter_m"] = sim_d
        working_data["main_pipe_length_m"] = sim_l
        working_data["manning_n"] = sim_n
        working_data["total_minor_k"] = sim_k
        working_data["pipe_segments"] = [{
            "segment_name": "ท่อหลัก / อุโมงค์ส่งน้ำ (Custom Input Conduit)",
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
            formula_choice = st.selectbox("📐 เลือกสมการคำนวณ Major Loss:", ["Manning's Equation", "Darcy-Weisbach"])
        with cfg2:
            flow_unit_plot = st.selectbox("📊 หน่วยแกน X ของกราฟ:", ["m³/s", "m³/h"])

        selected_method = "Darcy" if "Darcy" in formula_choice else "Manning"
        calc_res = run_hydraulic_engine(working_data, formula_override=selected_method)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ระยะยกน้ำสถิต (Hst)", f"{calc_res['h_static_m']:.2f} m")
        m2.metric("ความเสียดทานท่อตรง (Major Loss)", f"{calc_res['total_major_loss_m']:.3f} m")
        m3.metric("การสูญเสียตามอุปกรณ์ (Minor Loss)", f"{calc_res['total_minor_loss_m']:.3f} m")
        m4.metric("เฮดรวมของระบบ (Total System Head)", f"{calc_res['total_system_head_m']:.3f} m")

        use_m3h = "m³/h" in flow_unit_plot
        scale_factor = 3600.0 if use_m3h else 1.0
        k_display = calc_res['k_system_m3s'] / (scale_factor ** 2)
        q_label = "Q (m³/h)" if use_m3h else "Q (m³/s)"
        q_design = calc_res['q_total_m3h'] if use_m3h else calc_res['q_total_m3s']

        st.info(f"🧮 **สมการ System Curve:**  `H = {k_display:.6e} · Q² + {calc_res['h_static_m']:.2f}`")

        with st.expander("📑 ดูตารางแจกแจงการคำนวณความเร็ว (V) และ Head Loss รายช่วง", expanded=True):
            st.dataframe(calc_res["breakdown_df"], use_container_width=True, hide_index=True)

        # ==========================================
        # 8. ส่วนที่ 3: เปรียบเทียบกับ PUMP CURVE
        # ==========================================
        st.divider()
        st.subheader("3️⃣ ส่วนที่ 3: เปรียบเทียบกับกราฟปั๊ม (Pump Curve vs. System Curve) และหาจุด Operating Point")

        allow_step_3 = False
        if step3_ok:
            st.success("✅ **พบข้อมูลสเปคเฮดปั๊ม/กราฟปั๊ม (Gatekeeper ส่วนที่ 3 ผ่านเกณฑ์):** พร้อมพล็อตกราฟเปรียบเทียบหาจุดตัด Operating Point")
            allow_step_3 = True
        else:
            st.warning("🔒 **ส่วนที่ 3 ถูกล็อกชั่วคราว (Gatekeeper Locked):** ไม่พบข้อมูลเฮดปั๊มที่เสนอ หรือพิกัดกราฟปั๊มในเอกสาร")
            for m3 in missing_s3:
                st.error(f"❌ ขาดข้อมูลปั๊ม: {m3}")
            force_step_3 = st.toggle("🔓 **บังคับเข้าถึงส่วนที่ 3 เพื่อกรอกสเปคเฮดปั๊มหรือพิกัดกราฟปั๊มเอง (Force Access Part 3)**", value=False)
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
                        {"Q_single_m3s": float(pt.get("q_single_m3s", 0.0)), "Head_m": float(pt.get("head_m", 0.0)), "Eff_pct": float(pt.get("eff_pct") if pt.get("eff_pct") is not None else eff_default)}
                        for pt in ai_points
                    ]
                else:
                    init_rows = [{"Q_single_m3s": round(q_unit_default, 2), "Head_m": round(h_duty_default, 2), "Eff_pct": round(eff_default, 1)}]
                st.session_state['custom_pump_points_df'] = pd.DataFrame(init_rows)

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown("##### ⚙️ รูปแบบการเดินเครื่องสูบน้ำ")
                operation_mode = st.radio("เลือกโหมดการเดินเครื่อง:", ["เดินเครื่องแบบขนาน (Parallel Operation)", "เดินเครื่องสูบน้ำเดี่ยว (Single Pump)"], index=0 if num_pumps_default > 1 else 1)
                num_active_pumps = st.number_input("จำนวนปั๊มที่เดินขนานพร้อมกัน (เครื่อง):", min_value=1, max_value=20, value=max(num_pumps_default, 2), step=1) if "ขนาน" in operation_mode else 1
                if st.button("✨ โหลดตัวอย่างพิกัดจริง 5 จุด (Exact Model Curve)", use_container_width=True):
                    st.session_state['custom_pump_points_df'] = pd.DataFrame([
                        {"Q_single_m3s": 0.0, "Head_m": round(h_duty_default * 1.34, 2), "Eff_pct": 0.0},
                        {"Q_single_m3s": round(q_unit_default * 0.6, 2), "Head_m": round(h_duty_default * 1.23, 2), "Eff_pct": 74.0},
                        {"Q_single_m3s": round(q_unit_default * 0.8, 2), "Head_m": round(h_duty_default * 1.15, 2), "Eff_pct": 82.0},
                        {"Q_single_m3s": round(q_unit_default * 1.0, 2), "Head_m": round(h_duty_default * 1.00, 2), "Eff_pct": eff_default},
                        {"Q_single_m3s": round(q_unit_default * 1.2, 2), "Head_m": round(h_duty_default * 0.79, 2), "Eff_pct": 76.0},
                    ])
                    st.rerun()

            with p_col2:
                st.markdown("##### 📝 ตารางพิกัดกราฟปั๊มต่อ 1 เครื่อง (เพิ่ม/แก้ไขแถวได้ — ใส่ ≥ 3 จุดเพื่อทำ Exact Polynomial Fit)")
                edited_pump_df = st.data_editor(
                    st.session_state['custom_pump_points_df'],
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Q_single_m3s": st.column_config.NumberColumn("Q ปั๊มเดี่ยว (m³/s)", min_value=0.0, step=0.5, format="%.2f"),
                        "Head_m": st.column_config.NumberColumn("Head ปั๊ม (m)", min_value=0.0, step=0.5, format="%.2f"),
                        "Eff_pct": st.column_config.NumberColumn("Efficiency (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
                    }
                )

            pump_eval = evaluate_pump_vs_system(df_points=edited_pump_df, num_pumps=num_active_pumps, k_sys_m3s=calc_res["k_system_m3s"], h_static_m=calc_res["h_static_m"], q_target_total_m3s=calc_res["q_total_m3s"], use_m3h=use_m3h)

            if pump_eval["is_synthesized"]:
                st.error("⚠️ **หมายเหตุสำคัญ (Synthesized Curve):** กราฟเส้นโค้งของปั๊มด้านล่างเป็น **กราฟจำลองจากการประมาณการ ($H_{shutoff} \\approx 1.30 H_{duty}$)** กรุณาใส่จุดอย่างน้อย 3 จุดเพื่อความแม่นยำ")
            else:
                st.success(f"✅ **Exact Model Curve Active:** สร้างสมการเส้นโค้งปั๊มด้วยวิธี **Polynomial Fit (กำลังสอง)** จากข้อมูลพิกัดจริงจำนวน {len(pump_eval['clean_points_df'])} จุดเรียบร้อยแล้ว")

            op1, op2, op3, op4 = st.columns(4)
            if pump_eval["q_op_plot"] is not None:
                q_op_disp = f"{pump_eval['q_op_plot']:,.2f} {'m³/h' if use_m3h else 'm³/s'}"
                h_op_disp = f"{pump_eval['h_op_m']:,.2f} m"
                eff_op_disp = f"{pump_eval['eff_op_pct']:.1f} %" if pump_eval["eff_op_pct"] is not None else "N/A"
                q_diff_m3s = pump_eval["q_op_m3s"] - calc_res["q_total_m3s"]
                flow_status = "✅ เพียงพอ (PASS)" if q_diff_m3s >= -0.05 else f"❌ ขาดไป {abs(q_diff_m3s):.2f} m³/s"
            else:
                q_op_disp, h_op_disp, eff_op_disp, flow_status = "ไม่ตัดกัน", "ไม่ตัดกัน", "N/A", "❌ ปั๊มเฮดไม่พอ"

            op1.metric("จุดตัดอัตราการไหล (Operating Q)", q_op_disp)
            op2.metric("จุดตัดเฮดทำงานจริง (Operating H)", h_op_disp)
            op3.metric("ประสิทธิภาพ ณ จุดทำงาน (Eff)", eff_op_disp)
            op4.metric("การประเมินเทียบ Q เป้าหมาย", flow_status)

            # พล็อตกราฟรวม SYSTEM CURVE vs PUMP CURVE & OPERATING POINT
            a_p, b_p, c_p = pump_eval["poly_coeffs"]
            q_max_candidates = [q_design * 1.35]
            if pump_eval["q_op_plot"]:
                q_max_candidates.append(pump_eval["q_op_plot"] * 1.25)
            q_max_plot = max(q_max_candidates) if max(q_max_candidates) > 0 else 100.0

            q_vals = np.linspace(0, q_max_plot, 120)
            h_sys_vals = (k_display * (q_vals ** 2)) + calc_res["h_static_m"]
            h_pump_parallel_vals = (a_p * (q_vals ** 2)) + (b_p * q_vals) + c_p
            h_pump_parallel_vals = np.where(h_pump_parallel_vals >= 0, h_pump_parallel_vals, np.nan)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q_vals, y=h_sys_vals, mode='lines', name='System Curve (H = KQ² + Hst)', line=dict(color='#00B4D8', width=4)))
            
            pump_curve_name = f"Pump Curve ({num_active_pumps} เครื่องเดินขนาน) - {'[จำลอง/Synthesized]' if pump_eval['is_synthesized'] else '[Exact PolyFit]'}"
            fig.add_trace(go.Scatter(x=q_vals, y=h_pump_parallel_vals, mode='lines', name=pump_curve_name, line=dict(color='#FF006E', width=4, dash='dashdot' if pump_eval['is_synthesized'] else 'solid')))

            if num_active_pumps > 1:
                q_single_vals = q_vals / num_active_pumps
                h_single_vals = (a_p * ((q_single_vals * num_active_pumps) ** 2)) + (b_p * (q_single_vals * num_active_pumps)) + c_p
                h_single_vals = np.where(h_single_vals >= 0, h_single_vals, np.nan)
                fig.add_trace(go.Scatter(x=q_single_vals, y=h_single_vals, mode='lines', name='Single Pump Curve (ปั๊มเดี่ยว 1 เครื่อง)', line=dict(color='#FB5607', width=2, dash='dot')))

            clean_pts = pump_eval["clean_points_df"]
            if not clean_pts.empty:
                fig.add_trace(go.Scatter(x=clean_pts["Q_single_m3s"] * num_active_pumps * scale_factor, y=clean_pts["Head_m"], mode='markers', name='พิกัดข้อมูลปั๊มที่ป้อน (Input Data Points)', marker=dict(color='#FFBE0B', size=9, symbol='diamond')))

            fig.add_trace(go.Scatter(x=[q_design], y=[calc_res["total_system_head_m"]], mode='markers+text', name='จุดความต้องการของระบบ (Target System Duty)', marker=dict(color='#FFD166', size=12, symbol='circle-open', line=dict(width=3)), text=[f"Target: ({q_design:,.1f}, {calc_res['total_system_head_m']:.2f} m)"], textposition="bottom right"))

            if pump_eval["q_op_plot"] is not None and pump_eval["h_op_m"] is not None:
                is_pass_op = pump_eval["q_op_m3s"] >= (calc_res["q_total_m3s"] - 0.05)
                op_color = "#06D6A0" if is_pass_op else "#EF476F"
                fig.add_trace(go.Scatter(x=[pump_eval["q_op_plot"]], y=[pump_eval["h_op_m"]], mode='markers+text', name='จุดตัดการทำงานจริง (Operating Point)', marker=dict(color=op_color, size=16, symbol='star', line=dict(color='white', width=1.5)), text=[f"Operating Point: ({pump_eval['q_op_plot']:,.1f}, {pump_eval['h_op_m']:.2f} m)"], textposition="top left"))

            fig.add_hline(y=calc_res["h_static_m"], line_dash="dash", line_color="#90E0EF", annotation_text=f"Static Head (Hst = {calc_res['h_static_m']:.2f} m)", annotation_position="bottom right")

            fig.update_layout(title=f"📈 กราฟเปรียบเทียบ System Curve vs. Pump Curve ({pump_eval['curve_type_label']})", xaxis_title=f"อัตราการไหลรวม {q_label}", yaxis_title="Head (เมตร, m)", template="plotly_dark", hovermode="x unified", height=560, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # ==========================================
            # 9. ส่วนที่ 4: พิจารณาเลือกปั๊ม กำลังมอเตอร์ และความเสี่ยง (พร้อมปุ่ม Force Access #4)
            # ==========================================
            st.divider()
            st.subheader("4️⃣ ส่วนที่ 4: พิจารณาเลือกปั๊ม กำลังมอเตอร์ ความเร็วน้ำ และประเมินความเสี่ยง (Engineering Evaluation)")

            est_eff_pump = float(pump_eval.get("eff_op_pct") or working_data.get("proposed_pump_eff_pct") or 85.0)
            est_eff_gear = float(working_data.get("proposed_gear_eff_pct") or 98.0)
            est_q_unit = max((pump_eval.get("q_op_m3s") or 0.0) / max(num_active_pumps, 1), q_unit_default)
            est_h = max(pump_eval.get("h_op_m") or 0.0, h_duty_default, calc_res["total_system_head_m"])
            est_p_gear = ((1000.0 * 9.81 * est_q_unit * est_h) / 1000.0) / (est_eff_pump / 100.0) / (est_eff_gear / 100.0)
            est_min_motor = est_p_gear * 1.05
            est_rec_motor = select_standard_motor_kw(est_min_motor)

            allow_step_4 = False
            if step4_ok:
                st.success("✅ **ข้อมูลกำลังมอเตอร์และพารามิเตอร์ทางกลครบถ้วน (Gatekeeper ส่วนที่ 4 ผ่านเกณฑ์):** แสดงผลการวิเคราะห์ความเหมาะสมครบทุกด้าน")
                allow_step_4 = True
                edit_step4_manual = st.toggle("🛠️ ปรับแต่งขนาดมอเตอร์ / ประสิทธิภาพ / NPSHr ด้วยตนเอง (Custom Mechanical Parameters Override)", value=False)
            else:
                st.warning("🔒 **ส่วนที่ 4 ถูกล็อกชั่วคราว (Gatekeeper Locked):** ขาดพารามิเตอร์ทางกลสำหรับประเมินกำลังมอเตอร์และความเสี่ยงในส่วนที่ 4 ดังนี้:")
                for m4_item in missing_s4:
                    st.error(f"❌ ขาดข้อมูลสำหรับส่วนที่ 4: {m4_item}")

                st.info(f"💡 **ข้อมูลช่วยตัดสินใจเบื้องต้นจากจุดตัดกราฟ:** ณ จุดทำงาน ระบบต้องการกำลังเพลาเกียร์ประมาณ **`{est_p_gear:,.1f} kW/เครื่อง`** เมื่อคิดเผื่อ Safety Margin 105% จะต้องการมอเตอร์ขั้นต่ำ **`{est_min_motor:,.1f} kW/เครื่อง`** (ขนาดมาตรฐานแนะนำคือ **`{est_rec_motor:,.0f} kW`**)")

                edit_step4_manual = st.toggle("🔓 **บังคับเข้าถึงส่วนที่ 4 เพื่อกรอกขนาดมอเตอร์ (kW) และพารามิเตอร์ทางกลด้วยตนเอง (Force Access Part 4)**", value=False)
                if edit_step4_manual:
                    allow_step_4 = True

            default_motor_val = float(working_data.get("proposed_motor_kw") or est_rec_motor)
            default_pump_eff = float(pump_eval.get("eff_op_pct") or working_data.get("proposed_pump_eff_pct") or 85.0)
            default_gear_eff = float(working_data.get("proposed_gear_eff_pct") or 98.0)
            default_npshr = float(working_data.get("proposed_npshr_m") or 4.5)
            margin_input = 105.0

            if edit_step4_manual:
                st.info("✍️ **แผงป้อนค่าขนาดมอเตอร์และพารามิเตอร์ทางกลด้วยตนเอง:** ลองเปลี่ยนขนาดมอเตอร์ (kW) ด้านล่างเพื่อทดสอบว่าเกิด Overload หรือไม่")
                g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
                with g_col1: motor_kw_input = st.number_input("1. ขนาดมอเตอร์ที่ต้องการทดสอบ (kW):", min_value=1.0, max_value=20000.0, value=default_motor_val, step=50.0)
                with g_col2: pump_eff_input = st.number_input("2. ประสิทธิภาพปั๊ม ณ จุดทำงาน (%):", min_value=10.0, max_value=98.0, value=min(max(default_pump_eff, 10.0), 98.0), step=1.0)
                with g_col3: gear_eff_input = st.number_input("3. ประสิทธิภาพเกียร์ทดรอบ (%):", min_value=50.0, max_value=100.0, value=default_gear_eff, step=0.5)
                with g_col4: margin_input = st.number_input("4. Safety Margin มอเตอร์ (%):", min_value=100.0, max_value=150.0, value=105.0, step=1.0)
                with g_col5: npshr_input = st.number_input("5. ค่า NPSHr ของปั๊ม (m):", min_value=0.5, max_value=25.0, value=default_npshr, step=0.5)
            else:
                motor_kw_input = working_data.get("proposed_motor_kw")
                pump_eff_input = default_pump_eff
                gear_eff_input = default_gear_eff
                npshr_input = default_npshr

            if allow_step_4:
                phase4_res = evaluate_phase4_full(
                    working_data=working_data, calc_res=calc_res, pump_eval=pump_eval, num_active_pumps=num_active_pumps,
                    pump_eff_pct=pump_eff_input, gear_eff_pct=gear_eff_input, safety_margin_pct=margin_input,
                    npshr_m=npshr_input, user_motor_kw=motor_kw_input, is_force_unlocked=(not step4_ok and edit_step4_manual),
                    missing_step4_list=missing_s4
                )

                p4_c1, p4_c2, p4_c3, p4_c4 = st.columns(4)
                p4_c1.metric("กำลังเพลาปั๊ม (Pump Shaft Power)", f"{phase4_res['p_pump_shaft_kw']:,.1f} kW/เครื่อง")
                p4_c2.metric("กำลังเพลาเกียร์ (Gear Input Power)", f"{phase4_res['p_gear_shaft_kw']:,.1f} kW/เครื่อง")
                p4_c3.metric(f"มอเตอร์ขั้นต่ำ (เผื่อ {margin_input:.0f}%)", f"{phase4_res['p_min_motor_kw']:,.1f} kW/เครื่อง")
                p4_c4.metric("การประเมินขนาดมอเตอร์", phase4_res["power_status"])

                st.markdown("#### 📝 คำอธิบายพฤติกรรมบนกราฟและผลการพิจารณาความเหมาะสม (Engineering Evaluation Narrative)")

                with st.container(border=True):
                    st.markdown(f"**1. รูปแบบการเดินเครื่องและสิ่งที่ปรากฏบนกราฟ ({phase4_res['mode_title']}):**")
                    st.write(phase4_res["graph_narrative"])

                    st.markdown("---")
                    st.markdown(f"**2. ความเพียงพอของอัตราการไหล (Q) และเฮด (H) — [{phase4_res['qh_status']}]:**")
                    if phase4_res["qh_pass"]: st.success(phase4_res["qh_narrative"])
                    else: st.error(phase4_res["qh_narrative"])

                    st.markdown("---")
                    st.markdown(f"**3. ประสิทธิภาพ กำลังเพลา และความเหมาะสมของขนาดมอเตอร์ (Efficiency / Power / Margin) — [{phase4_res['power_status']}]:**")
                    if phase4_res["proposed_motor_kw"] is None: st.info(phase4_res["power_narrative"])
                    elif phase4_res["power_pass"]: st.success(phase4_res["power_narrative"])
                    else: st.error(phase4_res["power_narrative"])

                    st.markdown("---")
                    st.markdown(f"**4. ความเหมาะสมของขนาดท่อ แนวท่อ และความเร็วน้ำ (Pipe Sizing & Velocity Check) — [{phase4_res['pipe_status']}]:**")
                    st.write(phase4_res["pipe_narrative"])
                    for w in phase4_res["vel_warnings"]: st.warning(w)
                    with st.expander("🔍 ดูรายละเอียดความเร็วน้ำในแต่ละช่วงท่อที่ผ่านเกณฑ์", expanded=False):
                        for d_item in phase4_res["vel_details"]: st.write(d_item)

                    st.markdown("---")
                    st.markdown("**5. การแจ้งเตือนจุดเสี่ยง ผลกระทบทางชลศาสตร์ และค่า NPSH / Cavitation Flags:**")
                    for flag in phase4_res["risk_flags"]:
                        if flag.startswith("🚨"): st.error(flag)
                        elif flag.startswith("⚠️"): st.warning(flag)
                        elif flag.startswith("ℹ️"): st.info(flag)
                        else: st.success(flag)

                st.session_state['hydraulic_results'] = calc_res
                st.session_state['pump_eval_results'] = pump_eval
                st.session_state['phase4_results'] = phase4_res
