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
st.caption("จาก Input Parameters → คำนวณ Head Loss & System Curve → เปรียบเทียบ Pump Curve หาจุดตัด Operating Point")

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================================
# 1. ฟังก์ชันช่วยเหลือ & ตรวจสอบความครบถ้วน (HELPERS & GATEKEEPER)
# ==========================================
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

def check_data_completeness(data):
    """ตรวจสอบอย่างเข้มงวดว่าข้อมูลเพียงพอสำหรับคำนวณ Head Loss และสร้าง System Curve หรือไม่"""
    reasons = list(data.get("missing_critical_data") or [])

    q_val = data.get("flow_rate_q_m3s")
    h_st = data.get("static_head_m")
    main_l = data.get("main_pipe_length_m")
    main_d = data.get("main_pipe_diameter_m")
    segments = data.get("pipe_segments") or []

    has_valid_segment = False
    for seg in segments:
        seg_d = seg.get("diameter_m") or seg.get("area_m2")
        seg_l = float(seg.get("length_m") or 0.0)
        seg_k = float(seg.get("minor_loss_k_sum") or 0.0)
        seg_fixed = float(seg.get("fixed_minor_loss_m") or 0.0)
        if seg_d and (seg_l > 0 or seg_k > 0 or seg_fixed > 0):
            has_valid_segment = True
            break

    has_valid_main = bool(main_d and float(main_d) > 0 and main_l and float(main_l) > 0)

    if not q_val or float(q_val) <= 0:
        if not any("อัตราการไหล" in r or "Q" in r for r in reasons):
            reasons.append("อัตราการไหลที่ต้องการ (Flow Rate Q)")
    if h_st is None:
        if not any("Static" in r or "ระดับน้ำ" in r or "ระยะยก" in r for r in reasons):
            reasons.append("ระยะยกน้ำสถิต (Static Head) หรือระดับน้ำต้นทาง-ปลายทาง")
    if not (has_valid_segment or has_valid_main):
        if not any("ความยาว" in r for r in reasons):
            reasons.append("ความยาวของท่อหรืออุโมงค์ (Pipe/Tunnel Length > 0 m) และขนาดท่อ")

    is_complete = (len(reasons) == 0) and (q_val is not None and float(q_val) > 0) and (h_st is not None) and (has_valid_segment or has_valid_main)
    return is_complete, reasons

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
    """
    สร้างสมการ Pump Curve (Exact Polynomial Fit หรือ Synthesized Curve)
    พร้อมคำนวณจุดตัด Operating Point (Q_op, H_op) เทียบกับ System Curve
    """
    scale = 3600.0 if use_m3h else 1.0
    k_sys = k_sys_m3s / (scale ** 2)
    q_target_plot = q_target_total_m3s * scale

    # กรองเฉพาะแถวที่มีตัวเลขครบ
    clean_df = df_points.dropna(subset=["Q_single_m3s", "Head_m"]).copy()
    clean_df = clean_df[clean_df["Head_m"] > 0].sort_values("Q_single_m3s")

    is_synthesized = len(clean_df) < 3

    if not is_synthesized:
        # กรณีที่ 1: มีข้อมูลมากกว่าหรือเท่ากับ 3 จุด -> ใช้ Polynomial Fit ดีกรี 2 (H = a*Q_total^2 + b*Q_total + c)
        q_single_arr = clean_df["Q_single_m3s"].values.astype(float)
        q_total_arr = q_single_arr * num_pumps * scale
        h_arr = clean_df["Head_m"].values.astype(float)

        coeffs = np.polyfit(q_total_arr, h_arr, 2)
        a_p, b_p, c_p = coeffs[0], coeffs[1], coeffs[2]
        curve_type_label = "Exact Model Curve (Polynomial Fit จากข้อมูลจริง)"

        # ฟิตกราฟ Efficiency ถ้ามีข้อมูล
        if "Eff_pct" in clean_df.columns and clean_df["Eff_pct"].dropna().count() >= 3:
            eff_Coef = np.polyfit(q_total_arr, clean_df["Eff_pct"].fillna(0).values.astype(float), 2)
        else:
            eff_Coef = None
    else:
        # กรณีที่ 2: มีข้อมูลน้อยกว่า 3 จุด -> สร้างกราฟจำลอง (Synthesized / Approximated Curve)
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
        # สมการปั๊มจำลองมาตรฐาน: H_shutoff = 1.30 * H_duty -> H(Q) = H_shutoff - c * Q^2
        h_shutoff = 1.30 * h_duty
        a_p = -0.30 * h_duty / (q_duty_total_plot ** 2)
        b_p = 0.0
        c_p = h_shutoff
        curve_type_label = "Synthesized / Approximated Curve (กราฟจำลองประมาณการ)"

        # จำลองพาราโบลา Efficiency โดยมีจุดสูงสุด (BEP) อยู่ที่จุด Duty
        eff_Coef = [-eff_duty / (q_duty_total_plot ** 2), (2.0 * eff_duty) / q_duty_total_plot, 0.0]

    # หาจุดตัด Operating Point: a_p*Q^2 + b_p*Q + c_p = k_sys*Q^2 + H_st
    # จัดรูปสมการกำลังสอง: (a_p - k_sys)*Q^2 + b_p*Q + (c_p - H_st) = 0
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
                # เลือกค่าบวกที่ใกล้เคียงช่วงทำงานจริง
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
# 4. โครงสร้างคำสั่ง AI (SCHEMA รองรับการดึงจุด PUMP CURVE)
# ==========================================
extraction_tool = [{
    "type": "function",
    "function": {
        "name": "extract_hydraulic_parameters",
        "description": "สกัดพารามิเตอร์ระบบท่อ อุโมงค์ ท่อไซฟอน สเปคปั๊ม และพิกัดจุดบนกราฟ Pump Curve",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "flow_rate_q_m3s": {"type": "number", "description": "อัตราการไหลรวมที่ต้องการของระบบในหน่วย m3/s"},
                "flow_rate_original_text": {"type": "string"},
                "num_parallel_pumps": {"type": "integer", "description": "จำนวนเครื่องสูบน้ำที่เดินขนานกัน (ถ้าเป็นปั๊มเดี่ยวใส่ 1)"},
                "static_head_m": {"type": "number", "description": "ระยะยกน้ำสถิต (Static Head / Hst) หน่วยเมตร ห้ามเดาเอง"},
                "main_pipe_length_m": {"type": "number", "description": "ความยาวรวมของท่อหรืออุโมงค์หลัก (m) ห้ามเดาเอง"},
                "main_pipe_diameter_m": {"type": "number", "description": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์หลัก (m)"},
                "pipe_material_or_roughness": {"type": "string"},
                "manning_n": {"type": "number"},
                "darcy_f": {"type": "number"},
                "system_type": {"type": "string"},
                "proposed_pump_head_m": {"type": "number", "description": "ค่า Operating Head หรือ Rated Head ของปั๊มที่ผู้รับจ้างเสนอ (m)"},
                "proposed_pump_shutoff_head_m": {"type": "number", "description": "ค่า Shutoff Head (Head ที่ Q=0) ของปั๊ม (m) ถ้ามีระบุ"},
                "proposed_pump_eff_pct": {"type": "number", "description": "ประสิทธิภาพปั๊มที่จุดออกแบบ (%) ถ้ามี"},
                "proposed_motor_kw": {"type": "number", "description": "ขนาดกำลังมอเตอร์ที่เสนอ (kW) ถ้ามี"},
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
                    "description": "พิกัดจุด (Q, H, Efficiency) บนกราฟ Pump Curve ของปั๊ม 1 เครื่อง (หากอ่านได้จากตารางหรือรูปภาพกราฟ)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "q_single_m3s": {"type": "number", "description": "อัตราการไหลของปั๊ม 1 เครื่อง (m3/s)"},
                            "head_m": {"type": "number", "description": "ค่า Head ของปั๊มที่จุดนั้น (m)"},
                            "eff_pct": {"type": "number", "description": "ค่าประสิทธิภาพปั๊มที่จุดนั้น (%) ถ้ามี"}
                        },
                        "required": ["q_single_m3s", "head_m"]
                    }
                },
                "pump_curve_confidence": {
                    "type": "string",
                    "enum": ["high_multi_point", "single_duty_point_only", "uncertain_from_pdf_or_blurry", "no_pump_data"],
                    "description": "ระดับความมั่นใจของ AI ในการอ่านกราฟ Pump Curve: เลือก 'high_multi_point' เมื่อพบตารางหรือกราฟที่มีตั้งแต่ 3 จุดขึ้นไปชัดเจน, เลือก 'single_duty_point_only' เมื่อพบแค่จุดใช้งานจุดเดียว, เลือก 'uncertain_from_pdf_or_blurry' เมื่อมาจากไฟล์ PDF ที่ไม่มีตารางจุดหรือภาพไม่ชัด"
                },
                "ai_curve_advice_th": {
                    "type": "string",
                    "description": "คำแนะนำภาษาไทยจาก AI กรณีไม่เห็นกราฟปั๊มชัดเจน เช่น แนะนำให้อัปโหลดรูปภาพกราฟ Pump Curve หรือบอกจุด Shutoff (Q=0), จุด BEP และจุด Runout"
                },
                "missing_critical_data": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["missing_critical_data", "pump_curve_confidence"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, ตารางคำนวณ, รูปภาพกราฟ หรือข้อความจากหน้างาน และสกัดพารามิเตอร์ทางชลศาสตร์ออกมา

กฎเหล็ก (Operational Rules):
1. สื่อสารเป็น "ภาษาไทย" เสมอ ห้ามเดาตัวเลขระบบท่อ (ความยาว, ขนาดท่อ, Static Head) หากไม่มีให้ใส่ null และระบุใน `missing_critical_data`
2. การสกัดข้อมูลกราฟปั๊ม (`pump_curve_points`):
   - หากมีตารางพิกัดกราฟปั๊ม หรือมีรูปภาพกราฟ Pump Curve ที่อ่านค่าจุด (Q, H) ได้ชัดเจนตั้งแต่ 3 จุดขึ้นไป ให้ดึงทุกจุดใส่ลงใน `pump_curve_points` (แปลง Q ให้เป็นอัตราไหลต่อ 1 เครื่องในหน่วย m3/s) และตั้งค่า `pump_curve_confidence` เป็น `"high_multi_point"`
   - หากในเอกสารหรือไฟล์ PDF มีเฉพาะตัวเลขจุดออกแบบจุดเดียว (เช่น บอกแค่ Q=10 m3/s, H=23 m) หรือเป็นไฟล์ PDF ที่ไม่มีตารางพิกัดกราฟ ให้ใส่จุดที่มีลงใน `pump_curve_points` แล้วตั้งค่า `pump_curve_confidence` เป็น `"single_duty_point_only"` หรือ `"uncertain_from_pdf_or_blurry"`
   - เมื่อความมั่นใจไม่ใช่ `"high_multi_point"` ให้เขียนข้อความใน `ai_curve_advice_th` แนะนำผู้ใช้งานอย่างสุภาพว่า: เพื่อให้ได้กราฟ Pump Curve ที่แม่นยำตามรุ่นจริง แนะนำให้ถ่ายรูปหรือแคปภาพหน้ากราฟ Pump Curve (JPG/PNG) มาอัปโหลด หรือระบุจุดสำคัญอย่างน้อย 3 จุด ได้แก่ 1) จุด Shutoff Head (ที่ Q=0), 2) จุดออกแบบ (Design/BEP Point), และ 3) จุดปลายกราฟ (Max Flow / Runout Point)
"""

# ==========================================
# 5. ส่วนแสดงผล STEP 1: นำเข้าและสกัดข้อมูล
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
            # ล้างตารางจุดปั๊มเดิมเมื่อมีการสกัดข้อมูลใหม่
            if 'custom_pump_points_df' in st.session_state:
                del st.session_state['custom_pump_points_df']
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์, ถ่ายรูป, หรือพิมพ์ข้อความก่อนกดปุ่มวิเคราะห์ครับ")

# ==========================================
# แสดงผลการ์ดพารามิเตอร์ & ระบบล็อก GATEKEEPER
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

    with st.expander("🔧 ดูรายการอุปกรณ์ประกอบ (Minor Loss Fittings) และลักษณะระบบ", expanded=False):
        st.write(f"**ลักษณะระบบ:** {format_metric(raw_data.get('system_type'), '')} | **จำนวนปั๊มเดินขนาน:** {format_metric(raw_data.get('num_parallel_pumps'), 'เครื่อง', 0)}")
        fittings = raw_data.get('minor_loss_fittings') or []
        if fittings:
            for item in fittings:
                st.write(f"- {item}")
        else:
            st.write("- ไม่พบข้อมูลอุปกรณ์ประกอบ")

    is_complete, missing_reasons = check_data_completeness(raw_data)
    working_data = copy.deepcopy(raw_data)
    allow_step_2 = False

    if is_complete:
        st.success("✅ **ข้อมูลระบบท่อครบถ้วน (Gatekeeper Passed):** ระบบพร้อมคำนวณ Head Loss และเปรียบเทียบกราฟ Pump Curve")
        allow_step_2 = True
    else:
        st.warning("🛑 **ระงับการคำนวณชั่วคราว (Gatekeeper Alert):** ข้อมูลไม่เพียงพอสำหรับการคำนวณ Head Loss และสร้างสมการ System Curve")
        for m in missing_reasons:
            st.error(f"❌ ขาดข้อมูล: {m}")

        st.markdown("---")
        force_proceed = st.toggle(
            "🔓 **บังคับเข้าสู่โหมดจำลองข้อมูลเอง (Force Proceed / Sandbox Mode)** — *เปิดสวิตช์นี้หากต้องการสมมติตัวเลขที่ขาดหายไปเพื่อทดลองพล็อตกราฟ*",
            value=False
        )

        if force_proceed:
            st.info("🛠️ **โหมดจำลองพารามิเตอร์ (Sandbox Mode):** ปรับแต่งตัวเลขด้านล่างเพื่อจำลอง System Curve และทดสอบกราฟปั๊ม")
            sb1, sb2, sb3 = st.columns(3)
            with sb1:
                sim_q = st.number_input("1. อัตราการไหลรวม Q (m³/s):", min_value=0.01, value=float(raw_data.get("flow_rate_q_m3s") or 60.0), step=1.0)
                sim_hst = st.number_input("2. ระยะยกน้ำสถิต Static Head (m):", value=float(raw_data.get("static_head_m") if raw_data.get("static_head_m") is not None else 5.50), step=0.5)
            with sb2:
                sim_d = st.number_input("3. ขนาดเส้นผ่านศูนย์กลางท่อ D (m):", min_value=0.05, value=float(raw_data.get("main_pipe_diameter_m") or 5.0), step=0.1)
                sim_l = st.number_input("4. ความยาวท่อ/อุโมงค์ L (m):", min_value=1.0, value=float(raw_data.get("main_pipe_length_m") or 8435.0), step=50.0)
            with sb3:
                sim_n = st.number_input("5. ค่าความขรุขระ Manning's n:", min_value=0.008, max_value=0.050, value=float(raw_data.get("manning_n") or 0.014), step=0.001, format="%.4f")
                sim_k = st.number_input("6. ผลรวมสัมประสิทธิ์ Minor Loss (ΣK):", min_value=0.0, value=float(raw_data.get("total_minor_k") or 28.5), step=0.5)

            working_data["flow_rate_q_m3s"] = sim_q
            working_data["static_head_m"] = sim_hst
            working_data["main_pipe_diameter_m"] = sim_d
            working_data["main_pipe_length_m"] = sim_l
            working_data["manning_n"] = sim_n
            working_data["total_minor_k"] = sim_k
            working_data["pipe_segments"] = [{
                "segment_name": "ท่อ/อุโมงค์จำลอง (Simulated Conduit)",
                "carries_single_pump_flow": False,
                "diameter_m": sim_d,
                "length_m": sim_l,
                "manning_n": sim_n,
                "darcy_f": float(raw_data.get("darcy_f") or 0.015),
                "minor_loss_k_sum": sim_k,
                "fixed_minor_loss_m": 0.0
            }]
            allow_step_2 = True

    # ==========================================
    # 6. ส่วนแสดงผล STEP 2: คำนวณ HEAD LOSS & SYSTEM CURVE
    # ==========================================
    if allow_step_2:
        st.divider()
        st.subheader("2️⃣ ส่วนที่ 2: คำนวณ Head Loss และสร้างสมการ System Curve (H = KQ² + Hst)")

        cfg1, cfg2 = st.columns(2)
        with cfg1:
            formula_choice = st.selectbox(
                "📐 เลือกสมการคำนวณความเสียดทานท่อตรง (Major Loss):",
                ["Manning's Equation (งานอุโมงค์/ท่อระบายน้ำ)", "Darcy-Weisbach (งานท่อแรงดันมาตรฐาน)"]
            )
        with cfg2:
            flow_unit_plot = st.selectbox(
                "📊 หน่วยอัตราการไหลบนแกน X ของกราฟ:",
                ["m³/s (ลูกบาศก์เมตรต่อวินาที)", "m³/h (ลูกบาศก์เมตรต่อชั่วโมง)"]
            )

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

        eq_str = (
            f"H = ({k_display:.4e}) · Q² + {calc_res['h_static_m']:.2f}"
            if use_m3h else
            f"H = {k_display:.6f} · Q² + {calc_res['h_static_m']:.2f}"
        )
        st.info(f"🧮 **สมการ System Curve ของระบบ:**  `{eq_str}`   *(โดยที่ Dynamic Head Loss รวม = `{calc_res['total_dynamic_loss_m']:.3f} m` ที่ Q = `{q_design:,.1f}`)*")

        with st.expander("📑 ดูตารางแจกแจงการคำนวณความเร็ว (V) และ Head Loss รายช่วง", expanded=False):
            st.dataframe(calc_res["breakdown_df"], use_container_width=True, hide_index=True)

        # ==========================================
        # 7. ส่วนแสดงผล STEP 3: เปรียบเทียบกับ PUMP CURVE และหาจุดตัด OPERATING POINT
        # ==========================================
        st.divider()
        st.subheader("3️⃣ ส่วนที่ 3: เปรียบเทียบกับกราฟปั๊ม (Pump Curve vs. System Curve) และหาจุด Operating Point")

        num_pumps_default = int(working_data.get("num_parallel_pumps") or 1)
        q_unit_default = calc_res["q_total_m3s"] / max(num_pumps_default, 1)
        h_duty_default = float(working_data.get("proposed_pump_head_m") or calc_res["total_system_head_m"] or 23.0)
        eff_default = float(working_data.get("proposed_pump_eff_pct") or 85.0)

        # เตรียมตารางจุดพิกัด Pump Curve เริ่มต้นจากที่ AI สกัดได้
        if 'custom_pump_points_df' not in st.session_state:
            ai_points = working_data.get("pump_curve_points") or []
            if len(ai_points) >= 1:
                init_rows = [
                    {
                        "Q_single_m3s": float(pt.get("q_single_m3s", 0.0)),
                        "Head_m": float(pt.get("head_m", 0.0)),
                        "Eff_pct": float(pt.get("eff_pct") or eff_default)
                    }
                    for pt in ai_points
                ]
            else:
                # ถ้ามีแค่ค่า Duty Head เดี่ยวๆ ให้ใส่จุดเดียวไว้ก่อน เพื่อให้ระบบแจ้งเตือนว่าเป็นกราฟจำลอง
                init_rows = [
                    {"Q_single_m3s": round(q_unit_default, 2), "Head_m": round(h_duty_default, 2), "Eff_pct": round(eff_default, 1)}
                ]
            st.session_state['custom_pump_points_df'] = pd.DataFrame(init_rows)

        # ตรวจสอบความมั่นใจของ AI ในการอ่านกราฟจาก PDF / รูปภาพ
        curve_conf = working_data.get("pump_curve_confidence", "single_duty_point_only")
        ai_advice = working_data.get("ai_curve_advice_th")

        if curve_conf != "high_multi_point" or len(st.session_state['custom_pump_points_df']) < 3:
            st.warning(
                "📸 **คำแนะนำจาก AI (ไม่พบพิกัดกราฟปั๊มแบบหลายจุดจากไฟล์เอกสาร/PDF):**\n"
                "- การอ่านไฟล์ PDF ด้วยข้อความอาจทำให้ AI มองไม่เห็นเส้นกราฟ Pump Curve หรือพบเพียงจุดใช้งานจุดเดียว (Duty Point)\n"
                "- **วิธีเพิ่มความแม่นยำ:** คุณสามารถ **แคปหน้าจอกราฟ Pump Curve เป็นไฟล์รูปภาพ (PNG/JPG) แล้วอัปโหลดในส่วนที่ 1** เพื่อให้ AI Vision อ่านเส้นกราฟ หรือ **พิมพ์เพิ่มจุดสำคัญจากกราฟอย่างน้อย 3 จุด** (เช่น จุด `Q=0 Shutoff`, จุด `Design/BEP`, และจุด `Max Flow/Runout`) ลงในตารางด้านล่างได้โดยตรงครับ"
            )
            if ai_advice:
                st.caption(f"💬 *หมายเหตุจาก AI: {ai_advice}*")

        # แผงควบคุมการเดินเครื่องขนาน และตารางแก้ไขจุดพิกัด Pump Curve
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.markdown("##### ⚙️ รูปแบบการเดินเครื่องสูบน้ำ")
            operation_mode = st.radio(
                "เลือกโหมดการเดินเครื่อง:",
                ["เดินเครื่องแบบขนาน (Parallel Operation)", "เดินเครื่องสูบน้ำเดี่ยว (Single Pump)"],
                index=0 if num_pumps_default > 1 else 1
            )
            if "ขนาน" in operation_mode:
                num_active_pumps = st.number_input(
                    "จำนวนปั๊มที่เดินขนานพร้อมกัน (เครื่อง):",
                    min_value=1, max_value=20, value=max(num_pumps_default, 2), step=1
                )
            else:
                num_active_pumps = 1

            # ปุ่มตัวช่วยเติมจุดมาตรฐาน 5 จุดจากแคตตาล็อกโครงการบึงหนองบิน (เผื่อทดสอบ Exact Polynomial Fit ทันที)
            if st.button("✨ โหลดตัวอย่างพิกัดจริง 5 จุด (Exact Model Curve)", use_container_width=True):
                st.session_state['custom_pump_points_df'] = pd.DataFrame([
                    {"Q_single_m3s": 0.0,  "Head_m": round(h_duty_default * 1.34, 2), "Eff_pct": 0.0},
                    {"Q_single_m3s": round(q_unit_default * 0.6, 2), "Head_m": round(h_duty_default * 1.23, 2), "Eff_pct": 74.0},
                    {"Q_single_m3s": round(q_unit_default * 0.8, 2), "Head_m": round(h_duty_default * 1.15, 2), "Eff_pct": 82.0},
                    {"Q_single_m3s": round(q_unit_default * 1.0, 2), "Head_m": round(h_duty_default * 1.00, 2), "Eff_pct": eff_default},
                    {"Q_single_m3s": round(q_unit_default * 1.2, 2), "Head_m": round(h_duty_default * 0.79, 2), "Eff_pct": 76.0},
                ])
                st.rerun()

        with p_col2:
            st.markdown("##### 📝 ตารางพิกัดกราฟปั๊มต่อ 1 เครื่อง (เพิ่ม/ลบแถวได้ — ใส่ ≥ 3 จุดเพื่อทำ Exact Polynomial Fit)")
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

        # คำนวณสมการ Pump Curve และหาจุดตัด Operating Point
        pump_eval = evaluate_pump_vs_system(
            df_points=edited_pump_df,
            num_pumps=num_active_pumps,
            k_sys_m3s=calc_res["k_system_m3s"],
            h_static_m=calc_res["h_static_m"],
            q_target_total_m3s=calc_res["q_total_m3s"],
            use_m3h=use_m3h
        )

        # แจ้งเตือนชัดเจนว่าเป็นกราฟจำลอง (Synthesized) หรือกราฟจริง (Exact Model Fit)
        if pump_eval["is_synthesized"]:
            st.error(
                "⚠️ **หมายเหตุสำคัญ (Synthesized / Approximated Curve):** เนื่องจากมีข้อมูลพิกัดปั๊มน้อยกว่า 3 จุด "
                "กราฟเส้นโค้งของปั๊มด้านล่างจึงเป็น **กราฟจำลองจากการประมาณการทางวิศวกรรม ($H_{shutoff} \\approx 1.30 H_{duty}$)** "
                "หากต้องการผลลัพธ์จุดตัด Operating Point ที่แม่นยำตามรุ่นจริง กรุณาอัปโหลดรูปภาพกราฟ Pump Curve หรือกรอกพิกัด (Q, H) ในตารางด้านบนให้ครบอย่างน้อย 3 จุดครับ"
            )
        else:
            st.success(
                f"✅ **Exact Model Curve Active:** สร้างสมการเส้นโค้งปั๊มด้วยวิธี **Polynomial Fit (กำลังสอง)** จากข้อมูลพิกัดจริงจำนวน {len(pump_eval['clean_points_df'])} จุดเรียบร้อยแล้ว"
            )

        # แสดงการ์ดสรุปจุดตัด Operating Point
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

        # ==========================================
        # พล็อตกราฟรวม SYSTEM CURVE vs PUMP CURVE & OPERATING POINT
        # ==========================================
        a_p, b_p, c_p = pump_eval["poly_coeffs"]
        q_max_candidates = [q_design * 1.35]
        if pump_eval["q_op_plot"]:
            q_max_candidates.append(pump_eval["q_op_plot"] * 1.25)
        q_max_plot = max(q_max_candidates) if max(q_max_candidates) > 0 else 100.0

        q_vals = np.linspace(0, q_max_plot, 120)
        h_sys_vals = (k_display * (q_vals ** 2)) + calc_res["h_static_m"]

        # คำนวณเส้นกราฟปั๊มรวม (N เครื่องเดินขนาน) และเส้นปั๊มเดี่ยว (1 เครื่อง)
        h_pump_parallel_vals = (a_p * (q_vals ** 2)) + (b_p * q_vals) + c_p
        h_pump_parallel_vals = np.where(h_pump_parallel_vals >= 0, h_pump_parallel_vals, np.nan)

        fig = go.Figure()

        # 1. เส้น System Curve (H = KQ^2 + Hst)
        fig.add_trace(go.Scatter(
            x=q_vals, y=h_sys_vals,
            mode='lines',
            name='System Curve (H = KQ² + Hst)',
            line=dict(color='#00B4D8', width=4)
        ))

        # 2. เส้น Pump Curve (กรณีเดินขนาน N เครื่อง หรือปั๊มเดี่ยว)
        pump_curve_name = (
            f"Pump Curve ({num_active_pumps} เครื่องเดินขนาน) - {'[จำลอง/Synthesized]' if pump_eval['is_synthesized'] else '[Exact PolyFit]'}"
        )
        fig.add_trace(go.Scatter(
            x=q_vals, y=h_pump_parallel_vals,
            mode='lines',
            name=pump_curve_name,
            line=dict(color='#FF006E', width=4, dash='dashdot' if pump_eval['is_synthesized'] else 'solid')
        ))

        # 3. หากเดินขนานมากกว่า 1 เครื่อง ให้พล็อตเส้นปั๊มเดี่ยว (1 เครื่อง) เป็นเส้นประจางๆ เพื่อเปรียบเทียบ
        if num_active_pumps > 1:
            q_single_vals = q_vals / num_active_pumps
            h_single_vals = (a_p * ((q_single_vals * num_active_pumps) ** 2)) + (b_p * (q_single_vals * num_active_pumps)) + c_p
            h_single_vals = np.where(h_single_vals >= 0, h_single_vals, np.nan)
            fig.add_trace(go.Scatter(
                x=q_single_vals, y=h_single_vals,
                mode='lines',
                name='Single Pump Curve (ปั๊มเดี่ยว 1 เครื่อง)',
                line=dict(color='#FB5607', width=2, dash='dot')
            ))

        # 4. พล็อตจุดพิกัดจริงจากตาราง (คูณจำนวนเครื่องเดินขนานแล้ว)
        clean_pts = pump_eval["clean_points_df"]
        if not clean_pts.empty:
            fig.add_trace(go.Scatter(
                x=clean_pts["Q_single_m3s"] * num_active_pumps * scale_factor,
                y=clean_pts["Head_m"],
                mode='markers',
                name='พิกัดข้อมูลปั๊มที่ป้อน (Input Data Points)',
                marker=dict(color='#FFBE0B', size=9, symbol='diamond')
            ))

        # 5. จุดความต้องการของระบบที่อัตราการไหลเป้าหมาย (Target System Duty)
        fig.add_trace(go.Scatter(
            x=[q_design],
            y=[calc_res["total_system_head_m"]],
            mode='markers+text',
            name='จุดความต้องการของระบบ (Target System Duty)',
            marker=dict(color='#FFD166', size=12, symbol='circle-open', line=dict(width=3)),
            text=[f"Target: ({q_design:,.1f}, {calc_res['total_system_head_m']:.2f} m)"],
            textposition="bottom right"
        ))

        # 6. จุดตัดการทำงานจริง (Operating Point Intersection)
        if pump_eval["q_op_plot"] is not None and pump_eval["h_op_m"] is not None:
            is_pass_op = pump_eval["q_op_m3s"] >= (calc_res["q_total_m3s"] - 0.05)
            op_color = "#06D6A0" if is_pass_op else "#EF476F"
            fig.add_trace(go.Scatter(
                x=[pump_eval["q_op_plot"]],
                y=[pump_eval["h_op_m"]],
                mode='markers+text',
                name='จุดตัดการทำงานจริง (Operating Point)',
                marker=dict(color=op_color, size=16, symbol='star', line=dict(color='white', width=1.5)),
                text=[f"Operating Point: ({pump_eval['q_op_plot']:,.1f}, {pump_eval['h_op_m']:.2f} m)"],
                textposition="top left"
            ))

        # เส้นระดับ Static Head
        fig.add_hline(
            y=calc_res["h_static_m"],
            line_dash="dash",
            line_color="#90E0EF",
            annotation_text=f"Static Head (Hst = {calc_res['h_static_m']:.2f} m)",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title=f"📈 กราฟเปรียบเทียบ System Curve vs. Pump Curve ({pump_eval['curve_type_label']})",
            xaxis_title=f"อัตราการไหลรวม {q_label}",
            yaxis_title="Head (เมตร, m)",
            template="plotly_dark",
            hovermode="x unified",
            height=560,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # บันทึกผลลัพธ์ไว้สำหรับ Step 4 & 5 (สรุปความเหมาะสม มอเตอร์ NPSH และรายงาน Integrated)
        st.session_state['hydraulic_results'] = calc_res
        st.session_state['pump_eval_results'] = pump_eval
