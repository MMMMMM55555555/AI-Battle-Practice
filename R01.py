import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
from openai import OpenAI
import json
import base64
import math
import plotly.graph_objects as go

# --- ตั้งค่าหน้าจอ (PAGE CONFIG) ---
st.set_page_config(page_title="AI Pump & System Curve Validator", layout="wide")
st.title("🤖 ระบบ AI ตรวจสอบความเหมาะสมเครื่องสูบน้ำและระบบอุโมงค์")
st.caption("รองรับตั้งแต่งานระบบท่อทั่วไป จนถึงระบบอุโมงค์ระบายน้ำและท่อไซฟอนขนาดใหญ่ (ตามมาตรฐานวิศวกรรม)")

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================================
# 1. ฟังก์ชันช่วยเหลือ & จัดการแสดงผลหน่วย (UI HELPERS)
# ==========================================
def format_metric(val, unit="", decimals=2):
    """ฟังก์ชันแก้ปัญหา 'ไม่พบข้อมูล m' - จะแสดงหน่วยก็ต่อเมื่อมีตัวเลขจริงเท่านั้น"""
    if val is None or val == "" or str(val).strip().lower() in ["none", "null", "n/a", "ไม่พบข้อมูล"]:
        return "ไม่พบข้อมูล"
    try:
        num = float(val)
        # ถ้าน้อยกว่า 1 และไม่ใช่ 0 ให้โชว์ทศนิยมละเอียดขึ้น (เช่น Manning n = 0.014)
        if 0 < abs(num) < 0.1 and decimals == 2:
            formatted = f"{num:.4f}"
        elif num.is_integer():
            formatted = f"{int(num):,}"
        else:
            formatted = f"{num:,.{decimals}f}"
        return f"{formatted} {unit}".strip()
    except (ValueError, TypeError):
        return f"{val} {unit}".strip() if unit else str(val)

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
    """
    คำนวณ Major Loss, Minor Loss, ความเร็ว V และสร้างสมการ H = K*Q^2 + Hst
    รองรับทั้งท่อเดี่ยวทั่วไป และระบบหลายช่วง (เช่น อุโมงค์ดูด + ไซฟอน + อุโมงค์ระบาย)
    """
    g = 9.81
    q_total = float(params.get("flow_rate_q_m3s") or 0.0)
    h_static = float(params.get("static_head_m") or 0.0)
    num_pumps = int(params.get("num_parallel_pumps") or 1)
    
    segments = params.get("pipe_segments") or []
    
    # หาก AI ดึงมาแต่ค่าพื้นฐาน (ไม่มี array ของ segments) ให้สร้าง segment อัตโนมัติจากค่าหลัก
    if not segments and params.get("main_pipe_diameter_m") and params.get("main_pipe_length_m"):
        segments = [{
            "segment_name": "ท่อหลัก / อุโมงค์ส่งน้ำ (Main Conduit)",
            "flow_regime": params.get("system_type") or "Closed_Pressure",
            "carries_single_pump_flow": False,
            "diameter_m": float(params.get("main_pipe_diameter_m")),
            "area_m2": None,
            "length_m": float(params.get("main_pipe_length_m")),
            "manning_n": float(params.get("manning_n") or 0.014),
            "darcy_f": float(params.get("darcy_f") or 0.015),
            "minor_loss_k_sum": float(params.get("total_minor_k") or 0.0),
            "fixed_minor_loss_m": 0.0
        }]

    breakdown_rows = []
    total_major_loss = 0.0
    total_minor_loss = 0.0

    for idx, seg in enumerate(segments, start=1):
        name = seg.get("segment_name", f"ช่วงท่อที่ {idx}")
        is_unit_flow = bool(seg.get("carries_single_pump_flow", False))
        q_seg = (q_total / max(num_pumps, 1)) if is_unit_flow else q_total

        d_m = seg.get("diameter_m")
        area = seg.get("area_m2")
        w_perim = seg.get("wetted_perimeter_m")

        # คำนวณพื้นที่หน้าตัด A และรัศมีชลศาสตร์ R_h
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

        # เลือกสมการคำนวณ Major Loss (Manning vs Darcy-Weisbach)
        calc_method = formula_override or seg.get("preferred_formula") or "Manning"
        if "darcy" in calc_method.lower():
            h_major = f_val * (length / d_m) * ((velocity ** 2) / (2.0 * g)) if length > 0 else 0.0
            method_label = f"Darcy-Weisbach (f={f_val:.4f})"
        else:
            h_major = ((n_val ** 2) * (velocity ** 2) * length) / (r_h ** (4.0 / 3.0)) if length > 0 else 0.0
            method_label = f"Manning (n={n_val:.3f})"

        # คำนวณ Minor Loss จากค่าสัมประสิทธิ์ K (zeta) + ค่าสูญเสียคงที่ (เช่น ฝาย/ตะแกรงที่ทราบค่า)
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

    # สร้างสัมประสิทธิ์ความต้านทานของระบบ K จากสมการ H = K * Q^2 + Hst
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
# 3. โครงสร้างคำสั่ง AI (THE GATEKEEPER & MULTI-SEGMENT SCHEMA)
# ==========================================
extraction_tool = [{
    "type": "function",
    "function": {
        "name": "extract_hydraulic_parameters",
        "description": "สกัดพารามิเตอร์ระบบท่อ อุโมงค์ ท่อไซฟอน และสเปคปั๊ม สำหรับคำนวณ Head Loss และ System Curve",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "ชื่อโครงการ (ถ้ามี)"},
                "flow_rate_q_m3s": {"type": "number", "description": "อัตราการไหลรวมที่ต้องการของระบบในหน่วย m3/s (หากต้นฉบับเป็น m3/h ให้หาร 3600 เป็น m3/s)"},
                "flow_rate_original_text": {"type": "string", "description": "ค่าอัตราการไหลพร้อมหน่วยตามที่ปรากฏในเอกสาร เช่น '60 m3/s' หรือ '400 m3/h'"},
                "num_parallel_pumps": {"type": "integer", "description": "จำนวนเครื่องสูบน้ำที่เดินขนานกัน (ถ้าเป็นปั๊มเดี่ยวให้ใส่ 1)"},
                "static_head_m": {"type": "number", "description": "ระยะยกน้ำสถิต (Static Head / Hst) หรือผลต่างระดับน้ำต้นทาง-ปลายทาง (m)"},
                "suction_wl_m": {"type": "number", "description": "ระดับน้ำฝั่งดูด (m.MSL) ถ้ามี"},
                "discharge_wl_m": {"type": "number", "description": "ระดับน้ำฝั่งระบายออก (m.MSL) ถ้ามี"},
                "main_pipe_length_m": {"type": "number", "description": "ความยาวรวมของท่อหรืออุโมงค์หลัก (m)"},
                "main_pipe_diameter_m": {"type": "number", "description": "ขนาดเส้นผ่านศูนย์กลางท่อหรืออุโมงค์หลักในหน่วยเมตร (m)"},
                "pipe_material_or_roughness": {"type": "string", "description": "วัสดุท่อ และค่าความขรุขระ เช่น 'Concrete (Manning n = 0.014)'"},
                "manning_n": {"type": "number", "description": "ค่าสัมประสิทธิ์ความขรุขระ Manning's n (เช่น 0.014)"},
                "darcy_f": {"type": "number", "description": "ค่า Friction factor f ของ Darcy-Weisbach (ถ้าไม่ระบุให้ประเมินตามวัสดุ เช่น 0.015)"},
                "system_type": {"type": "string", "description": "ลักษณะระบบ เช่น 'ระบบปิด (Pressure)' หรือ 'ระบบเปิดผสมท่อปิด (Gravity + Pressure)'"},
                "proposed_pump_head_m": {"type": "number", "description": "ค่า Operating Head หรือ Total Head ของปั๊มที่ผู้รับจ้างเสนอ (m)"},
                "proposed_pump_max_head_m": {"type": "number", "description": "ค่า Specified Max Head ของปั๊ม (m) ถ้ามี"},
                "proposed_pump_eff_pct": {"type": "number", "description": "ประสิทธิภาพปั๊ม (%) ถ้ามี"},
                "proposed_motor_kw": {"type": "number", "description": "ขนาดกำลังมอเตอร์ที่เสนอ (kW) ถ้ามี"},
                "minor_loss_fittings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "รายการอุปกรณ์ที่ทำให้เกิด Minor Loss พร้อมค่า k หรือค่า loss (เป็นภาษาไทย)"
                },
                "pipe_segments": {
                    "type": "array",
                    "description": "แยกช่วงการคำนวณแนวท่อ/อุโมงค์/ท่อไซฟอน เพื่อคำนวณ Major และ Minor Loss อย่างละเอียด",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_name": {"type": "string", "description": "ชื่อช่วงท่อ เช่น '1. อุโมงค์ฝั่งดูด', '2. ท่อไซฟอนในสถานี', '3. อุโมงค์ฝั่งระบาย'"},
                            "carries_single_pump_flow": {"type": "boolean", "description": "เป็น True เฉพาะช่วงท่อที่แยกเข้าปั๊มแต่ละตัว เช่น ช่องรับน้ำหน้าปั๊มหรือท่อไซฟอน"},
                            "diameter_m": {"type": "number", "description": "เส้นผ่านศูนย์กลางท่อ (m)"},
                            "area_m2": {"type": "number", "description": "พื้นที่หน้าตัดการไหล (m2) กรณีเป็นท่อเหลี่ยมหรือไซฟอน"},
                            "length_m": {"type": "number", "description": "ความยาวช่วงท่อนี้ (m)"},
                            "manning_n": {"type": "number", "description": "ค่า Manning n ของช่วงนี้"},
                            "darcy_f": {"type": "number", "description": "ค่า Darcy f ของช่วงนี้"},
                            "minor_loss_k_sum": {"type": "number", "description": "ผลรวมค่าสัมประสิทธิ์การสูญเสียรอง (sum of k / zeta) ทั้งหมดในsegmentนี้ (คูณจำนวนจุดแล้ว)"},
                            "fixed_minor_loss_m": {"type": "number", "description": "ผลรวมค่า Head Loss ย่อยที่โจทย์ระบุมาเป็นหน่วยเมตรโดยตรง (m) เช่น ฝาย, ตะแกรง, จุดตัดปล่องที่ระบุเป็นเมตร"}
                        },
                        "required": ["segment_name"]
                    }
                },
                "missing_critical_data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "รายการข้อมูลสำคัญที่ขาดหายไปและจำเป็นต้องใช้คำนวณ System Curve (หากข้อมูลครบถ้วนให้ส่ง Array ว่าง [])"
                }
            },
            "required": ["missing_critical_data"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, ตารางคำนวณ, รูปภาพ หรือข้อความจากหน้างาน และสกัดพารามิเตอร์ทางชลศาสตร์ออกมาเพื่อส่งให้เครื่องคิดเลข Python คำนวณ System Curve

กฎสำคัญในการสกัดข้อมูล:
1. สื่อสารเป็น "ภาษาไทย" เสมอ (ทับศัพท์เทคนิคภาษาอังกฤษได้ เช่น Static Head, Minor Loss, System Curve)
2. หากข้อมูลใดไม่ปรากฏในเอกสาร ให้ใส่ค่าเป็น null (ห้ามเดาตัวเลขเองเด็ดขาด) และให้ระบุชื่อข้อมูลที่ขาดลงใน `missing_critical_data`
3. ข้อมูลขั้นต่ำที่จำเป็นต้องมีเพื่อคำนวณ System Curve คือ: อัตราการไหล (Q), Static Head (หรือระดับน้ำต้นทาง-ปลายทาง), ขนาดท่อ (Diameter/Area), และความยาวท่อ (Length) หากมีครบแล้ว ให้ `missing_critical_data` เป็น Array ว่าง `[]`
4. แยกข้อมูลแนวท่อลงใน `pipe_segments` ให้ละเอียด:
   - หากเป็นงานท่อทั่วไป ให้สร้าง 1 segment
   - หากเป็นงานระบบอุโมงค์ซับซ้อน (เช่น มีอุโมงค์ดูด, ท่อไซฟอนในสถานี, อุโมงค์ระบาย) ให้แยกเป็นหลาย segment ตามจริง โดยรวมค่า k ของข้องอ/ทางเข้า/ทางออก ไว้ใน `minor_loss_k_sum` หรือถ้าระบุเป็นหน่วยเมตรมาแล้วให้ใส่ใน `fixed_minor_loss_m`
"""

# ==========================================
# 4. ส่วนแสดงผล STEP 1: นำเข้าและสกัดข้อมูล
# ==========================================
st.subheader("1️⃣ ส่วนที่ 1: รับข้อมูลนำเข้า (Input Parameters & AI Extraction)")
st.write("อัปโหลดเอกสาร Submittal (PDF/Excel/CSV), รูปภาพตาราง/แบบแปลน (JPG/PNG), ถ่ายรูปจากกล้อง หรือพิมพ์ข้อความ")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "📂 อัปโหลดไฟล์เอกสารหรือรูปภาพ",
        accept_multiple_files=True,
        type=['pdf', 'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'txt']
    )
    camera_photo = st.camera_input("📸 หรือถ่ายรูปเอกสาร/ตารางสเปค")

with col2:
    manual_input = st.text_area(
        "✍️ หรือกรอกข้อมูลสเปคปั๊มและระบบท่อด้วยตนเอง:",
        height=180,
        placeholder="ตัวอย่าง: สูบน้ำรวม 60 m3/s (เดินขนาน 6 เครื่อง) ระดับน้ำดูด -3.00 m ระดับน้ำระบาย +2.50 m อุโมงค์ดูด Dia 5.0 m ยาว 8,435 m (n=0.014)..."
    )

if st.button("🚀 สกัดข้อมูลและวิเคราะห์พารามิเตอร์ (Extract & Parse Data)", type="primary", use_container_width=True):
    if uploaded_files or camera_photo or manual_input:
        with st.spinner("🤖 AI กำลังอ่านข้อมูลและจัดโครงสร้างพารามิเตอร์ชลศาสตร์..."):
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

            user_content = [{"type": "text", "text": f"กรุณาสกัดพารามิเตอร์ทางชลศาสตร์จากข้อมูลต่อไปนี้:\n\n{text_content}"}]
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
            st.session_state['extracted_params'] = json.loads(tool_call.function.arguments)
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์, ถ่ายรูป, หรือพิมพ์ข้อความก่อนกดปุ่มวิเคราะห์ครับ")

# ==========================================
# แสดงผลข้อมูลที่ดึงได้ (แก้ไขหน่วยไม่ออกเมื่อไม่พบข้อมูลแล้ว)
# ==========================================
if 'extracted_params' in st.session_state:
    data = st.session_state['extracted_params']
    st.divider()
    st.markdown("### 📋 สรุปพารามิเตอร์ที่ระบบอ่านได้จากเอกสารและรูปภาพ")

    # แถวที่ 1: Q, Static Head, Pump Head
    c1, c2, c3 = st.columns(3)
    q_display = data.get('flow_rate_original_text') or format_metric(data.get('flow_rate_q_m3s'), "m³/s")
    c1.metric("อัตราการไหล (Q)", q_display)
    c2.metric("ระยะยกน้ำ (Static Head)", format_metric(data.get('static_head_m'), "m"))
    c3.metric("Head ปั๊มที่เสนอ", format_metric(data.get('proposed_pump_head_m'), "m"))

    # แถวที่ 2: Pipe Length, Diameter, Roughness
    c4, c5, c6 = st.columns(3)
    c4.metric("ความยาวท่อหลัก (Pipe Length)", format_metric(data.get('main_pipe_length_m'), "m"))
    c5.metric("ขนาดท่อหลัก (Pipe Diameter)", format_metric(data.get('main_pipe_diameter_m'), "m"))
    c6.metric("วัสดุ/ความขรุขระ (Roughness)", format_metric(data.get('pipe_material_or_roughness'), ""))

    # แสดงรายการอุปกรณ์ Minor Loss
    with st.expander("🔧 ดูรายการอุปกรณ์ประกอบ (Minor Loss Fittings) และลักษณะระบบ", expanded=True):
        st.write(f"**ลักษณะระบบ:** {format_metric(data.get('system_type'), '')} | **จำนวนปั๊มเดินขนาน:** {format_metric(data.get('num_parallel_pumps'), 'เครื่อง', 0)}")
        fittings = data.get('minor_loss_fittings') or []
        if fittings:
            for item in fittings:
                st.write(f"- {item}")
        else:
            st.write("- ไม่พบข้อมูลอุปกรณ์ประกอบ")

    # ตรวจสอบ Gatekeeper
    missing = data.get('missing_critical_data') or []
    if missing:
        st.warning("⚠️ **ข้อมูลสำคัญขาดหาย (Gatekeeper Alert):** ระบบต้องการข้อมูลเหล่านี้เพิ่มเติมเพื่อคำนวณ System Curve ให้สมบูรณ์")
        for m in missing:
            st.error(f"❌ {m}")
        st.info("💡 *คุณสามารถป้อนค่าที่ขาดหายไปในส่วนที่ 2 ด้านล่างด้วยตนเอง เพื่อจำลองกราฟ System Curve ต่อได้ทันที*")
    else:
        st.success("✅ **ข้อมูลครบถ้วน:** พร้อมคำนวณ Head Loss และสร้างสมการ System Curve ในส่วนที่ 2!")

    # ==========================================
    # 5. ส่วนแสดงผล STEP 2: คำนวณ HEAD LOSS และสร้าง SYSTEM CURVE
    # ==========================================
    st.divider()
    st.subheader("2️⃣ ส่วนที่ 2: คำนวณ Head Loss และสร้างสมการ System Curve (H = KQ² + Hst)")

    # แถบตั้งค่าการคำนวณ (รองรับการปรับแก้ตัวเลขหน้างาน หรือเลือกสมการ Manning vs Darcy-Weisbach)
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        formula_choice = st.selectbox(
            "📐 เลือกสมการคำนวณ Major Loss:",
            ["Manning's Equation (ตามเอกสาร/งานอุโมงค์)", "Darcy-Weisbach (งานท่อแรงดันมาตรฐาน)"]
        )
    with cfg2:
        flow_unit_plot = st.selectbox(
            "📊 หน่วยอัตราการไหลบนแกน X ของกราฟ:",
            ["m³/s (ลูกบาศก์เมตรต่อวินาที)", "m³/h (ลูกบาศก์เมตรต่อชั่วโมง)"]
        )
    with cfg3:
        # เปิดโอกาสให้แก้ไขค่า Static Head หรือเติมค่าที่ขาดได้ทันที
        override_static = st.number_input(
            "ระดับยกน้ำสถิต Hst (m) [ปรับแก้ได้]:",
            value=float(data.get("static_head_m") or 5.50),
            step=0.5
        )
        data["static_head_m"] = override_static

    # ตรวจสอบว่ามีค่า Q และขนาดท่อเพียงพอสำหรับรันสมการหรือไม่
    can_calculate = (data.get("flow_rate_q_m3s") is not None) and (
        bool(data.get("pipe_segments")) or (data.get("main_pipe_diameter_m") and data.get("main_pipe_length_m"))
    )

    if can_calculate:
        selected_method = "Darcy" if "Darcy" in formula_choice else "Manning"
        calc_res = run_hydraulic_engine(data, formula_override=selected_method)

        # แสดงการ์ดสรุปผลการคำนวณ Head Loss
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ระยะยกน้ำสถิต (Hst)", f"{calc_res['h_static_m']:.2f} m")
        m2.metric("ความเสียดทานท่อตรง (Major Loss)", f"{calc_res['total_major_loss_m']:.3f} m")
        m3.metric("การสูญเสียตามอุปกรณ์ (Minor Loss)", f"{calc_res['total_minor_loss_m']:.3f} m")
        m4.metric("เฮดรวมของระบบ (Total System Head)", f"{calc_res['total_system_head_m']:.3f} m")

        # แสดงสมการ System Curve H = KQ^2 + Hst
        use_m3h = "m³/h" in flow_unit_plot
        if use_m3h:
            k_display = calc_res['k_system_m3s'] / (3600.0 ** 2)
            q_label = "Q (m³/h)"
            q_design = calc_res['q_total_m3h']
            eq_str = f"H = ({k_display:.4e}) · Q² + {calc_res['h_static_m']:.2f}"
        else:
            k_display = calc_res['k_system_m3s']
            q_label = "Q (m³/s)"
            q_design = calc_res['q_total_m3s']
            eq_str = f"H = {k_display:.6f} · Q² + {calc_res['h_static_m']:.2f}"

        st.info(f"🧮 **สมการ System Curve ของระบบ:**  `{eq_str}`   *(โดยที่ Dynamic Head Loss รวม = `{calc_res['total_dynamic_loss_m']:.3f} m` ที่ Q = `{q_design:,.1f}`)*")

        # ตารางแยกรายละเอียด Major & Minor Loss และความเร็ว V ในแต่ละช่วง
        st.markdown("#### 📑 ตารางแจกแจงการคำนวณความเร็ว (V) และ Head Loss รายช่วง")
        st.dataframe(calc_res["breakdown_df"], use_container_width=True, hide_index=True)

        # สร้างกราฟพาราโบลา System Curve (H = K*Q^2 + Hst)
        q_max_plot = q_design * 1.35 if q_design > 0 else 100.0
        q_vals = np.linspace(0, q_max_plot, 100)
        h_sys_vals = (k_display * (q_vals ** 2)) + calc_res["h_static_m"]
        h_loss_only_vals = k_display * (q_vals ** 2)

        fig = go.Figure()

        # เส้นที่ 1: System Curve รวม (H = KQ^2 + Hst)
        fig.add_trace(go.Scatter(
            x=q_vals, y=h_sys_vals,
            mode='lines',
            name='System Curve (H = KQ² + Hst)',
            line=dict(color='#00B4D8', width=4)
        ))

        # เส้นที่ 2: เส้น Dynamic Head Loss ล้วนๆ (KQ^2)
        fig.add_trace(go.Scatter(
            x=q_vals, y=h_loss_only_vals,
            mode='lines',
            name='Dynamic Head Loss (Major + Minor = KQ²)',
            line=dict(color='#F77F00', width=2, dash='dot')
        ))

        # เส้นที่ 3: เส้นระดับ Static Head (Hst)
        fig.add_hline(
            y=calc_res["h_static_m"],
            line_dash="dash",
            line_color="#90E0EF",
            annotation_text=f"Static Head (Hst = {calc_res['h_static_m']:.2f} m)",
            annotation_position="bottom right"
        )

        # จุดที่ 4: จุดความต้องการของระบบที่อัตราการไหลออกแบบ (Design System Duty Point)
        fig.add_trace(go.Scatter(
            x=[q_design],
            y=[calc_res["total_system_head_m"]],
            mode='markers+text',
            name='ความต้องการของระบบ (System Required Head)',
            marker=dict(color='#FFD166', size=13, symbol='circle'),
            text=[f"System Req: ({q_design:,.1f}, {calc_res['total_system_head_m']:.2f} m)"],
            textposition="top left"
        ))

        # จุดที่ 5: หากผู้รับจ้างเสนอค่า Head ของปั๊มมาแล้ว ให้พล็อตเทียบทันที
        pump_h = data.get("proposed_pump_head_m")
        if pump_h is not None:
            pump_h = float(pump_h)
            is_pass = pump_h >= calc_res["total_system_head_m"]
            marker_color = "#06D6A0" if is_pass else "#EF476F"
            status_tag = "ผ่านเกณฑ์ (PASS)" if is_pass else "เฮดไม่พอ (DEFICIT)"

            fig.add_trace(go.Scatter(
                x=[q_design],
                y=[pump_h],
                mode='markers+text',
                name=f'สเปคปั๊มที่เสนอ ({status_tag})',
                marker=dict(color=marker_color, size=15, symbol='star'),
                text=[f"Pump Spec: {pump_h:.2f} m ({status_tag})"],
                textposition="bottom right"
            ))

        fig.update_layout(
            title="📈 กราฟความสัมพันธ์ระหว่างอัตราการไหล (Q) และเฮดของระบบ (System Curve: H = KQ² + Hst)",
            xaxis_title=f"อัตราการไหล {q_label}",
            yaxis_title="Head (เมตร, m)",
            template="plotly_dark",
            hovermode="x unified",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # เก็บผลลัพธ์ไว้ใช้ต่อใน Phase 3 (เปรียบเทียบกับกราฟ Pump Curve เต็มรูปแบบ)
        st.session_state['hydraulic_results'] = calc_res
    else:
        st.warning("⏳ กรุณาอัปโหลดหรือป้อนข้อมูลที่มี **อัตราการไหล (Q)**, **ขนาดท่อ (Diameter)** และ **ความยาวท่อ (Length)** ให้ครบถ้วน เพื่อให้ระบบคำนวณและพล็อตกราฟ System Curve ครับ")
