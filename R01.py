import streamlit as st
import pandas as pd
import PyPDF2
from openai import OpenAI
import json
import base64

# --- ตั้งค่าหน้าจอ (PAGE CONFIG) ---
st.set_page_config(page_title="AI Pump Validator", layout="wide")
st.title("🤖 ระบบ AI ตรวจสอบความเหมาะสมเครื่องสูบน้ำ (AI Pump Spec Validator)")
st.subheader("ส่วนที่ 1: รับข้อมูลนำเข้า (Data Extraction & Parsing)")

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================================
# ฟังก์ชันอ่านไฟล์และแปลงรูปภาพ (UTILITIES)
# ==========================================
def extract_text_from_file(uploaded_file):
    """ฟังก์ชันดึงข้อความจากไฟล์ PDF, CSV, Excel"""
    text = ""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        text = df.to_string()
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
        text = df.to_string()
    else:
        text = uploaded_file.getvalue().decode("utf-8")
    return text

def encode_image(uploaded_file):
    """ฟังก์ชันแปลงไฟล์ภาพให้เป็น Base64 เพื่อส่งให้ OpenAI Vision"""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# ==========================================
# โครงสร้างคำสั่ง AI (THE GATEKEEPER & SCHEMA)
# ==========================================
extraction_tool = [{
    "type": "function",
    "function": {
        "name": "extract_hydraulic_parameters",
        "description": "ดึงข้อมูลพารามิเตอร์ทางชลศาสตร์และสเปคปั๊มจากข้อความหรือรูปภาพตาราง",
        "parameters": {
            "type": "object",
            "properties": {
                "flow_rate_q": {"type": "number", "description": "อัตราการไหลหรือความจุของปั๊ม (m3/s หรือ m3/h)"},
                "static_head": {"type": "number", "description": "ระยะยกน้ำสถิต หรือส่วนต่างระดับน้ำต้นทาง-ปลายทาง (m)"},
                "pipe_length": {"type": "number", "description": "ความยาวรวมของท่อหรืออุโมงค์ (m)"},
                "pipe_diameter": {"type": "number", "description": "ขนาดเส้นผ่านศูนย์กลางท่อ (m หรือ mm)"},
                "pipe_material_or_roughness": {"type": "string", "description": "วัสดุท่อ, ค่า Manning's n, หรือ Friction factor"},
                "minor_loss_fittings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "รายการอุปกรณ์ที่ทำให้เกิด Minor Loss (เช่น ข้องอ, วาล์ว, ปล่องรับน้ำ)"
                },
                "proposed_pump_head": {"type": "number", "description": "ค่า Head ของปั๊มที่ผู้รับจ้างเสนอขออนุมัติ (m)"},
                "missing_critical_data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "รายการข้อมูลสำคัญที่ขาดหายไปและจำเป็นต้องใช้คำนวณ System Curve (ให้ตอบเป็นภาษาไทย)"
                }
            },
            "required": ["missing_critical_data"]
        }
    }
}]

system_prompt = """
คุณคือวิศวกรที่ปรึกษาอาวุโส (Lead QA/QC Site Engineer) ของบริษัท TEAM Consulting 
หน้าที่ของคุณคืออ่านเอกสาร Submittal, BOQ, รายการคำนวณ, หรือรูปภาพถ่ายหน้างาน (ภาพตาราง/แบบแปลน)
และดึงข้อมูลพารามิเตอร์ทางชลศาสตร์ที่จำเป็นออกมาให้ครบถ้วนที่สุด 
- คุณต้องสื่อสารและรายงานผลเป็น "ภาษาไทย" ทั้งหมด (ใช้ศัพท์วิศวกรรมทับศัพท์ได้ เช่น Head, Flow Rate)
- หากข้อมูลใดสำคัญแต่ผู้ใช้งานไม่ได้ให้มา ให้ระบุลงใน 'missing_critical_data' เป็นภาษาไทยอย่างชัดเจนเพื่อแจ้งเตือนผู้ใช้งาน
"""

# ==========================================
# ส่วนแสดงผลบนเว็บ (STREAMLIT UI)
# ==========================================
st.write("📥 **นำเข้าข้อมูล:** อัปโหลดเอกสาร Submittal (PDF/Excel), รูปภาพตาราง (JPG/PNG), ถ่ายรูปจากกล้อง หรือพิมพ์ข้อความ")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("📂 อัปโหลดไฟล์เอกสารและรูปภาพ", accept_multiple_files=True, type=['pdf', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'])
    camera_photo = st.camera_input("📸 ถ่ายรูปภาพเอกสาร / หน้าจอ / ตารางสเปค")

with col2:
    manual_input = st.text_area("✍️ หรือกรอกข้อมูลด้วยตนเอง (Manual Input):", height=150, placeholder="พิมพ์รายละเอียดสเปคปั๊มและแนวท่อที่นี่...")

if st.button("🚀 สกัดข้อมูลและวิเคราะห์พารามิเตอร์ (Extract Data)", type="primary", use_container_width=True):
    
    if uploaded_files or camera_photo or manual_input:
        with st.spinner("🤖 AI กำลังอ่านเอกสาร วิเคราะห์รูปภาพ และดึงพารามิเตอร์..."):
            
            text_content = manual_input
            image_base64_list = []
            
            # 1. จัดการไฟล์ที่อัปโหลดและรูปถ่าย
            for file in (uploaded_files or []):
                if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_base64_list.append(encode_image(file))
                else:
                    text_content += f"\n--- ข้อมูลจากไฟล์ {file.name} ---\n"
                    text_content += extract_text_from_file(file)
            
            if camera_photo:
                image_base64_list.append(encode_image(camera_photo))
                
            # 2. สร้างโครงสร้างข้อความเพื่อส่งให้ OpenAI (รวมข้อความและภาพ)
            user_content = [{"type": "text", "text": f"กรุณาดึงข้อมูลพารามิเตอร์ทางชลศาสตร์จากข้อมูลและรูปภาพเหล่านี้:\n\n{text_content}"}]
            
            for img_b64 in image_base64_list:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
                
            # 3. เรียกใช้ OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o", # ต้องใช้รุ่น 4o เพื่อให้รองรับ Vision
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                tools=extraction_tool,
                tool_choice={"type": "function", "function": {"name": "extract_hydraulic_parameters"}}
            )
            
            # 4. แปลงข้อมูล JSON ที่ AI ส่งกลับมา
            tool_call = response.choices[0].message.tool_calls[0]
            extracted_data = json.loads(tool_call.function.arguments)
            
            st.success("✅ ดึงข้อมูลสำเร็จเรียบร้อย!")
            
            # ==========================================
            # แสดงผลข้อมูลบนหน้าจอ (THAI UI)
            # ==========================================
            st.markdown("### 📋 สรุปพารามิเตอร์ที่ระบบอ่านได้จากเอกสารและรูปภาพ")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("อัตราการไหล (Q)", f"{extracted_data.get('flow_rate_q', 'ไม่พบข้อมูล')}")
            c2.metric("ระยะยกน้ำ (Static Head)", f"{extracted_data.get('static_head', 'ไม่พบข้อมูล')} m")
            c3.metric("Head ปั๊มที่เสนอ", f"{extracted_data.get('proposed_pump_head', 'ไม่พบข้อมูล')} m")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("ความยาวท่อ (Pipe Length)", f"{extracted_data.get('pipe_length', 'ไม่พบข้อมูล')} m")
            c5.metric("ขนาดท่อ (Pipe Diameter)", f"{extracted_data.get('pipe_diameter', 'ไม่พบข้อมูล')}")
            c6.metric("วัสดุ/ความขรุขระ (Roughness)", f"{extracted_data.get('pipe_material_or_roughness', 'ไม่พบข้อมูล')}")
            
            st.markdown("#### 🔧 อุปกรณ์ประกอบ (Minor Loss Fittings):")
            fittings = extracted_data.get('minor_loss_fittings', [])
            if fittings:
                for item in fittings:
                    st.write(f"✔️ {item}")
            else:
                st.write("- ไม่พบข้อมูลอุปกรณ์ประกอบในเอกสาร")
            
            # ระบบ Gatekeeper แจ้งเตือนข้อมูลที่หายไป
            missing = extracted_data.get('missing_critical_data', [])
            if missing:
                st.warning("⚠️ **ข้อมูลสำคัญขาดหาย (Gatekeeper Alert):** ระบบต้องการข้อมูลเหล่านี้เพิ่มเติมเพื่อคำนวณ System Curve ให้สมบูรณ์")
                for m in missing:
                    st.error(f"❌ {m}")
            else:
                st.info("✅ ข้อมูลครบถ้วน พร้อมส่งต่อเข้าสู่กระบวนการคำนวณ System Curve ในขั้นตอนต่อไป")
                    
            # บันทึกค่าไว้ใน Session State เพื่อใช้คำนวณต่อใน Flow ถัดไป
            st.session_state['extracted_params'] = extracted_data
            
    else:
        st.error("🚨 กรุณาอัปโหลดไฟล์เอกสาร, ถ่ายรูปภาพ, หรือกรอกข้อมูลก่อนกดปุ่มวิเคราะห์ครับ")