import streamlit as st
from openai import OpenAI
import json
import numpy as np
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="TEAM Pump QA/QC Assistant", layout="wide")
st.title("AI-Integrated Pump Spec Validator")

# Initialize OpenAI Client (Replace with your actual key or use st.secrets)
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()

# ==========================================
# 1. THE PYTHON TOOL (DETERMINISTIC MATH)
# ==========================================
def calculate_hydraulics(pipe_len, pipe_dia_mm, static_head, q_duty, h_duty):
    """Calculates head loss, generates curve data, and finds BEP deviation."""
    d_m = pipe_dia_mm / 1000.0
    c_val = 130.0 # Standard for HDPE/Steel
    
    # Generate arrays for plotting
    q_arr = np.linspace(50, max(q_duty * 2, 700), 50)
    q_m3s = q_arr / 3600.0
    
    # Hazen-Williams Head Loss
    hf = 10.67 * pipe_len * (q_m3s ** 1.852) / ((c_val ** 1.852) * (d_m ** 4.87))
    sys_head = static_head + hf
    
    # Calculate head loss specifically at duty point
    q_duty_m3s = q_duty / 3600.0
    hf_duty = 10.67 * pipe_len * (q_duty_m3s ** 1.852) / ((c_val ** 1.852) * (d_m ** 4.87))
    actual_system_head = static_head + hf_duty
    
    # Return calculated data and arrays for plotting
    return {
        "calculated_system_head_at_duty": round(actual_system_head, 2),
        "pump_head_deficit": round(actual_system_head - h_duty, 2),
        "q_arr": q_arr.tolist(),
        "sys_head": sys_head.tolist()
    }

# ==========================================
# 2. THE AI INSTRUCTION MANUAL (TOOL SCHEMA)
# ==========================================
# This tells the AI exactly what variables to extract from the user's text.
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_hydraulics",
            "description": "Calculates the hydraulic system curve and checks if the pump's duty head can overcome the system resistance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fluid_type": {"type": "string", "description": "Type of fluid (e.g., clean, canal, brackish)"},
                    "casing_mat": {"type": "string", "description": "Pump casing material"},
                    "impeller_mat": {"type": "string", "description": "Pump impeller material"},
                    "pipe_len": {"type": "number", "description": "Length of pipe in meters"},
                    "pipe_dia_mm": {"type": "number", "description": "Diameter of pipe in millimeters"},
                    "static_head": {"type": "number", "description": "Static head in meters"},
                    "q_duty": {"type": "number", "description": "Target flow rate (m3/h)"},
                    "h_duty": {"type": "number", "description": "Target pump head (m)"}
                },
                "required": ["fluid_type", "casing_mat", "impeller_mat", "pipe_len", "pipe_dia_mm", "static_head", "q_duty", "h_duty"]
            }
        }
    }
]

system_prompt = """
You are a Lead QA/QC Site Engineer at TEAM Consulting. 
1. If the user input is missing ANY required parameters (fluid type, pump materials, duty Q/H, pipe length, pipe dia, static head), DO NOT call the function. Instead, reply in Thai, list what you have, and ask for what is missing.
2. If all data is present, call the `calculate_hydraulics` tool.
3. Once the tool returns the math, write a professional Site Memo in Thai/English evaluating Material Suitability and Hydraulic Capability based on the tool's strict mathematical result. Do not recalculate the math yourself.
"""

# ==========================================
# 3. STREAMLIT UI & EXECUTION LOOP
# ==========================================
st.write("### 📝 Field Engineer Input")
user_input = st.text_area("Paste site notes or pump specs here:", height=120)

if st.button("Run AI Analysis"):
    if user_input:
        with st.spinner("AI is evaluating input and orchestrating tools..."):
            
            # Step 1: Send user text and tools to AI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Step 2: Check if AI decided to call the tool (meaning data is complete)
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                
                st.success("✅ ข้อมูลครบถ้วน: AI successfully extracted parameters and triggered the Python math engine.")
                
                # Execute the deterministic Python function
                math_result = calculate_hydraulics(
                    pipe_len=args['pipe_len'],
                    pipe_dia_mm=args['pipe_dia_mm'],
                    static_head=args['static_head'],
                    q_duty=args['q_duty'],
                    h_duty=args['h_duty']
                )
                
                # Plot the System Curve using Streamlit/Plotly
                st.write("### 📈 System Curve Analysis")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=math_result["q_arr"], y=math_result["sys_head"], mode='lines', name='System Curve', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=[args['q_duty']], y=[args['h_duty']], mode='markers+text', name='Proposed Pump Duty', 
                                         marker=dict(size=12, color='blue'), text=["Duty Point"], textposition="top left"))
                fig.update_layout(xaxis_title="Flow Rate (m³/h)", yaxis_title="Head (m)", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Step 3: Send math back to AI for final memo synthesis
                st.write("### 📄 AI Executive Site Memo")
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps({"actual_system_head": math_result["calculated_system_head_at_duty"], "head_deficit": math_result["pump_head_deficit"]})
                        }
                    ]
                )
                st.markdown(final_response.choices[0].message.content)
                
            else:
                # If no tool was called, the AI acts as the Gatekeeper
                st.warning("⚠️ ข้อมูลไม่ครบถ้วน (Incomplete Data)")
                st.info(message.content)