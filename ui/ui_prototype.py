import streamlit as st
import io # Needed for handling the uploaded file object

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="VLM Industrial Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header and Project Info ---
st.title("🏭 VLM-Powered Industrial Anomaly Detection")
st.markdown("Use natural language to define normal and anomalous conditions in a video stream.")
st.markdown("---")


# --- 1. File Upload and Input Handling ---
with st.sidebar:
    st.header("1. Upload Video")
    video_file = st.file_uploader(
        "Upload Video for Analysis", 
        type=['mp4', 'mov', 'avi'], 
        help="Upload an industrial video up to 200MB."
    )
    
    st.header("2. Define Conditions")
    # Text input for the 'Normal' condition
    normal_prompt = st.text_input(
        "Normal Condition Prompt",
        value="The machine is running smoothly and quietly.",
        help="Describe what a 'normal' scene looks like."
    )
    # Text input for the 'Anomaly' condition
    anomaly_prompt = st.text_input(
        "Anomalous Condition Prompt",
        value="A mechanical part is sparking or an oil leak is visible.",
        help="Describe the specific anomaly to detect."
    )

    # 3. Analysis Button (Week 1 Mock-up)
    st.markdown("---")
    start_analysis_button = st.button("▶️ Start Anomaly Analysis", type="primary", use_container_width=True)

# --- Main Display Area ---
st.header("Analysis Dashboard")

if video_file is not None:
    # Display the uploaded video (This is just a mock display)
    st.subheader("Uploaded Video Preview")
    st.video(video_file)
else:
    st.info("Please upload a video and define the conditions in the sidebar to begin analysis.")


# --- Results Placeholder (Will be completed in Phase 3) ---
st.markdown("---")

# This is the mock logic for the Week 1 deliverable
if start_analysis_button:
    if video_file is None or not normal_prompt or not anomaly_prompt:
        st.error("❌ Please ensure a video is uploaded and both text prompts are defined.")
    else:
        # Mock success and state capture for Week 1
        st.success("✅ Inputs Captured Successfully!")
        st.markdown(f"**Video Name:** `{video_file.name}`")
        st.markdown(f"**Normal Prompt:** `{normal_prompt}`")
        st.markdown(f"**Anomaly Prompt:** `{anomaly_prompt}`")
        
        st.info("In Phase 3, the ML Backend will be called here to process the video and generate scores.")

# Placeholder for results visualization
st.subheader("Anomaly Score Over Time (Placeholder)")
st.warning("Visualization coming in Phase 3: System Integration.")
