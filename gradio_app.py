import gradio as gr
import requests
import numpy as np
import json

# Replace with your actual Cloud Run URL
CLOUD_RUN_API_URL = "https://fmri-api-883088739263.us-central1.run.app/predict"

def predict_age(random_seed):
    """
    Generates a mock PCA-transformed fMRI scan and sends it to the API.
    (Our model expects the 40 PCA components extracted from the 19,900 brain connections).
    """
    # Generate 40 mock features (PCA components) for demonstration
    np.random.seed(int(random_seed))
    mock_features = np.random.normal(loc=0, scale=1, size=40).tolist()
    
    try:
        response = requests.post(
            CLOUD_RUN_API_URL, 
            json={"features": mock_features},
            timeout=10
        )
        if response.status_code == 200:
            predicted_age = response.json().get("predicted_age", 0)
            return f"{predicted_age:.1f} years old", "API Request Successful!"
        else:
            return "Error", f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return "Error", str(e)

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Brain Age Predictor (fMRI)")
    gr.Markdown(
        "This application uses a machine learning model (XGBoost) trained on pediatric fMRI "
        "functional connectome data to predict a subject's chronological age based on their brain activity."
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Generate Brain Scan")
            gr.Markdown("*(Since an fMRI matrix has 19,900 features, use this slider to generate a mock patient profile represented by 40 PCA components)*")
            seed_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Patient ID (Random Seed)", value=42)
            predict_btn = gr.Button("Analyze Brain Scan", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 2. Prediction Results")
            age_output = gr.Textbox(label="Predicted Chronological Age", text_align="center")
            log_output = gr.Textbox(label="API Status Logs")
            
    predict_btn.click(
        fn=predict_age,
        inputs=[seed_slider],
        outputs=[age_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
