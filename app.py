import gradio as gr
import whisper
from transformers import pipeline
import pandas as pd
import datetime
import re

# ---------------- Load Models ----------------
whisper_model = whisper.load_model("base")

# Use BART large MNLI for contextual sentiment classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ---------------- Core Function ----------------
def analyze_audio(audio_file):
    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_file, language="en")
    text = result["text"]

    # Clean text
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text).strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # Custom labels for classification
    labels = ["Selected", "Need to Verify", "Rejected"]

    # Analyze with BART
    result = classifier(clean_text, labels)
    prediction = result["labels"][0]
    confidence = round(result["scores"][0], 3)

    # Recommendation logic (same as prediction)
    if prediction == "Selected":
        sentiment_label = "POSITIVE"
        emoji = "🟢"
    elif prediction == "Need to Verify":
        sentiment_label = "NEUTRAL"
        emoji = "🟡"
    else:
        sentiment_label = "NEGATIVE"
        emoji = "🔴"

    recommendation = f"{emoji} Volunteer Recommendation: **{prediction}**"

    # Save log to CSV
    data = {
        "Timestamp": [datetime.datetime.now()],
        "Audio_File": [audio_file],
        "Transcribed_Text": [clean_text],
        "Predicted_Label": [prediction],
        "Confidence_Score": [confidence],
    }

    csv_path = "volunteer_sentiment_log.csv"
    try:
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, pd.DataFrame(data)], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Final formatted output
    return (
        f"### 🎧 Transcribed Text:\n{clean_text}\n\n"
        f"### 📊 Analysis Result:\n{emoji} **{sentiment_label}**\n\n"
        f"**Confidence:** {confidence}\n\n"
        f"### {recommendation}\n\n"
        f"✅ Result saved to `volunteer_sentiment_log.csv`"
    )

# ---------------- Custom White-Blue UI ----------------
with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    body {
        background-color: #f6f9fc;
        font-family: 'Inter', sans-serif;
    }
    .header {
        background-color: #007bff;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .gr-button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600;
    }
    .gr-button:hover {
        background-color: #0056b3 !important;
    }
    footer {display: none !important;}
    #root > div.absolute.bottom-0.left-0.right-0 {display: none !important;}
    """
) as demo:

    # Header
    gr.HTML("<div class='header'>🎙️ Volunteer Sentiment Verification System</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='card'><h3>🎧 Upload or Record Voice</h3></div>")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="")
            analyze_button = gr.Button("🔍 Analyze")

        with gr.Column(scale=1):
            gr.HTML("<div class='card'><h3>📊 Result</h3></div>")
            result_output = gr.Markdown(label="Analysis Result")

    analyze_button.click(analyze_audio, inputs=audio_input, outputs=result_output)

# Launch app
demo.launch(show_api=False, share=False, server_name="127.0.0.1")