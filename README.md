# Volunteer Sentiment Ensemble System

An AI-powered system that analyzes volunteer voice recordings to automatically determine sentiment and recommendation outcomes such as **Selected**, **Need to Verify**, or **Rejected**.
It uses speech transcription with Whisper and an ensemble of three state-of-the-art NLP models for reliable classification.

---

## Overview

The **Volunteer Sentiment Ensemble System** combines speech-to-text and natural language understanding to evaluate spoken responses from volunteers.
By leveraging an ensemble of transformer-based models, it ensures a balanced and consistent decision-making process.

---

## Features

* **Automatic speech transcription** using OpenAI Whisper
* **Ensemble prediction** across three models (BART, DistilBART, and DeBERTa)
* **Confidence score averaging** for final label prediction
* **Clean, interactive Gradio interface** for uploading or recording audio
* **CSV logging** of all transcribed and analyzed results
* **Custom white-blue themed dashboard** designed for clarity and usability

---

## Tech Stack

| Component           | Model / Library                                                                                  | Purpose                              |
| ------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------ |
| Speech Recognition  | `openai/whisper-base`                                                                            | Converts voice to text               |
| Text Classification | `facebook/bart-large-mnli`, `valhalla/distilbart-mnli-12-3`, `MoritzLaurer/DeBERTa-v3-base-mnli` | Sentiment ensemble                   |
| Deep Learning       | `transformers`, `torch`                                                                          | Model execution and inference        |
| Interface           | `gradio`                                                                                         | Web UI for audio upload and analysis |
| Logging             | `pandas`                                                                                         | Stores results with timestamps       |
| Data Handling       | `numpy`, `re`, `datetime`                                                                        | Text processing and statistics       |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/volunteer-sentiment-ensemble.git
cd volunteer-sentiment-ensemble
```

### 2. Install Dependencies

```bash
pip install gradio torch transformers pandas numpy openai-whisper
```

For faster performance on GPU:

```bash
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## How It Works

1. **Upload or record** a volunteer’s audio response.
2. The **Whisper model** transcribes speech into text.
3. Three transformer models analyze the text and classify it as:

   * Selected
   * Need to Verify
   * Rejected
4. The **ensemble engine** averages probabilities from all three models to produce a final, balanced recommendation.
5. All results, including transcription, prediction, and confidence, are automatically saved to `volunteer_sentiment_log.csv`.


## Output Example

**Transcribed Text**

> The student appears to be genuine and dedicated.

**Model Predictions**

* Model 1 (BART): Selected (0.94)
* Model 2 (DistilBART): Selected (0.91)
* Model 3 (DeBERTa): Selected (0.88)

**Final Result**
**Selected** with confidence **0.91**

Data is logged automatically in `volunteer_sentiment_log.csv`.


## File Structure

📁 volunteer-sentiment-ensemble/
│
├── app.py                        # Main application file
├── volunteer_sentiment_log.csv    # Output CSV (auto-generated)
└── README.md                     # Documentation



## Notes

* Ensure your audio files are clear (16 kHz WAV recommended).
* Ensemble averaging helps minimize bias from individual models.
* The system can easily be extended for multilingual speech or domain-specific sentiment detection.


## Future Enhancements

* Tamil to English translation support
* Voice-based feedback generation (Text-to-Speech)
* Custom fine-tuning for domain-specific sentiment tone
* Deployment on Hugging Face Spaces or Streamlit


## Author

**Avinash R**
AI enthusiast | SIST'27
Focused on AI applications, automation, and data-driven innovovation
