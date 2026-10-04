# 🌿 Crop Disease Detection Web App

A Streamlit-based web app to detect plant diseases from leaf images using a trained MobileNetV2 model.

## 🧠 Model Info
- **Model:** MobileNetV2 (fine-tuned)
- **Input Size:** 224x224
- **Classes:** 38 plant disease types
- **Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🛠 Tech Stack
- Python, TensorFlow, Keras
- Streamlit, NumPy, Pillow

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/Plant_Disease.git
cd Plant_Disease
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
