

import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from pyngrok import ngrok

# =========================
# CONFIG
# =========================
MODEL_PATH = "/content/drive/MyDrive/brain_tumor_resnet50.h5"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/uploads"
PORT = 5000

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Brain Tumor Detection</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
*{
    box-sizing:border-box;
    font-family:'Poppins',sans-serif;
}

body{
    margin:0;
    height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    background-size:300% 300%;
    animation:bgMove 10s ease infinite;
}

@keyframes bgMove{
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

.glass{
    width:460px;
    padding:35px;
    border-radius:25px;
    background:rgba(255,255,255,0.15);
    backdrop-filter:blur(15px);
    -webkit-backdrop-filter:blur(15px);
    box-shadow:0 25px 50px rgba(0,0,0,0.4);
    color:white;
    text-align:center;
}

.logo{
    font-size:48px;
}

h1{
    font-size:26px;
    margin:10px 0 5px;
    font-weight:600;
}

.subtitle{
    font-size:14px;
    opacity:0.85;
    margin-bottom:25px;
}

.upload-box{
    border:2px dashed rgba(255,255,255,0.4);
    border-radius:18px;
    padding:25px;
    transition:0.4s;
}

.upload-box:hover{
    background:rgba(255,255,255,0.08);
    transform:scale(1.02);
}

input[type=file]{
    color:white;
    margin-top:10px;
}

button{
    margin-top:20px;
    background:linear-gradient(135deg,#ff512f,#dd2476);
    border:none;
    color:white;
    padding:14px 30px;
    font-size:16px;
    border-radius:50px;
    cursor:pointer;
    box-shadow:0 10px 25px rgba(0,0,0,0.3);
    transition:0.4s;
}

button:hover{
    transform:translateY(-3px);
    box-shadow:0 15px 35px rgba(0,0,0,0.5);
}

.result-card{
    margin-top:25px;
    background:rgba(255,255,255,0.12);
    padding:20px;
    border-radius:18px;
    animation:fadeIn 0.8s ease;
}

@keyframes fadeIn{
    from{opacity:0;transform:translateY(15px)}
    to{opacity:1;transform:translateY(0)}
}

.result{
    font-size:26px;
    font-weight:700;
}

.tumor{color:#ff4b4b;}
.normal{color:#2ecc71;}

.confidence{
    margin-top:10px;
    font-size:16px;
}

.progress{
    height:10px;
    width:100%;
    background:rgba(255,255,255,0.3);
    border-radius:10px;
    margin-top:8px;
    overflow:hidden;
}

.progress-bar{
    height:100%;
    background:linear-gradient(to right,#00f260,#0575e6);
    width:{{ confidence }}%;
    transition:1s;
}

img{
    margin-top:15px;
    max-width:100%;
    border-radius:15px;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

.footer{
    margin-top:20px;
    font-size:12px;
    opacity:0.75;
}
</style>
</head>

<body>

<div class="glass">
    <div class="logo">🧠</div>
    <h1>AI Brain Tumor Detection</h1>
    <div class="subtitle">Deep Learning Powered MRI Analysis</div>

    <form method="POST" enctype="multipart/form-data">
        <div class="upload-box">
            <strong>Upload Brain MRI Image</strong><br>
            <input type="file" name="file" required>
        </div>
        <button type="submit">🔍 Analyze MRI</button>
    </form>

    {% if label %}
    <div class="result-card">
        <div class="result {% if label=='Tumor' %}tumor{% else %}normal{% endif %}">
            {{ label }}
        </div>

        <div class="confidence">
            Confidence: {{ confidence }}%
            <div class="progress">
                <div class="progress-bar"></div>
            </div>
        </div>

        <img src="{{ image_path }}">
    </div>
    {% endif %}

    <div class="footer">
        © 2026 | Developed by <b>Hasibullah Habibi</b><br>
        AI-Powered Medical Diagnosis System
    </div>
</div>

</body>
</html>
"""


# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def predict():
    label = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        image_path = "/" + path

        # Image preprocessing
        img = image.load_img(path, target_size=IMG_SIZE)
        x = image.img_to_array(img)

        # Ensure RGB
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Prediction
        prob = float(model.predict(x)[0][0])

        if prob > 0.5:
            label = "Tumor"
            confidence = round(prob * 100, 2)
        else:
            label = "Normal"
            confidence = round((1 - prob) * 100, 2)

    return render_template_string(
        HTML,
        label=label,
        confidence=confidence,
        image_path=image_path
    )

# =========================
# RUN WITH NGROK
# =========================

# 🔑 PUT YOUR NGROK AUTHTOKEN HERE
ngrok.set_auth_token("37hjaAjmNfrjNh3RCCozVZn2jYq_6gipATtnF8PFPRyAnPVfF")

public_url = ngrok.connect(PORT)
print("🌍 Public URL:", public_url)

app.run(port=PORT)
