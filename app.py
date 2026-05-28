import os
import re
import sqlite3
import smtplib
import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
import moviepy.editor as mp

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, render_template, request, redirect, url_for, session
import torchvision.models as models
from config import Config

# ================= FLASK =================
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
UPLOAD_FOLDER = app.config["UPLOAD_FOLDER"]
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= CLASSES =================
classes = [
    'FakeVideo-FakeAudio',
    'FakeVideo-RealAudio',
    'RealVideo-FakeAudio',
    'RealVideo-RealAudio'
]

# ================= MODEL =================
class MultiModalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(512, 256, batch_first=True)

        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.audio_fc = nn.Linear(64 * 32 * 32, 256)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, video, audio):
        B, T, C, H, W = video.shape

        video = video.view(B*T, C, H, W)
        feats = self.cnn(video)
        feats = feats.view(B, T, -1)

        _, (h, _) = self.lstm(feats)
        video_feat = h[-1]

        audio_feat = self.audio_cnn(audio)
        audio_feat = self.audio_fc(audio_feat)

        combined = torch.cat([video_feat, audio_feat], dim=1)
        return self.fc(combined)

# Load model
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("Model/best_model.pth", map_location=device))
model.eval()

print("Model Loaded")

# ================= FEATURE EXTRACTION =================
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        count += 1

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((224,224,3)))

    frames = np.array(frames[:max_frames])
    return torch.tensor(frames).permute(0,3,1,2).float().unsqueeze(0)


def extract_audio(video_path):
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio.to_soundarray(fps=16000)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    mel_db = np.resize(mel_db, (128, 128))
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()


def get_user_email(username):
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("SELECT email FROM info WHERE user = ?", (username,))
    data = cur.fetchone()
    con.close()
    return data[0] if data else None


def send_fake_alert_email(recipient_email, username, prediction, video_filename):
    smtp_host = app.config.get("SMTP_HOST")
    smtp_port = app.config.get("SMTP_PORT", 587)
    smtp_user = app.config.get("SMTP_USER")
    smtp_password = app.config.get("SMTP_PASSWORD")
    sender_email = app.config.get("SMTP_SENDER", smtp_user)

    if not all([smtp_host, smtp_user, smtp_password, sender_email, recipient_email]):
        return False, "SMTP settings are incomplete."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "DeepVision Alert: Fake media detected"

    body = f"""
Hello {username},

DeepVision detected potentially manipulated media in your recent upload.

Detected class: {prediction}
Final output: Spoof
Uploaded file: {video_filename}

Please review the result in the DeepVision dashboard immediately.

Regards,
DeepVision Alert System
""".strip()

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True, "Alert email sent successfully."
    except Exception as exc:
        return False, str(exc)

# ================= ROUTES =================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file:
        return redirect(url_for('home'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    video = extract_frames(filepath).to(device)
    audio = extract_audio(filepath).to(device)

    with torch.no_grad():
        outputs = model(video, audio)
        pred = torch.argmax(outputs, dim=1).item()

    prediction = classes[pred]
    output_format = "Spoof" if "Fake" in prediction else "Real"
    video_path = os.path.basename(filepath)
    email_status = None

    if output_format == "Spoof":
        username = session.get("username")
        if username:
            recipient_email = session.get("email") or get_user_email(username)
            if recipient_email:
                email_sent, email_message = send_fake_alert_email(
                    recipient_email=recipient_email,
                    username=username,
                    prediction=prediction,
                    video_filename=video_path
                )
                email_status = {
                    "sent": email_sent,
                    "message": email_message,
                    "recipient": recipient_email
                }
            else:
                email_status = {
                    "sent": False,
                    "message": "No registered email found for the logged-in user.",
                    "recipient": None
                }
        else:
            email_status = {
                "sent": False,
                "message": "No logged-in user found, so alert email was skipped.",
                "recipient": None
            }

    return render_template(
        "result.html",
        prediction=prediction,
        output_format=output_format,
        email_status=email_status,
        video_path=video_path
    )




@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            session["username"] = mail1
            session["email"] = get_user_email(mail1)
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            session["username"] = mail1
            session["email"] = get_user_email(mail1)
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/home1')
def home1():
	return render_template('home1.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


if __name__ == "__main__":
    app.run(debug=False)
