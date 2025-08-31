from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
import cv2
import pickle
import re
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


app = Flask(__name__)
app.secret_key = "secret_key"

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = 'static/files'
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb.cursors
from dotenv import load_dotenv
from flask_mysqldb import MySQL
load_dotenv()
from groq import Groq

app.config['MYSQL_HOST'] = '127.0.0.1'     # localhost ki jagah 127.0.0.1 use karo
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Drishtib#123'  # <-- apna root password
app.config['MYSQL_DB'] = 'health'
app.config['MYSQL_PORT'] = 3306



# Intialize MySQL
mysql = MySQL(app)


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
# Load the saved models
with open('chatbot_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load intents data
import json
with open('mental_health_intent_dataset.json') as file:
    data = json.load(file)


# Manual stopwords (used during preprocessing)
manual_stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ourselves', 'you', 'your', 'yours', 'he', 'him',
    'she', 'her', 'it', 'its', 'they', 'them', 'what', 'which', 'who', 'this', 'that', 'am',
    'is', 'are', 'was', 'be', 'been', 'have', 'has', 'do', 'does', 'did', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once', 'here'
])

def preprocess_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = " ".join(word.strip(".,!?()[]{}") for word in text.split())
    return text


@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()

        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return render_template('index.html')
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)




@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO accounts (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'newpassword' in request.form and 'confirmpassword' in request.form:
        username = request.form['username']
        new_password = request.form['newpassword']
        confirm_password = request.form['confirmpassword']

        # Check if the user exists in the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()

        if not account:
            msg = 'Account does not exist!'
        elif new_password != confirm_password:
            msg = 'Passwords do not match!'
        else:
            # Update the password
            cursor.execute('UPDATE accounts SET password = %s WHERE username = %s', (new_password, username))
            mysql.connection.commit()
            msg = 'Password successfully updated!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'

    return render_template('reset_password.html', msg=msg)


@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input]).toarray()
    predicted_tag_index = model.predict(input_vector)[0]
    predicted_tag = label_encoder.inverse_transform([predicted_tag_index])[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return np.random.choice(intent['responses'])

def generate_llm_response(user_input):
    """Generate a human-like response using Groq Llama 3 if configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        client = Groq(api_key=api_key)
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

        # Retrieve limited chat history from session for better context
        history = session.get('chat_history', [])
        trimmed_history = history[-8:] if len(history) > 8 else history

        system_prompt = (
            "You are MindSoothe, a warm, empathetic mental health support assistant. "
            "Be conversational, supportive, and human-like. Offer brief, practical next steps. "
            "Avoid medical diagnosis; encourage professional help when appropriate. Keep replies within 3-6 sentences."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        reply = completion.choices[0].message.content.strip()

        # Update session history
        updated_history = trimmed_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply}
        ]
        session['chat_history'] = updated_history[-12:]

        return reply
    except Exception:
        return None

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
   
    return render_template('service.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form.get("user_input")
    if user_input:
        # Try LLM first for human-like replies; fallback to intent model
        llm_reply = generate_llm_response(user_input)
        if llm_reply:
            return jsonify({"response": llm_reply})

        response = chatbot_response(user_input)
        return jsonify({"response": response})
    else:
        return jsonify({"response": "I didn't understand that. Can you please clarify?"})



PHQ2_QUESTIONS = [
    "Have you had little interest or pleasure in doing things?",
    "Have you been feeling down, depressed or hopeless?"
]

PHQ9_QUESTIONS = PHQ2_QUESTIONS + [
    "Have you had trouble falling or staying asleep or sleeping too much?",
    "Have you been feeling tired or have little energy?",
    "Have you had a poor appetite or have been overeating?",
    "Have you been feeling bad about yourself, or that you are a failure or have let yourself or family down?",
    "Have you had trouble concentrating on things, such as reading the newspaper or watching TV?",
    "Have you been speaking or moving so slowly that other people could have noticed? Or the opposite – have you been so fidgety or restless that you have been moving around a lot more than usual?",
    "Have you had thoughts that you would be better off dead, or of hurting yourself in some way?"
    
]

# Options for answers
total_score_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Function to interpret total score
def interpret_score(score):
    if 1 <= score <= 4:
        return "No depression"
    elif 5 <= score <= 9:
        return "Mild depression"
    elif 10 <= score <= 14:
        return "Moderate depression"
    elif 15 <= score <= 19:
        return "Moderately severe depression"
    elif 20 <= score <= 27:
        return "Severe depression"
    else:
        return "Invalid Score"

import os

@app.route('/depression_screen', methods=['GET', 'POST'])
def depression_screen():
    PHQ9_QUESTIONS = [
        "Have you had little interest or pleasure in doing things?",
        "Have you been feeling down, depressed or hopeless?",
        "Have you had trouble falling or staying asleep or sleeping too much?",
        "Have you been feeling tired or have little energy?",
        "Have you had a poor appetite or have been overeating?",
        "Have you been feeling bad about yourself, or that you are a failure or have let yourself or family down?",
        "Have you had trouble concentrating on things, such as reading the newspaper or watching TV?",
        "Have you been speaking or moving so slowly that other people could have noticed? Or the opposite – have you been so fidgety or restless that you have been moving around a lot more than usual?",
        "Have you had thoughts that you would be better off dead, or of hurting yourself in some way?"
    ]

    total_score_mapping = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}

    advice_mapping = {
        "No depression": """
        Healthy Mindset Maintenance:
        - Maintain a balanced lifestyle to prevent future mental stress.
        - Develop positive habits that support emotional well-being.

        Steps:
        1. Exercise Regularly – At least 30 minutes of walking, yoga, or any physical activity.
        2. Practice Mindfulness – Meditation, gratitude journaling, or breathing exercises.
        3. Stay Socially Connected – Spend time with family and friends.
        4. Engage in Hobbies – Reading, painting, music, or anything enjoyable.
        5. Get Quality Sleep – Maintain a proper sleep schedule (7-9 hours).
        6. Eat Healthy – Avoid excessive junk food; prefer nutritious meals.
        """,
        "Mild depression": """
        Mild Depression:
        - Identify triggers and manage stress.
        - Adopt positive coping mechanisms.

        Steps:
        1. Acknowledge Your Feelings – Accept emotions without judgment.
        2. Talk to Someone – Share thoughts with a trusted person.
        3. Engage in Relaxation Techniques – Deep breathing, guided meditation, or progressive muscle relaxation.
        4. Reduce Screen Time – Avoid excessive social media and news consumption.
        5. Make Small Lifestyle Changes – Wake up early, set small goals, and celebrate small wins.
        6. Seek Support Groups – Join mental wellness communities online or offline.
        """,
        "Moderate depression": """
        Moderate Depression:
        - Focus on structured daily routines and therapy techniques.
        - Seek professional advice if symptoms persist.

        Steps:
        1. Set a Daily Routine – Structure your day with planned activities.
        2. Challenge Negative Thoughts – Use Cognitive Behavioral Therapy (CBT) exercises to reframe thinking.
        3. Practice Journaling – Write down emotions and reflect on positive moments.
        4. Limit Alcohol and Caffeine – These can worsen mood swings.
        5. Try Light Therapy – Exposure to sunlight or artificial bright light can boost mood.
        6. Seek Counseling – If symptoms continue for more than two weeks, consult a mental health professional.
        """,
        "Moderately severe depression": """
        Moderately Severe Depression:
        - Prioritize professional therapy and support systems.
        - Take steps to prevent worsening conditions.

        Steps:
        1. Seek Professional Help – Consult a psychologist or therapist.
        2. Try Psychotherapy (CBT or Talk Therapy) – Helps in managing negative thought patterns.
        3. Build a Strong Support Network – Family, close friends, or support groups.
        4. Limit Isolation – Engage in social activities, even if it's difficult.
        5. Consider Medication (If Prescribed) – Only under medical supervision.
        6. Use Emergency Helplines – Reach out if feeling overwhelmed or suicidal.
        """,
        "Severe depression": """
        Severe Depression (Immediate Help Required):
        - Get urgent medical and professional support.
        - Avoid self-isolation and harmful thoughts.

        Steps:
        1. Contact a Mental Health Helpline – Immediate professional guidance.
           - India: Vandrevala Foundation (1860 266 2345), Snehi (91-9582208181)
           - USA: National Suicide Prevention Lifeline (988)
           - UK: Samaritans (116 123)
        2. Inform a Trusted Person – Let a close friend or family member know about your situation.
        3. Avoid Being Alone – Stay around people or in a safe environment.
        4. Seek Immediate Psychiatric Care – If experiencing suicidal thoughts or severe distress.
        5. Follow a Crisis Plan – Have an emergency action plan in case of emotional breakdowns.
        6. Take Medical Treatment (If Prescribed) – Follow up with a psychiatrist for proper medication if needed.
        """
    }

    if request.method == 'POST':
        username = session.get('username', 'Unknown')
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        responses = {f'question_{i}': request.form.get(f'question_{i}', "Not at all") for i in range(9)}
        total_score = sum(total_score_mapping.get(responses[f'question_{i}'], 0) for i in range(9))

        # Determine depression level
        def interpret_score(score):
            if score <= 4:
                return "No depression"
            elif score <= 9:
                return "Mild depression"
            elif score <= 14:
                return "Moderate depression"
            elif score <= 19:
                return "Moderately severe depression"
            else:
                return "Severe depression"

        result = interpret_score(total_score)
        advice = advice_mapping.get(result, "")

        # Identify disorder type
        shaded_positives = sum(total_score_mapping.get(responses[f'question_{i}']) >= 2 for i in range(9))
        disorder_type = None
        if shaded_positives >= 5 and any(responses[f'question_{j}'] in ['More than half the days', 'Nearly every day'] for j in [0, 1]):
            disorder_type = "Major Depressive Disorder"
        elif 2 <= shaded_positives <= 4 and any(responses[f'question_{j}'] in ['More than half the days', 'Nearly every day'] for j in [0, 1]):
            disorder_type = "Other Depressive Disorder"

        # Save data to an Excel file
        file_path = "depression_results.xlsx"
        new_data = {
            "Username": username,
            "Timestamp": timestamp,
            "File Path": file_path,
            "Question 1": responses['question_0'],
            "Question 2": responses['question_1'],
            "Question 3": responses['question_2'],
            "Question 4": responses['question_3'],
            "Question 5": responses['question_4'],
            "Question 6": responses['question_5'],
            "Question 7": responses['question_6'],
            "Question 8": responses['question_7'],
            "Question 9": responses['question_8'],
            "Total Score": sum(total_score_mapping.get(request.form.get(f'question_{i}', "Not at all"), 0) for i in range(9)),
            "Depression Level": interpret_score(sum(total_score_mapping.get(request.form.get(f'question_{i}', "Not at all"), 0) for i in range(9))),
            "Disorder Type": "Major Depressive Disorder" if sum(total_score_mapping.get(request.form.get(f'question_{i}', "Not at all"), 0) for i in range(9)) >= 5 else "Other Depressive Disorder" if 2 <= sum(total_score_mapping.get(request.form.get(f'question_{i}', "Not at all"), 0) for i in range(9)) <= 4 else "None"
        }

        # Check if the file exists and append new data
        if os.path.exists(file_path):
            existing_data = pd.read_excel(file_path)
            updated_data = pd.concat([existing_data, pd.DataFrame([new_data])], ignore_index=True)
        else:
            updated_data = pd.DataFrame([new_data])

        updated_data.to_excel(file_path, index=False)

        return render_template('result.html', score=total_score, result=result, disorder_type=disorder_type, advice=advice)

    return render_template('phq9.html', questions=PHQ9_QUESTIONS)
from flask import Flask, render_template, Response, redirect, url_for
import cv2
import imutils
import numpy as np
import time
import pandas as pd
import base64
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
import getpass 

emotion_data = []
capture_duration = 15  # seconds


# Get the system username
import getpass
# Get the system username
username = getpass.getuser()
print(f"Captured Username: {username}")  # Debugging

# Create a directory for storing results
save_dir = os.path.join(os.getcwd(), "emotion_tracking_results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Generate timestamped filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f"emotion_tracking_{username}_{timestamp}.xlsx"
file_path = os.path.join(save_dir, file_name)

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        # If camera cannot be opened, yield a blank frame with message
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    start_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=5, minSize=(30, 30))

        canvas = np.zeros((frame.shape[0], 300, 3), dtype="uint8")

        for (fX, fY, fW, fH) in faces:
            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            accuracy = emotion_probability * 100

            current_time = time.time()
            if current_time - start_time >= capture_duration:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                print(f"Appending Data: {timestamp}, {label}, {accuracy}, {username}")  # Debugging
                emotion_data.append([timestamp, label, accuracy, username])

                # Save data with additional details
                df = pd.DataFrame(emotion_data, columns=['Timestamp', 'Emotion', 'Accuracy', 'Username'])
                df.to_excel(file_path, index=False)

                print(f"Emotion data saved to: {file_path}")
                camera.release()
                return

            cv2.putText(frame, f"{label}: {accuracy:.2f}%", (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0, 0, 255), 2)

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = f"{emotion}: {prob*100:.2f}%"
                w = int(prob * 250)
                cv2.rectangle(canvas, (7, i*35 + 5), (w, i*35 + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, i*35 + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        combined_frame = np.hstack([frame, canvas])

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        combined_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

    try:
        camera.release()
    except Exception:
        pass

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload_image1')
def upload_image1():
    return render_template('result1.html', emotion_data=emotion_data)


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """Accept a base64-encoded image from the browser, analyze emotion, and return label and accuracy."""
    try:
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get('image')
        if not image_b64:
            return jsonify({"error": "missing image"}), 400

        # Data URL format handling
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]

        image_bytes = base64.b64decode(image_b64)
        np_image = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        result = {"faces": len(faces), "emotion": None, "accuracy": None}

        if len(faces) > 0:
            # Use first face for quick feedback
            (fX, fY, fW, fH) = faces[0]
            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = float(np.max(preds))
            label = EMOTIONS[int(np.argmax(preds))]

            result["emotion"] = label
            result["accuracy"] = round(emotion_probability * 100.0, 2)

            # Optionally append to session list for summary
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            emotion_data.append([timestamp, label, result["accuracy"], username])

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


label_encoder_one = pickle.load(open("label_encoder_one.pkl", "rb"))
vectorizer_one = pickle.load(open("tfidf_vectorizer_one.pkl", "rb"))
model_one = pickle.load(open("model_one.pkl", "rb"))

def preprocess_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = " ".join(word.strip(".,!?()[]{}") for word in text.split())
    return text

@app.route('/prediction_one', methods=['GET', 'POST'])
def prediction_one():
    if request.method == 'POST':
        input_text = request.form['statement']
        cleaned_text = preprocess_text(input_text)
        transformed_text = vectorizer_one.transform([cleaned_text])
        prediction = model_one.predict(transformed_text)
        output_label = label_encoder_one.inverse_transform(prediction)[0]
        return render_template('service_one.html', prediction_text=f'Mental Health Status: {output_label}')
    # Handle GET request by rendering the page
    return render_template('service_one.html', prediction_text=None)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
