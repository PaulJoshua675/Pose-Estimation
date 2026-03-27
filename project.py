from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_cors import CORS
import cv2
import numpy as np
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

LANDMARKS = {
    'RIGHT_SHOULDER': 12, 'RIGHT_ELBOW': 14, 'RIGHT_WRIST': 16,
    'RIGHT_HIP': 24, 'RIGHT_KNEE': 26, 'RIGHT_ANKLE': 28,
    'LEFT_SHOULDER': 11, 'LEFT_ELBOW': 13, 'LEFT_WRIST': 15,
    'LEFT_HIP': 23, 'LEFT_KNEE': 25, 'LEFT_ANKLE': 27
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_squat(hip, knee, ankle):
    return calculate_angle(hip, knee, ankle) <= 90

def is_push_up(shoulder, elbow, hip, knee):
    return calculate_angle(shoulder, elbow, hip) <= 90 and calculate_angle(hip, knee, hip) <= 90

def is_leg_raise(hip, knee, ankle, shoulder):
    return 150 < calculate_angle(hip, knee, ankle) <= 210 and 60 < calculate_angle(shoulder, hip, knee) <= 120

def is_sit_up(shoulder, hip, knee):
    return 60 < calculate_angle(shoulder, hip, knee) <= 120

def is_tadasana(shoulder, hip, knee, ankle, wrist):
    return (150 < calculate_angle(wrist, shoulder, hip) <= 210 and 
            150 < calculate_angle(shoulder, hip, knee) <= 210 and 
            150 < calculate_angle(hip, knee, ankle) <= 210)

def is_bridge(shoulder, hip, knee, ankle):
    return (150 <= calculate_angle(shoulder, hip, knee) <= 230 and 
            50 <= calculate_angle(hip, knee, ankle) <= 120)

def is_kneepush_up(shoulder, elbow, hip, knee, ankle):
    return calculate_angle(shoulder, elbow, hip) <= 90 and calculate_angle(hip, knee, ankle) <= 90

def is_t_pose(shoulder, hip, elbow, ankle):
    return (160 < calculate_angle(shoulder, hip, ankle) < 200 and 
            80 < calculate_angle(elbow, shoulder, hip) < 100)

def gen_frames(exercise="squat"):
    cap = cv2.VideoCapture(0)
    model_path = 'pose_landmarker_lite.task'
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        exercise_count = 0
        frame_counter = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_counter * 33
            
            results = landmarker.detect_for_video(mp_image, [timestamp_ms])
            
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]
                    
                    hip = [landmarks[LANDMARKS['RIGHT_HIP']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_HIP']+1] * frame.shape[0]]
                    knee = [landmarks[LANDMARKS['RIGHT_KNEE']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_KNEE']+1] * frame.shape[0]]
                    ankle = [landmarks[LANDMARKS['RIGHT_ANKLE']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_ANKLE']+1] * frame.shape[0]]
                    shoulder = [landmarks[LANDMARKS['RIGHT_SHOULDER']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_SHOULDER']+1] * frame.shape[0]]
                    elbow = [landmarks[LANDMARKS['RIGHT_ELBOW']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_ELBOW']+1] * frame.shape[0]]
                    wrist = [landmarks[LANDMARKS['RIGHT_WRIST']] * frame.shape[1], landmarks[LANDMARKS['RIGHT_WRIST']+1] * frame.shape[0]]
                    
                    for idx in LANDMARKS.values():
                        x = int(landmarks[idx] * frame.shape[1])
                        y = int(landmarks[idx+1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    connections = [(12,14),(14,16),(24,26),(26,28),(12,24)]
                    for conn in connections:
                        x1 = int(landmarks[conn[0]] * frame.shape[1])
                        y1 = int(landmarks[conn[0]+1] * frame.shape[0])
                        x2 = int(landmarks[conn[1]] * frame.shape[1])
                        y2 = int(landmarks[conn[1]+1] * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    exercise_map = {
                        "squat": is_squat(hip, knee, ankle),
                        "push_up": is_push_up(shoulder, elbow, hip, knee),
                        "leg_raise": is_leg_raise(hip, knee, ankle, shoulder),
                        "sit_up": is_sit_up(shoulder, hip, knee),
                        "tadasana": is_tadasana(shoulder, hip, knee, ankle, wrist),
                        "glute_bridge": is_bridge(shoulder, hip, knee, ankle),
                        "knee_push_up": is_kneepush_up(shoulder, elbow, hip, knee, ankle),
                        "t_pose": is_t_pose(shoulder, hip, elbow, ankle)
                    }
                    
                    if frame_counter % 30 == 0:
                        if exercise_map.get(exercise, False):
                            exercise_count += 1
            
            except:
                pass
            
            cv2.putText(frame, f"{exercise}: {exercise_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            frame_counter += 1
            
    cap.release()

@app.route('/')
def home():
    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == 'test' and password == 'test':
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return redirect(url_for('home', error='Invalid credentials'))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('home'))
    return render_template('dashboard.html')

@app.route('/webcam/<exercise>')
def webcam(exercise):
    if not session.get('logged_in'):
        return redirect(url_for('home'))
    return render_template('webcam.html', exercise=exercise)

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    return Response(gen_frames(exercise), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
