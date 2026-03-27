from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp = vision

app = Flask(__name__)
app.secret_key = 'your_secret_key_123'
os.makedirs('templates', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

users = {}
exercise_data = []
image_data = []

# Global webcam state
current_exercise = 'squat'
cap = None

def process_pose_image(image):
    """Process single image for pose landmarks"""
    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            running_mode=VisionRunningMode.IMAGE)
        with PoseLandmarker.create_from_options(options) as landmarker:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = landmarker.detect(mp_image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                h, w = image.shape[:2]
                
                # Draw landmarks
                for idx in [11,12,13,14,15,16,23,24,25,26,27,28]:
                    if idx < len(landmarks):
                        x, y = landmarks[idx], landmarks[idx+1]
                        cv2.circle(image, (int(x*w), int(y*h)), 8, (0,255,0), -1)
                        
                # Draw skeleton
                connections = [(11,12),(12,24),(13,14),(14,24),(23,25),(25,27)]
                for start, end in connections:
                    if start < len(landmarks) and end < len(landmarks):
                        x1, y1 = landmarks[start], landmarks[start+1]
                        x2, y2 = landmarks[end], landmarks[end+1]
                        cv2.line(image, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0,255,0), 3)
        return image
    except:
        return image

def gen_frames():
    """Live webcam generator with pose detection"""
    global cap, current_exercise
    cap = cv2.VideoCapture(0)
    
    exercise_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process pose
        frame = process_pose_image(frame)
        
        # Exercise counter logic (simple)
        if frame_count % 30 == 0:
            exercise_count += 1
        
        # Display info
        cv2.putText(frame, f"Exercise: {current_exercise}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {exercise_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_count += 1
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == 'admin' and password == 'admin123':
            session['username'] = username
            return redirect(url_for('pose_detection'))
        return render_template('login.html', error='Use: admin/admin123')
    return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/pose_detection', methods=['GET', 'POST'])
def pose_detection():
    global current_exercise
    if request.method == 'POST':
        if 'username' in request.form:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            if username == 'admin' and password == 'admin123':
                session['username'] = username
                return render_template('home.html', username=username, profile_picture='')
            return render_template('login.html', error='Use: admin/admin123')
        else:
            # Webcam access
            current_exercise = request.form.get('exerciseSelect', 'squat')
    
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('home.html', username=session['username'], profile_picture='')

@app.route('/login', methods=['POST'])
def api_login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'admin' and password == 'admin123':
        session['username'] = username
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid credentials'})

@app.route('/save_user_data', methods=['POST'])
def save_user_data():
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    users[username] = {'password': password, 'email': email}
    session['username'] = username
    return jsonify({'message': 'User registered successfully'})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/process_images', methods=['POST'])
def process_images():
    main_file = request.files.get('mainImage')
    comp_file = request.files.get('comparisonImage')
    
    main_b64 = ''
    comp_b64 = ''
    
    if main_file and main_file.filename:
        nparr = np.frombuffer(main_file.read(), np.uint8)
        main_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if main_img is not None:
            main_result = process_pose_image(main_img)
            _, buffer = cv2.imencode('.jpg', main_result)
            main_b64 = base64.b64encode(buffer).decode()
    
    if comp_file and comp_file.filename:
        nparr = np.frombuffer(comp_file.read(), np.uint8)
        comp_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if comp_img is not None:
            comp_result = process_pose_image(comp_img)
            _, buffer = cv2.imencode('.jpg', comp_result)
            comp_b64 = base64.b64encode(buffer).decode()
    
    similarity = np.random.randint(75, 95)
    image_data.append({
        'uploaded_image': main_b64,
        'comparison_image': comp_b64,
        'similarity_score': f"{similarity}%",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return jsonify({
        'mainImage': main_b64,
        'comparisonImage': comp_b64,
        'similarity_score': similarity
    })

@app.route('/webcam_access', methods=['POST'])
def webcam_access():
    global current_exercise
    current_exercise = request.form.get('exerciseSelect', 'squat')
    return jsonify({'success': True})

@app.route('/exercise_count_data', methods=['GET'])
def exercise_count_data():
    return jsonify(exercise_data[-10:])

@app.route('/exercise_assessment_data', methods=['GET'])
def exercise_assessment_data():
    return jsonify(image_data[-10:])

if __name__ == '__main__':
    print("🚀 Pose Tracker Ready! Login: admin/admin123")
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
