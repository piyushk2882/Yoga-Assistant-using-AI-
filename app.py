from flask import Flask, render_template, Response
import numpy as np
import cv2
import PoseModule as pm
import tensorflow as tf
import tensorflow_hub as hub
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import deque
from datetime import datetime
import json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']
detector = pm.PoseDetector()
speech_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=2)


# Add this after the global variables
class PoseAccuracyTracker:
    def __init__(self, max_points=50):
        self.accuracies = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.current_pose = "None"
        self.total_duration = 0
        self.start_time = None
        self.is_tracking = False
        self.pose_thresholds = {
            "Vrkasana": {  # Tree Pose
                'right_leg': 180,
                'left_leg': 180,
                'right_arm': 160,
                'left_arm': 160
            },
            "Warrior": {
                'right_leg': 160,
                'left_leg': 90,
                'right_arm': 180,
                'left_arm': 180
            }
            # Add more poses with their angle thresholds
        }

    def detect_pose(self, angles):
        for pose_name, thresholds in self.pose_thresholds.items():
            matches = 0
            for part, expected_angle in thresholds.items():
                if part in angles:
                    accuracy, is_accurate = compare_pose(angles[part], expected_angle)
                    if is_accurate:
                        matches += 1
            
            if matches >= len(thresholds) * 0.7:  # 70% confidence threshold
                if self.current_pose != pose_name:
                    self.reset()
                    self.current_pose = pose_name
                    self.start_time = datetime.now()
                return True
        return False

    def add_accuracy(self, accuracy):
        if not self.is_tracking:
            self.is_tracking = True
            self.start_time = datetime.now()
        
        self.accuracies.append(accuracy)
        self.timestamps.append((datetime.now() - self.start_time).seconds)
        
    def get_data(self):
        return {
            'timestamps': list(self.timestamps),
            'accuracies': list(self.accuracies),
            'pose': self.current_pose,
            'duration': (datetime.now() - self.start_time).seconds if self.start_time else 0,
            'average_accuracy': sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0,
            'is_tracking': self.is_tracking
        }

    def reset(self):
        self.accuracies.clear()
        self.timestamps.clear()
        self.start_time = None
        self.is_tracking = False
        self.current_pose = "None"

# Add this after the global variables
pose_tracker = PoseAccuracyTracker()

# Camera setup
def setup_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height
    return cap

# Drawing functions
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

# Pose comparison functions
def compare_pose(actual_angle, expected_angle, threshold=10):
    if actual_angle <= expected_angle:
        accuracy = (actual_angle / expected_angle) * 100
    else:
        accuracy = 0
    
    is_accurate = abs(expected_angle - actual_angle) <= threshold
    return accuracy, is_accurate

def process_frame(frame, detector):
    try:
        frame = detector.findPose(frame, False)
        lmlist = detector.getPosition(frame, False)
        
        if len(lmlist) != 0:
            angles = {
                'right_arm': int(detector.findAngle(frame, 12, 14, 16)),
                'left_arm': int(detector.findAngle(frame, 11, 13, 15)),
                'right_leg': int(detector.findAngle(frame, 24, 26, 28)),
                'left_leg': int(detector.findAngle(frame, 23, 25, 27))
            }
            
            # Automatically detect pose
            pose_detected = pose_tracker.detect_pose(angles)
            
            # Calculate accuracy if pose is detected
            if pose_detected:
                total_accuracy = 0
                valid_angles = 0
                pose_thresholds = pose_tracker.pose_thresholds[pose_tracker.current_pose]
                
                for part, angle in angles.items():
                    if part in pose_thresholds:
                        accuracy, is_accurate = compare_pose(angle, pose_thresholds[part])
                        if accuracy > 0:
                            total_accuracy += accuracy
                            valid_angles += 1
                        color = (0, 255, 0) if is_accurate else (0, 0, 255)
                        cv2.putText(frame, f"{part}: {accuracy:.1f}%", 
                                  (10, 30 + list(angles.keys()).index(part) * 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Update pose tracker with overall accuracy
                if valid_angles > 0:
                    overall_accuracy = total_accuracy / valid_angles
                    pose_tracker.add_accuracy(overall_accuracy)
                
                # Add pose name and tracking status to frame
                cv2.putText(frame, f"Pose: {pose_tracker.current_pose}", 
                          (10, frame.shape[0] - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Tracking: Active", 
                          (10, frame.shape[0] - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", 
                          (10, frame.shape[0] - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame


def generate_frames():
    cap = setup_camera()
    frame_skip = 2
    frame_count = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            frame = cv2.flip(frame, 1)
            frame = process_frame(frame, detector)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        cap.release()

# Routes
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/tracks')
def tracks():
    return render_template('tracks.html')

@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/get_accuracy_data')
def get_accuracy_data():
    return json.dumps(pose_tracker.get_data())

@app.route('/reset_tracking')
def reset_tracking():
    pose_tracker.reset()
    return json.dumps({'status': 'success'})

@app.route('/set_pose/<pose_name>')
def set_pose(pose_name):
    pose_tracker.set_pose(pose_name)
    return json.dumps({'status': 'success'})



@app.route('/charts')
def charts():
    # Simplified chart data
    values = [80, 85, 90, 88, 92, 87]  # Example accuracy values
    labels = ['Session 1', 'Session 2', 'Session 3', 'Session 4', 'Session 5', 'Session 6']
    colors = ['#ff0000', '#0000ff', '#ffffe0', '#008000', '#800080', '#FFA500']
    return render_template('charts.html', values=values, labels=labels, colors=colors)

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, threaded=True)