from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import base64
from urllib.parse import parse_qs, urlparse
import traceback
from collections import deque
import tempfile
import os
from gtts import gTTS
import base64

print("Loading model...")
# Load the trained model
model = tf.keras.models.load_model('models/actionModel.keras')
print("Model loaded successfully!")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the actions that the model was trained on
# You may need to adjust this based on your actual training data
actions = np.array(['Alright', 'Good Afternoon', 'Good Evening', 'Good Morning', 'Good Night', 'Hello', 'How Are You', 'Pleased', 'Thank You'])
print(f"Actions configured: {actions}")

# Create a prediction history for smoothing
prediction_history = deque(maxlen=5)
last_prediction = None
prediction_threshold = 0.5  # Lower threshold for accepting predictions

# Store frames for sequence prediction
sequence_length = 10  # Reduced from 30 to 15 frames for faster collection
frame_buffer = {}  # Dictionary to store frame sequences for each client
frame_skip = 0  # Process every frame (no skipping)

# MediaPipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extract keypoints function
def extract_keypoints(results):
    # Debug info
    has_pose = results.pose_landmarks is not None
    has_face = results.face_landmarks is not None
    has_left_hand = results.left_hand_landmarks is not None
    has_right_hand = results.right_hand_landmarks is not None
    print(f"Landmarks detected: Pose={has_pose}, Face={has_face}, Left hand={has_left_hand}, Right hand={has_right_hand}")
    
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Smooth predictions to avoid flickering
def smooth_prediction(new_prediction):
    global last_prediction
    
    # Add the new prediction to history
    prediction_history.append(new_prediction)
    
    # Count occurrences of each action in the history
    action_counts = {}
    for pred in prediction_history:
        action = pred['action']
        confidence = pred['confidence']
        if action in action_counts:
            action_counts[action]['count'] += 1
            action_counts[action]['total_confidence'] += confidence
        else:
            action_counts[action] = {
                'count': 1,
                'total_confidence': confidence
            }
    
    # Find the most frequent action
    max_count = 0
    max_action = None
    max_confidence = 0
    
    for action, data in action_counts.items():
        if data['count'] > max_count:
            max_count = data['count']
            max_action = action
            max_confidence = data['total_confidence'] / data['count']
        elif data['count'] == max_count and data['total_confidence'] / data['count'] > max_confidence:
            max_action = action
            max_confidence = data['total_confidence'] / data['count']
    
    # Only change prediction if we have a stable new one or no previous prediction
    if last_prediction is None or (max_count >= 2 and max_action != last_prediction['action']):
        last_prediction = {
            'action': max_action,
            'confidence': max_confidence
        }
    
    return last_prediction

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        if self.path == '/':
            self._serve_file('index.html')
        elif self.path.startswith('/js/'):
            self._serve_file(self.path[1:])
        elif self.path.startswith('/css/'):
            self._serve_file(self.path[1:])
        elif self.path == '/app.html':
            self._serve_file('app.html')
        else:
            self.send_response(404)
            self.end_headers()
    
    def _serve_file(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
                
            if file_path.endswith('.html'):
                self._set_headers('text/html')
            elif file_path.endswith('.js'):
                self._set_headers('application/javascript')
            elif file_path.endswith('.css'):
                self._set_headers('text/css')
            else:
                self._set_headers('application/octet-stream')
                
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict':
            self._handle_predict()
        elif self.path == '/stop':
            self._handle_stop()
        elif self.path == '/tts':
            self._handle_tts()
        else:
            self.send_response(404)
            self.end_headers()
    
    def _handle_stop(self):
        """Handle client stop requests by cleaning up resources"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            client_id = data.get('clientId', 'default')
            
            # Clean up client resources
            if client_id in frame_buffer:
                del frame_buffer[client_id]
                print(f"Cleaned up resources for client {client_id}")
            
            # Send success response
            self._set_headers('application/json')
            response = {
                'success': True,
                'message': 'Client stopped and resources cleaned up'
            }
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error handling stop request: {str(e)}")
            self._set_headers('application/json')
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def _handle_predict(self):
        """Handle prediction requests"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            image_data = data['image']
            client_id = data.get('clientId', 'default')  # Get client ID or use default
            
            # Skip frames if needed (for performance)
            global frame_skip
            if frame_skip > 0:
                frame_skip -= 1
                # Return last prediction if available
                if last_prediction is not None:
                    response = last_prediction.copy()
                    response['success'] = True
                    response['skipped'] = True
                    self._set_headers('application/json')
                    self.wfile.write(json.dumps(response).encode())
                    return
            
            print(f"Processing frame from client {client_id}")
            
            # Decode base64 image
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Resize image for faster processing (optional)
            img = cv2.resize(img, (320, 240))
            
            # Process with MediaPipe
            with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
                image, results = mediapipe_detection(img, holistic)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                
                # Initialize frame buffer for this client if it doesn't exist
                if client_id not in frame_buffer:
                    frame_buffer[client_id] = []
                
                # Quality check for landmarks
                min_pose_landmarks = 10
                min_hand_landmarks = 5
                
                pose_detected = results.pose_landmarks is not None and len([lm for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]) >= min_pose_landmarks
                lh_detected = results.left_hand_landmarks is not None and len(results.left_hand_landmarks.landmark) >= min_hand_landmarks
                rh_detected = results.right_hand_landmarks is not None and len(results.right_hand_landmarks.landmark) >= min_hand_landmarks
                
                # Only add frames with sufficient landmarks
                if pose_detected or lh_detected or rh_detected:
                    # Add keypoints to the frame buffer
                    frame_buffer[client_id].append(keypoints)
                    
                    # Keep only the last sequence_length frames
                    if len(frame_buffer[client_id]) > sequence_length:
                        frame_buffer[client_id] = frame_buffer[client_id][-sequence_length:]
                    
                    # If we have enough frames, make a prediction
                    if len(frame_buffer[client_id]) == sequence_length:
                        try:
                            # Create sequence of frames
                            sequence = np.array(frame_buffer[client_id])
                            
                            # Reshape for model input (add batch dimension)
                            # The model expects input with shape (batch_size, sequence_length, num_features)
                            sequence = np.expand_dims(sequence, axis=0)
                            
                            # Predict action
                            prediction = model.predict(sequence, verbose=0)[0]
                            
                            # Get top 3 predictions for debugging
                            top_indices = np.argsort(prediction)[-3:][::-1]
                            for i in top_indices:
                                print(f"  {actions[i]}: {prediction[i]:.4f}")
                            
                            predicted_class = np.argmax(prediction)
                            predicted_action = actions[predicted_class]
                            confidence = float(prediction[predicted_class])
                            print(f"Predicted action: {predicted_action} with confidence: {confidence}")
                            
                            # Only accept predictions with confidence above threshold
                            if confidence > prediction_threshold:
                                raw_response = {
                                    'action': predicted_action,
                                    'confidence': confidence,
                                    'success': True
                                }
                                
                                # Apply smoothing to avoid flickering predictions
                                response = smooth_prediction(raw_response)
                                response['success'] = True
                            else:
                                if last_prediction is not None:
                                    # Use last stable prediction if current confidence is low
                                    response = last_prediction.copy()
                                    response['success'] = True
                                    print(f"Using previous prediction: {response['action']}")
                                else:
                                    response = {
                                        'success': False,
                                        'error': 'Low confidence prediction'
                                    }
                            
                            # After successful prediction, we can skip some frames for performance
                            frame_skip = 0  # No frame skipping for now
                            
                        except Exception as e:
                            print(f"Error during prediction: {str(e)}")
                            traceback.print_exc()
                            response = {
                                'success': False,
                                'error': f"Prediction error: {str(e)}"
                            }
                    else:
                        # Still collecting frames
                        frames_collected = len(frame_buffer[client_id])
                        frames_needed = sequence_length - frames_collected
                        print(f"Collecting frames: {frames_collected}/{sequence_length} ({frames_needed} more needed)")
                        
                        if last_prediction is not None:
                            # Use last stable prediction while collecting frames
                            response = last_prediction.copy()
                            response['success'] = True
                            response['collecting'] = True
                            response['frames_collected'] = frames_collected
                            response['frames_needed'] = frames_needed
                        else:
                            response = {
                                'success': False,
                                'collecting': True,
                                'frames_collected': frames_collected,
                                'frames_needed': frames_needed,
                                'error': 'Still collecting frames for prediction'
                            }
                else:
                    print("No sufficient landmarks detected")
                    response = {
                        'success': False,
                        'error': 'Not enough landmarks detected for prediction'
                    }
            
            self._set_headers('application/json')
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            traceback.print_exc()
            self._set_headers('application/json')
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def _handle_tts(self):
        """Handle text-to-speech requests for multiple languages"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            text = data.get('text', '')
            language = data.get('language', 'en')
            
            # Map language codes to Google TTS language codes
            # Note: gTTS doesn't directly support Kannada (kn) and Marathi (mr)
            # Using Hindi for Marathi and English for Kannada as fallbacks
            language_map = {
                'en': 'en',     # English
                'hi': 'hi',     # Hindi
                'kn': 'en-IN',  # Kannada - fallback to Indian English
                'mr': 'hi'      # Marathi - fallback to Hindi (closest available)
            }
            
            # Use default (English) if language not supported
            tts_lang = language_map.get(language, 'en')
            
            print(f"Converting text to speech: '{text}' in language '{tts_lang}'")
            
            # For Kannada and Marathi, we'll add a note about fallback
            fallback_note = ""
            if language == 'kn':
                fallback_note = "(Using Indian English as fallback for Kannada)"
            elif language == 'mr':
                fallback_note = "(Using Hindi as fallback for Marathi)"
            
            if fallback_note:
                print(fallback_note)
            
            # Use BytesIO instead of a temporary file to avoid file access issues
            from io import BytesIO
            mp3_fp = BytesIO()
            
            # Generate the speech audio directly to memory
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Convert to base64
            audio_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
            
            # Send the audio data back to the client
            self._set_headers('application/json')
            response = {
                'success': True,
                'audio': audio_data,
                'format': 'mp3',
                'language': tts_lang,
                'fallback': fallback_note if fallback_note else None
            }
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error processing TTS request: {str(e)}")
            traceback.print_exc()
            self._set_headers('application/json')
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
    
def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Server running on http://localhost:{port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
