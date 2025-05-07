import cv2
import numpy as np
from deepface import DeepFace
import logging
import time
from threading import Thread
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.cap = None
        self.last_analysis_time = 0
        self.analysis_interval = 1.0  # Analyze every 1 second
        self.current_emotion = {
            'dominant_emotion': 'neutral',
            'emotion_scores': {'neutral': 100},
            'confidence': 100
        }

    def start(self):
        """Start the face recognition thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Face recognition started")

    def stop(self):
        """Stop the face recognition thread"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info("Face recognition stopped")

    def _run(self):
        """Main face recognition loop"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    continue

                current_time = time.time()
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    self._analyze_frame(frame)
                    self.last_analysis_time = current_time

                # Add a small delay to prevent high CPU usage
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

    def _analyze_frame(self, frame):
        """Analyze a single frame for facial emotions"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = frame[y:y+h, x:x+w]

                # Analyze emotions using DeepFace
                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False
                )

                # Get emotion scores
                emotion_scores = result[0]['emotion']
                dominant_emotion = result[0]['dominant_emotion']
                
                # Convert scores to percentages
                total_score = sum(emotion_scores.values())
                emotion_percentages = {
                    emotion: (score / total_score) * 100 
                    for emotion, score in emotion_scores.items()
                }

                self.current_emotion = {
                    'dominant_emotion': dominant_emotion,
                    'emotion_scores': emotion_percentages,
                    'confidence': emotion_percentages[dominant_emotion]
                }

                # Put the result in the queue
                self.emotion_queue.put(self.current_emotion)

        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")

    def get_current_emotion(self):
        """Get the current emotion analysis"""
        try:
            # Get the latest emotion from the queue
            while not self.emotion_queue.empty():
                self.current_emotion = self.emotion_queue.get_nowait()
            return self.current_emotion
        except queue.Empty:
            return self.current_emotion

    def get_frame(self):
        """Get the current camera frame"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None 