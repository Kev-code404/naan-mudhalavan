from deepface import DeepFace
import cv2
import numpy as np
import logging

class VisualAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.emotion_mapping = {
            'angry': 'anger',
            'disgust': 'anger',
            'fear': 'anxiety',
            'happy': 'happiness',
            'sad': 'sadness',
            'surprise': 'anxiety',
            'neutral': 'calm'
        }

    def analyze(self, image_data):
        try:
            # Convert image data to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Analyze face and emotions
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]  # Take the first face if multiple faces detected
            
            # Extract emotion scores
            emotion_scores = result['emotion']
            
            # Map DeepFace emotions to our emotion categories
            mapped_emotions = {
                'happiness': emotion_scores.get('happy', 0) / 100,
                'sadness': emotion_scores.get('sad', 0) / 100,
                'anxiety': (emotion_scores.get('fear', 0) + emotion_scores.get('surprise', 0)) / 200,
                'anger': (emotion_scores.get('angry', 0) + emotion_scores.get('disgust', 0)) / 200,
                'calm': emotion_scores.get('neutral', 0) / 100
            }
            
            # Calculate mental health score based on emotions
            # Higher happiness and calm = higher score
            # Higher sadness, anxiety, and anger = lower score
            happiness_weight = 0.4
            calm_weight = 0.3
            sadness_weight = -0.2
            anxiety_weight = -0.1
            anger_weight = -0.1
            
            # Calculate weighted score (0-1)
            weighted_score = (
                mapped_emotions['happiness'] * happiness_weight +
                mapped_emotions['calm'] * calm_weight +
                mapped_emotions['sadness'] * sadness_weight +
                mapped_emotions['anxiety'] * anxiety_weight +
                mapped_emotions['anger'] * anger_weight
            ) + 0.5  # Add 0.5 to center the score around 0.5
            
            # Normalize to 0-1 range
            normalized_score = max(0, min(1, weighted_score))
            
            # Convert to mental health score (5-25)
            mental_health_score = 5 + (normalized_score * 20)
            
            return {
                'emotions': mapped_emotions,
                'mental_health_score': round(mental_health_score),
                'mental_health_status': self._get_mental_health_status(mental_health_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in visual analysis: {str(e)}")
            # Return default values in case of error
            return {
                'emotions': {
                    'happiness': 0.5,
                    'sadness': 0.3,
                    'anxiety': 0.2,
                    'anger': 0.1,
                    'calm': 0.4
                },
                'mental_health_score': 15,
                'mental_health_status': 'Neutral - Unable to analyze facial expressions'
            }

    def _get_mental_health_status(self, score):
        if score >= 22: return 'Mentally Healthy - Emotionally aware, good coping mechanisms'
        if score >= 18: return 'Stable but Vulnerable - Slight signs of worry, minor detachment'
        if score >= 14: return 'Mild Emotional Imbalance - Mood swings, early signs of stress'
        if score >= 10: return 'Moderate Issues Detected - Anxiety, emotional numbness'
        if score >= 6: return 'Severe Mental Distress - Major depression, trauma, PTSD'
        return 'Critical / Emergency - Severe emotional distress' 