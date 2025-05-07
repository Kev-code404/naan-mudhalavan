import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging

class EmotionDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model_path = 'models/emotion_model.joblib'
        self.scaler_path = 'models/emotion_scaler.joblib'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load pre-trained model if it exists
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.classifier = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logging.info("Loaded pre-trained emotion detection model")
        else:
            logging.warning("No pre-trained model found. Using untrained model.")

    def extract_features(self, audio_path):
        """Extract audio features using librosa."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3)
            
            # Extract features
            features = []
            
            # 1. Pitch features (voice modulation)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_features = {
                'mean_pitch': np.mean(pitches),
                'pitch_std': np.std(pitches),
                'pitch_range': np.max(pitches) - np.min(pitches),
                'pitch_variability': np.std(pitches) / (np.mean(pitches) + 1e-6)
            }
            features.extend([
                pitch_features['mean_pitch'],
                pitch_features['pitch_std'],
                pitch_features['pitch_range'],
                pitch_features['pitch_variability']
            ])
            
            # 2. Energy features (volume and intensity)
            rms = librosa.feature.rms(y=y)[0]
            energy_features = {
                'mean_energy': np.mean(rms),
                'energy_std': np.std(rms),
                'max_energy': np.max(rms),
                'energy_range': np.max(rms) - np.min(rms)
            }
            features.extend([
                energy_features['mean_energy'],
                energy_features['energy_std'],
                energy_features['max_energy'],
                energy_features['energy_range']
            ])
            
            # 3. Tempo features (speaking rate)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_features = {
                'tempo': tempo,
                'tempo_std': np.std(librosa.beat.beat_track(y=y, sr=sr)[1])
            }
            features.extend([
                tempo_features['tempo'],
                tempo_features['tempo_std']
            ])
            
            # 4. Spectral features (voice quality)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_features = {
                'mean_spectral': np.mean(spectral_centroids),
                'spectral_std': np.std(spectral_centroids),
                'spectral_range': np.max(spectral_centroids) - np.min(spectral_centroids)
            }
            features.extend([
                spectral_features['mean_spectral'],
                spectral_features['spectral_std'],
                spectral_features['spectral_range']
            ])
            
            # 5. Zero crossing rate (voice clarity)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_features = {
                'mean_zcr': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'zcr_range': np.max(zcr) - np.min(zcr)
            }
            features.extend([
                zcr_features['mean_zcr'],
                zcr_features['zcr_std'],
                zcr_features['zcr_range']
            ])
            
            # 6. MFCCs (voice characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_features = {
                'mean_mfcc': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1)
            }
            features.extend([
                mfcc_features['mean_mfcc'],
                mfcc_features['mfcc_std']
            ].flatten())
            
            # Calculate voice modulation score
            modulation_score = self._calculate_modulation_score(
                pitch_features,
                energy_features,
                tempo_features,
                spectral_features,
                zcr_features
            )
            
            return np.array(features), modulation_score
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            raise

    def _calculate_modulation_score(self, pitch, energy, tempo, spectral, zcr):
        """Calculate a comprehensive voice modulation score."""
        # Normalize each component to 0-1 range
        pitch_score = min(1.0, pitch['pitch_variability'] * 2)  # Higher variability = better modulation
        energy_score = min(1.0, energy['energy_range'] * 2)  # Higher range = better modulation
        tempo_score = min(1.0, tempo['tempo_std'] / 50)  # Normalize tempo variation
        spectral_score = min(1.0, spectral['spectral_range'] / 1000)  # Normalize spectral range
        zcr_score = min(1.0, zcr['zcr_range'] * 2)  # Higher range = better modulation
        
        # Weight the components
        weights = {
            'pitch': 0.3,
            'energy': 0.2,
            'tempo': 0.2,
            'spectral': 0.15,
            'zcr': 0.15
        }
        
        # Calculate weighted score
        total_score = (
            pitch_score * weights['pitch'] +
            energy_score * weights['energy'] +
            tempo_score * weights['tempo'] +
            spectral_score * weights['spectral'] +
            zcr_score * weights['zcr']
        )
        
        # Convert to 0-100 scale
        return round(total_score * 100, 2)

    def detect_emotion(self, audio_path):
        """Detect emotion from audio file."""
        try:
            # Extract features and modulation score
            features, modulation_score = self.extract_features(audio_path)
            
            # Reshape features for prediction
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict emotion
            emotion_idx = self.classifier.predict(features_scaled)[0]
            emotion = self.emotions[emotion_idx]
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = probabilities[emotion_idx]
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.emotions, probabilities)
                },
                'voice_modulation': {
                    'score': modulation_score,
                    'interpretation': self._interpret_modulation_score(modulation_score)
                }
            }
            
        except Exception as e:
            logging.error(f"Error detecting emotion: {str(e)}")
            raise

    def _interpret_modulation_score(self, score):
        """Interpret the voice modulation score."""
        if score >= 80:
            return "Excellent voice modulation with clear variation in pitch, volume, and pace"
        elif score >= 60:
            return "Good voice modulation with noticeable variation in speech patterns"
        elif score >= 40:
            return "Moderate voice modulation with some variation in speech"
        elif score >= 20:
            return "Limited voice modulation with minimal variation in speech patterns"
        else:
            return "Very limited voice modulation with little variation in speech"

    def train(self, audio_files, labels):
        """Train the emotion detection model."""
        try:
            features = []
            for audio_file in audio_files:
                features.append(self.extract_features(audio_file)[0])  # Only use features, not modulation score
            
            features = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train classifier
            self.classifier.fit(features_scaled, labels)
            
            # Save model and scaler
            joblib.dump(self.classifier, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logging.info("Model trained and saved successfully")
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise 