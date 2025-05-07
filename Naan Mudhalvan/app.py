from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from text_analyzer import TextAnalyzer
from audio_analyzer import AudioAnalyzer
from visual_analyzer import VisualAnalyzer
from recommendation_engine import RecommendationEngine
from emotion_detector import EmotionDetector
from models import db, User
import os
import logging
import traceback
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import whisper
import tempfile
import soundfile as sf
import numpy as np
import io
import subprocess
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

logger.info("Initializing Flask app...")

try:
    logger.info("Initializing TextAnalyzer...")
    text_analyzer = TextAnalyzer()
    logger.info("TextAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TextAnalyzer: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Initializing AudioAnalyzer...")
    audio_analyzer = AudioAnalyzer()
    logger.info("AudioAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AudioAnalyzer: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Initializing VisualAnalyzer...")
    visual_analyzer = VisualAnalyzer()
    logger.info("VisualAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize VisualAnalyzer: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Initializing RecommendationEngine...")
    recommendation_engine = RecommendationEngine()
    logger.info("RecommendationEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RecommendationEngine: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.info("Initializing EmotionDetector...")
    emotion_detector = EmotionDetector()
    logger.info("EmotionDetector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EmotionDetector: {str(e)}")
    logger.error(traceback.format_exc())

# Initialize Whisper model
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    logger.error(traceback.format_exc())
    whisper_model = None

# Initialize model and tokenizer for chatbot
try:
    logger.info("Loading DialoGPT model and tokenizer...")
    chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def convert_webm_to_wav(input_path, output_path):
    """Convert WebM audio to WAV format using ffmpeg."""
    try:
        # Try to find ffmpeg in common installation locations
        ffmpeg_paths = [
            shutil.which('ffmpeg'),  # Check PATH first
            r'C:\Program Files\FFmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'
        ]
        
        ffmpeg_path = None
        for path in ffmpeg_paths:
            if path and os.path.exists(path):
                ffmpeg_path = path
                logger.info(f"Found ffmpeg at: {ffmpeg_path}")
                break
        
        if ffmpeg_path is None:
            raise Exception("ffmpeg is not installed or not found in common locations. Please install ffmpeg to process audio files.")
        
        # Convert WebM to WAV using ffmpeg
        command = [
            ffmpeg_path,
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise Exception(f"ffmpeg conversion failed: {result.stderr}")
        
        logger.info(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise

# Mental health related keywords and responses
MENTAL_HEALTH_KEYWORDS = {
    # Anxiety-related questions
    'anxiety': {
        'what is anxiety': "Anxiety is a natural response to stress, but when it becomes excessive or persistent, it can interfere with daily life. It's characterized by feelings of worry, nervousness, or unease about something with an uncertain outcome.",
        'how to manage anxiety': "Here are some effective ways to manage anxiety:\n1. Practice deep breathing exercises\n2. Try progressive muscle relaxation\n3. Maintain a regular sleep schedule\n4. Exercise regularly\n5. Limit caffeine and alcohol\n6. Practice mindfulness meditation\n7. Keep a worry journal\n8. Challenge negative thoughts\n9. Stay connected with supportive people\n10. Consider professional help if needed",
        'anxiety symptoms': "Common anxiety symptoms include:\n- Excessive worry\n- Restlessness\n- Difficulty concentrating\n- Muscle tension\n- Sleep problems\n- Irritability\n- Rapid heartbeat\n- Sweating\n- Trembling\n- Shortness of breath",
        'anxiety attack': "During an anxiety attack, try these steps:\n1. Find a quiet place\n2. Practice deep breathing (4-7-8 technique)\n3. Use grounding techniques (5-4-3-2-1 method)\n4. Remind yourself it will pass\n5. Focus on the present moment\n6. Use positive self-talk\n7. Try progressive muscle relaxation",
        'anxiety medication': "While I can't provide medical advice, common anxiety medications include:\n- SSRIs (Selective Serotonin Reuptake Inhibitors)\n- SNRIs (Serotonin-Norepinephrine Reuptake Inhibitors)\n- Benzodiazepines (short-term use)\n- Beta-blockers\nAlways consult with a healthcare professional about medication options.",
    },
    
    # Depression-related questions
    'depression': {
        'what is depression': "Depression is a serious mental health condition characterized by persistent sadness, loss of interest in activities, and various physical and emotional problems. It's more than just feeling down - it's a medical condition that requires treatment.",
        'depression symptoms': "Common depression symptoms include:\n- Persistent sadness\n- Loss of interest in activities\n- Changes in sleep patterns\n- Changes in appetite\n- Fatigue\n- Feelings of worthlessness\n- Difficulty concentrating\n- Physical aches and pains\n- Thoughts of death or suicide",
        'how to help depression': "Ways to help manage depression:\n1. Seek professional help\n2. Maintain a regular routine\n3. Exercise regularly\n4. Eat a balanced diet\n5. Get enough sleep\n6. Practice self-care\n7. Stay connected with others\n8. Challenge negative thoughts\n9. Set small, achievable goals\n10. Consider therapy or counseling",
        'depression treatment': "Depression treatment options include:\n- Psychotherapy (talk therapy)\n- Medication\n- Lifestyle changes\n- Support groups\n- Alternative therapies\n- Hospitalization (in severe cases)\nA combination of treatments is often most effective.",
        'depression vs sadness': "While sadness is a normal emotion that comes and goes, depression is a persistent condition that affects daily functioning. Depression lasts longer (at least 2 weeks) and includes multiple symptoms beyond just feeling sad.",
    },
    
    # Stress-related questions
    'stress': {
        'what is stress': "Stress is the body's response to any demand or challenge. While some stress is normal and can be beneficial, chronic stress can negatively impact physical and mental health.",
        'stress symptoms': "Common stress symptoms include:\n- Headaches\n- Muscle tension\n- Fatigue\n- Sleep problems\n- Anxiety\n- Irritability\n- Difficulty concentrating\n- Changes in appetite\n- Digestive issues\n- Increased heart rate",
        'how to reduce stress': "Effective stress reduction techniques:\n1. Practice deep breathing\n2. Exercise regularly\n3. Get enough sleep\n4. Maintain a healthy diet\n5. Practice time management\n6. Set realistic goals\n7. Learn to say no\n8. Stay connected with others\n9. Try relaxation techniques\n10. Seek professional help if needed",
        'work stress': "To manage work stress:\n1. Set clear boundaries\n2. Take regular breaks\n3. Prioritize tasks\n4. Delegate when possible\n5. Practice good time management\n6. Maintain work-life balance\n7. Communicate with supervisors\n8. Take care of your physical health\n9. Develop coping strategies\n10. Consider professional support",
        'stress management': "Effective stress management includes:\n- Identifying stress triggers\n- Developing healthy coping mechanisms\n- Maintaining a support network\n- Practicing self-care\n- Setting realistic expectations\n- Learning relaxation techniques\n- Getting regular exercise\n- Maintaining a healthy lifestyle",
    },
    
    # Sleep-related questions
    'sleep': {
        'sleep problems': "Common sleep problems include:\n- Insomnia\n- Sleep apnea\n- Restless legs syndrome\n- Nightmares\n- Sleepwalking\n- Excessive daytime sleepiness\n- Irregular sleep patterns",
        'how to sleep better': "Tips for better sleep:\n1. Maintain a regular sleep schedule\n2. Create a bedtime routine\n3. Make your bedroom sleep-friendly\n4. Limit screen time before bed\n5. Avoid caffeine and alcohol\n6. Exercise regularly\n7. Manage stress\n8. Avoid large meals before bed\n9. Try relaxation techniques\n10. Consider sleep hygiene practices",
        'insomnia': "To manage insomnia:\n1. Stick to a sleep schedule\n2. Create a relaxing bedtime routine\n3. Make your bedroom comfortable\n4. Limit daytime naps\n5. Exercise regularly\n6. Manage stress\n7. Avoid stimulants\n8. Try relaxation techniques\n9. Consider cognitive behavioral therapy\n10. Consult a healthcare provider",
        'sleep hygiene': "Good sleep hygiene practices:\n- Keep a consistent sleep schedule\n- Create a comfortable sleep environment\n- Limit exposure to screens before bed\n- Avoid caffeine and alcohol\n- Exercise regularly\n- Manage stress\n- Create a bedtime routine\n- Use your bed only for sleep\n- Avoid large meals before bed\n- Get exposure to natural light during the day",
    },
    
    # Self-care questions
    'self care': {
        'what is self care': "Self-care is the practice of taking action to preserve or improve one's own health, well-being, and happiness. It involves activities and practices that we engage in regularly to reduce stress and maintain and enhance our well-being.",
        'self care activities': "Self-care activities include:\n- Physical activities (exercise, healthy eating)\n- Emotional activities (journaling, therapy)\n- Social activities (connecting with friends)\n- Spiritual activities (meditation, prayer)\n- Professional activities (setting boundaries)\n- Mental activities (reading, learning)\n- Environmental activities (organizing space)\n- Financial activities (budgeting)\n- Recreational activities (hobbies)\n- Personal activities (grooming, hygiene)",
        'self care routine': "A good self-care routine includes:\n1. Regular exercise\n2. Healthy eating\n3. Adequate sleep\n4. Stress management\n5. Social connection\n6. Personal hygiene\n7. Time for hobbies\n8. Setting boundaries\n9. Practicing gratitude\n10. Regular check-ins with yourself",
        'self care for mental health': "Self-care practices for mental health:\n- Practice mindfulness\n- Maintain social connections\n- Set healthy boundaries\n- Engage in regular exercise\n- Get enough sleep\n- Eat a balanced diet\n- Practice relaxation techniques\n- Seek professional help when needed\n- Engage in enjoyable activities\n- Practice self-compassion",
    },
    
    # Therapy-related questions
    'therapy': {
        'what is therapy': "Therapy is a treatment process where a trained professional helps individuals understand and work through their problems, develop coping strategies, and improve their mental health and well-being.",
        'types of therapy': "Common types of therapy include:\n- Cognitive Behavioral Therapy (CBT)\n- Psychodynamic Therapy\n- Humanistic Therapy\n- Dialectical Behavior Therapy (DBT)\n- Family Therapy\n- Group Therapy\n- Art Therapy\n- Music Therapy\n- Play Therapy\n- Online Therapy",
        'how to find therapist': "To find a therapist:\n1. Check with your insurance provider\n2. Ask for recommendations\n3. Use online directories\n4. Consider your specific needs\n5. Research credentials\n6. Schedule consultations\n7. Consider location and availability\n8. Check reviews and testimonials\n9. Consider cultural competence\n10. Trust your instincts",
        'therapy benefits': "Benefits of therapy include:\n- Improved mental health\n- Better coping skills\n- Increased self-awareness\n- Improved relationships\n- Better stress management\n- Enhanced problem-solving skills\n- Increased self-esteem\n- Better emotional regulation\n- Improved communication skills\n- Personal growth",
    },
    
    # Crisis-related questions
    'crisis': {
        'suicide prevention': "If you or someone you know is in crisis:\n1. Call emergency services\n2. Contact a crisis hotline\n3. Stay with the person\n4. Remove access to harmful means\n5. Listen without judgment\n6. Express concern and support\n7. Encourage professional help\n8. Create a safety plan\n9. Follow up regularly\n10. Take care of yourself too",
        'mental health emergency': "In a mental health emergency:\n1. Call emergency services\n2. Contact a crisis hotline\n3. Go to the nearest emergency room\n4. Stay with the person\n5. Remove access to harmful means\n6. Listen and provide support\n7. Encourage professional help\n8. Create a safety plan\n9. Follow up regularly\n10. Take care of yourself",
        'panic attack': "During a panic attack:\n1. Find a quiet place\n2. Practice deep breathing\n3. Use grounding techniques\n4. Focus on the present\n5. Use positive self-talk\n6. Try progressive muscle relaxation\n7. Remind yourself it will pass\n8. Seek support if needed\n9. Consider professional help\n10. Develop a prevention plan",
    },
    
    # General mental health questions
    'general': {
        'mental health': "Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel, and act. It also helps determine how we handle stress, relate to others, and make choices.",
        'mental health tips': "General mental health tips:\n1. Stay connected with others\n2. Get regular exercise\n3. Eat a balanced diet\n4. Get enough sleep\n5. Practice stress management\n6. Set realistic goals\n7. Take breaks when needed\n8. Practice self-care\n9. Seek help when needed\n10. Be kind to yourself",
        'mental health resources': "Mental health resources include:\n- Mental health professionals\n- Support groups\n- Crisis hotlines\n- Online resources\n- Community programs\n- Educational materials\n- Self-help books\n- Mobile apps\n- Workplace programs\n- School counseling services",
        'mental health stigma': "To combat mental health stigma:\n1. Educate yourself and others\n2. Be open about mental health\n3. Use respectful language\n4. Show compassion\n5. Share your experiences\n6. Challenge stereotypes\n7. Support mental health initiatives\n8. Advocate for change\n9. Be an ally\n10. Practice self-acceptance",
    },
    
    # Relationships and social support
    'relationships': {
        'healthy relationships': "Characteristics of healthy relationships include:\n- Mutual respect\n- Trust and honesty\n- Good communication\n- Individuality\n- Equality\n- Support\n- Healthy boundaries\n- Compromise\n- Shared values\n- Fun and enjoyment",
        'toxic relationships': "Signs of a toxic relationship:\n- Lack of trust\n- Disrespect\n- Poor communication\n- Controlling behavior\n- Emotional abuse\n- Physical abuse\n- Isolation\n- Manipulation\n- Constant criticism\n- Lack of support",
        'social support': "Building social support:\n1. Reach out to friends and family\n2. Join support groups\n3. Volunteer in your community\n4. Take classes or join clubs\n5. Use social media mindfully\n6. Attend community events\n7. Practice active listening\n8. Be open to new friendships\n9. Maintain existing relationships\n10. Seek professional help if needed",
        'loneliness': "Coping with loneliness:\n1. Acknowledge your feelings\n2. Reach out to others\n3. Join social activities\n4. Practice self-care\n5. Volunteer\n6. Get a pet\n7. Try new hobbies\n8. Use technology to connect\n9. Consider therapy\n10. Be patient with yourself",
    },
    
    # Mindfulness and meditation
    'mindfulness': {
        'what is mindfulness': "Mindfulness is the practice of being fully present and engaged in the current moment, without judgment. It involves paying attention to your thoughts, feelings, and sensations with curiosity and acceptance.",
        'mindfulness benefits': "Benefits of mindfulness include:\n- Reduced stress\n- Improved focus\n- Better emotional regulation\n- Enhanced self-awareness\n- Improved relationships\n- Better sleep\n- Reduced anxiety\n- Increased resilience\n- Better decision-making\n- Greater life satisfaction",
        'mindfulness exercises': "Simple mindfulness exercises:\n1. Body scan meditation\n2. Breathing exercises\n3. Mindful walking\n4. Mindful eating\n5. Five senses exercise\n6. Loving-kindness meditation\n7. Mindful listening\n8. Gratitude practice\n9. Mindful stretching\n10. Mindful journaling",
        'meditation techniques': "Different meditation techniques:\n- Focused attention\n- Body scan\n- Loving-kindness\n- Walking meditation\n- Mantra meditation\n- Visualization\n- Breath awareness\n- Progressive relaxation\n- Mindful movement\n- Open monitoring",
    },
    
    # Grief and loss
    'grief': {
        'stages of grief': "The stages of grief (not always linear):\n1. Denial\n2. Anger\n3. Bargaining\n4. Depression\n5. Acceptance\nRemember that everyone experiences grief differently and there's no 'right' way to grieve.",
        'coping with loss': "Ways to cope with loss:\n1. Allow yourself to grieve\n2. Seek support\n3. Take care of yourself\n4. Express your feelings\n5. Create rituals\n6. Be patient with yourself\n7. Consider therapy\n8. Join a support group\n9. Find meaning\n10. Honor memories",
        'grief support': "Getting support for grief:\n1. Talk to friends and family\n2. Join a grief support group\n3. Consider grief counseling\n4. Read about grief\n5. Write in a journal\n6. Create memorials\n7. Practice self-care\n8. Be patient with yourself\n9. Seek professional help\n10. Connect with others who understand",
    },
    
    # Trauma and PTSD
    'trauma': {
        'what is trauma': "Trauma is an emotional response to a terrible event like an accident, abuse, or natural disaster. It can cause lasting effects on mental, physical, and emotional well-being.",
        'trauma symptoms': "Common trauma symptoms include:\n- Flashbacks\n- Nightmares\n- Anxiety\n- Depression\n- Hypervigilance\n- Avoidance\n- Emotional numbness\n- Difficulty trusting\n- Self-destructive behavior\n- Physical symptoms",
        'trauma recovery': "Steps in trauma recovery:\n1. Acknowledge the trauma\n2. Seek professional help\n3. Practice self-care\n4. Build support network\n5. Learn coping skills\n6. Process emotions\n7. Set boundaries\n8. Practice grounding\n9. Be patient with yourself\n10. Celebrate progress",
        'PTSD': "Post-Traumatic Stress Disorder (PTSD) is a mental health condition triggered by experiencing or witnessing a traumatic event. Symptoms may include flashbacks, nightmares, severe anxiety, and uncontrollable thoughts about the event.",
    },
    
    # Addiction and recovery
    'addiction': {
        'what is addiction': "Addiction is a complex condition where a person compulsively uses a substance or engages in a behavior despite harmful consequences. It affects the brain's reward system and can lead to physical and psychological dependence.",
        'addiction signs': "Signs of addiction include:\n- Loss of control\n- Continued use despite problems\n- Neglecting responsibilities\n- Relationship problems\n- Tolerance development\n- Withdrawal symptoms\n- Failed attempts to quit\n- Time spent obtaining/using\n- Giving up activities\n- Physical health problems",
        'recovery steps': "Steps in addiction recovery:\n1. Acknowledge the problem\n2. Seek professional help\n3. Join support groups\n4. Develop coping skills\n5. Build healthy habits\n6. Repair relationships\n7. Create a support network\n8. Set goals\n9. Practice self-care\n10. Stay committed",
        'supporting recovery': "How to support someone in recovery:\n1. Educate yourself\n2. Be patient\n3. Set boundaries\n4. Offer encouragement\n5. Avoid enabling\n6. Practice self-care\n7. Be available\n8. Celebrate progress\n9. Be honest\n10. Seek support for yourself",
    },
    
    # Eating disorders
    'eating disorders': {
        'types of eating disorders': "Common eating disorders include:\n- Anorexia Nervosa\n- Bulimia Nervosa\n- Binge Eating Disorder\n- Avoidant/Restrictive Food Intake Disorder\n- Other Specified Feeding or Eating Disorder\nEach has unique symptoms and requires specialized treatment.",
        'eating disorder signs': "Warning signs of eating disorders:\n- Extreme weight changes\n- Preoccupation with food\n- Distorted body image\n- Secretive eating\n- Excessive exercise\n- Social withdrawal\n- Physical symptoms\n- Mood changes\n- Food rituals\n- Denial of problem",
        'eating disorder help': "Getting help for eating disorders:\n1. Seek professional help\n2. Consider treatment programs\n3. Build support network\n4. Learn about nutrition\n5. Address underlying issues\n6. Practice self-compassion\n7. Set realistic goals\n8. Develop healthy habits\n9. Be patient with recovery\n10. Stay committed to treatment",
    },
}
GENERAL_SUPPORT = [
    "I'm here to listen and support you. Would you like to talk about what's been on your mind? Sometimes sharing our thoughts can help us feel less alone.",
    "I hear you're going through a difficult time. Let's explore some coping strategies together. What's been helping you get through this?",
    "Your feelings are completely valid. Would you like to try some relaxation techniques or talk about what's been weighing on you?",
    "I'm here to support you. Let's work together to find strategies that can help you feel better. What would be most helpful to focus on right now?",
    "I understand this is challenging for you. Let's break this down into smaller, more manageable steps. What's one small thing you can do for yourself today?"
]

CONVERSATIONAL_PROMPTS = {
    'greetings': {
        'hi': [
            "Hello! How are you feeling today?",
            "Hi there! I'm here to listen and chat. How's your day going?",
            "Hey! It's nice to meet you. How are you doing?",
            "Hello! I'm glad you're here. How can I support you today?",
            "Hi! I'm here to chat. How are you feeling?"
        ],
        'hello': [
            "Hello! How are you doing today?",
            "Hi! I'm here to listen. How can I help you?",
            "Hey there! How's your day going?",
            "Hello! It's nice to meet you. How are you feeling?",
            "Hi! I'm glad you're here. What's on your mind?"
        ],
        'hey': [
            "Hey! How are you doing today?",
            "Hi there! How can I help you?",
            "Hey! I'm here to chat. How are you feeling?",
            "Hello! How's your day going?",
            "Hey! What's on your mind?"
        ]
    },
    'how_are_you': {
        'how are you': [
            "I'm doing well, thank you for asking! How about you?",
            "I'm here and ready to chat! How are you feeling?",
            "I'm doing good, thanks! How's your day going?",
            "I'm well, thank you! How are you doing today?",
            "I'm here to support you! How are you feeling?"
        ],
        'how are you doing': [
            "I'm doing great, thanks! How about you?",
            "I'm here and ready to listen! How are you?",
            "I'm doing well! How's your day going?",
            "I'm good, thank you! How are you feeling?",
            "I'm here to chat! How are you doing?"
        ]
    },
    'farewell': {
        'bye': [
            "Goodbye! Take care of yourself!",
            "Bye! Remember I'm here if you need to talk again.",
            "Goodbye! Wishing you a great day!",
            "Bye! Take care and be kind to yourself!",
            "Goodbye! Feel free to come back anytime!"
        ],
        'goodbye': [
            "Goodbye! Take care and be well!",
            "Goodbye! Remember to practice self-care!",
            "Goodbye! Wishing you peace and happiness!",
            "Goodbye! You're always welcome to chat again!",
            "Goodbye! Take care of your mental health!"
        ],
        'see you': [
            "See you later! Take care!",
            "See you! Remember I'm here for you!",
            "See you! Have a wonderful day!",
            "See you! Be kind to yourself!",
            "See you! Feel free to come back anytime!"
        ]
    },
    'thanks': {
        'thank you': [
            "You're welcome! I'm here to help.",
            "You're welcome! Feel free to ask anything.",
            "You're welcome! I'm glad I could help.",
            "You're welcome! Take care of yourself.",
            "You're welcome! Remember I'm here for you."
        ],
        'thanks': [
            "You're welcome! How else can I help?",
            "You're welcome! Is there anything else you'd like to talk about?",
            "You're welcome! I'm here to support you.",
            "You're welcome! Take care and be well.",
            "You're welcome! Feel free to chat anytime."
        ]
    },
    'small_talk': {
        'whats up': [
            "Not much, just here to chat! How about you?",
            "I'm here to listen and support you! What's going on with you?",
            "Just here to help! How's your day going?",
            "I'm here to chat! What's on your mind?",
            "Ready to talk! How are you doing?"
        ],
        'how is it going': [
            "It's going well! How about you?",
            "I'm here and ready to chat! How are you?",
            "All good here! How's your day?",
            "I'm doing well! How are you feeling?",
            "Everything's good! How can I help you?"
        ]
    }
}

def generate_mental_health_response(user_input):
    # Convert input to lowercase for case-insensitive matching
    input_lower = user_input.lower()
    
    # Check for crisis keywords first
    if any(keyword in input_lower for keyword in ['suicide', 'kill myself', 'end it all', 'want to die']):
        return random.choice(MENTAL_HEALTH_KEYWORDS['crisis'])
    
    # Check for other mental health keywords
    for keyword, responses in MENTAL_HEALTH_KEYWORDS.items():
        if keyword in input_lower:
            return random.choice(responses)
    
    # If no specific keyword matches, return a general supportive response
    return random.choice(GENERAL_SUPPORT)

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return render_template('login.html')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if current_user.is_authenticated:
            return redirect(url_for('home'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Please enter both username and password', 'error')
                return redirect(url_for('login'))
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user)
                next_page = request.args.get('next')
                logger.info(f"User {username} logged in successfully")
                return redirect(next_page or url_for('home'))
            else:
                logger.warning(f"Failed login attempt for username: {username}")
                flash('Invalid username or password', 'error')
        
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login route: {str(e)}")
        logger.error(traceback.format_exc())
        flash('An error occurred during login. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/analyze/text', methods=['POST'])
@login_required
def analyze_text():
    try:
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            logger.error("No text provided in request")
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text.strip():
            logger.error("Empty text provided")
            return jsonify({'error': 'Empty text provided'}), 400
        
        logger.info(f"Analyzing text: {text[:100]}...")  # Log first 100 chars
        
        # Perform text analysis
        try:
            analysis = text_analyzer.analyze(text)
            logger.info("Text analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Text analysis error: {str(e)}'}), 500
        
        # Get recommendations
        try:
            recommendations = recommendation_engine.get_recommendations(analysis)
            logger.info("Recommendations generated successfully")
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            recommendations = []  # Return empty recommendations if there's an error
        
        return jsonify({
            'analysis': analysis,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in text analysis endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

@app.route('/analyze/audio', methods=['POST'])
@login_required
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Create temporary file for audio processing
    temp_webm_path = None
    temp_wav_path = None
    
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
            temp_webm_path = temp_webm.name
            audio_file.save(temp_webm_path)
            logger.info(f"Saved audio to temporary file: {temp_webm_path}")
        
        # Convert WebM to WAV for analysis
        temp_wav_path = temp_webm_path + '.wav'
        convert_webm_to_wav(temp_webm_path, temp_wav_path)
        
        # Perform voice modulation and emotion analysis
        voice_analysis = emotion_detector.detect_emotion(temp_wav_path)
        
        # Perform text analysis on transcribed audio (separate from voice analysis)
        text_analysis = None
        if whisper_model is not None:
            result = whisper_model.transcribe(temp_wav_path)
            text = result['text']
            text_analysis = {
                'transcription': text,
                'text_analysis': text_analyzer.analyze(text)
            }
        
        # Combine results
        analysis = {
            'voice_analysis': voice_analysis,
            'text_analysis': text_analysis
        }
        
        recommendations = recommendation_engine.get_recommendations(analysis)
        
        return jsonify({
            'analysis': analysis,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Audio analysis error: {str(e)}'}), 500
    finally:
        # Clean up temporary files
        for path in [temp_webm_path, temp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.route('/analyze/visual', methods=['POST'])
@login_required
def analyze_visual():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Analyze the image using DeepFace
        analysis = visual_analyzer.analyze(image_data)
        
        return jsonify({
            'analysis': analysis,
            'recommendations': recommendation_engine.get_recommendations(analysis)
        })
    except Exception as e:
        logger.error(f"Error in visual analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    if whisper_model is None:
        return jsonify({'error': 'Speech recognition model not loaded'}), 500
    
    audio_file = request.files['audio']
    logger.info(f"Received audio file: {audio_file.filename}, size: {audio_file.content_length} bytes")
    
    # Create temporary files
    temp_webm_path = None
    temp_wav_path = None
    
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
            temp_webm_path = temp_webm.name
            audio_file.save(temp_webm_path)
            logger.info(f"Saved audio to temporary file: {temp_webm_path}")
        
        # Check if the file exists and has content
        if not os.path.exists(temp_webm_path) or os.path.getsize(temp_webm_path) == 0:
            logger.error(f"Temporary file is missing or empty: {temp_webm_path}")
            return jsonify({'error': 'Invalid audio file'}), 400
        
        # Try to convert WebM to WAV
        try:
            temp_wav_path = temp_webm_path + '.wav'
            convert_webm_to_wav(temp_webm_path, temp_wav_path)
            
            # Transcribe the audio using Whisper
            logger.info("Starting transcription with Whisper")
            result = whisper_model.transcribe(temp_wav_path)
            logger.info("Transcription completed successfully")
            
            return jsonify({'text': result['text']})
        except Exception as e:
            logger.warning(f"Error converting audio: {str(e)}")
            logger.warning("Attempting direct transcription with Whisper")
            
            # Try to transcribe the WebM file directly with Whisper
            try:
                logger.info("Attempting to transcribe WebM file directly with Whisper")
                result = whisper_model.transcribe(temp_webm_path)
                logger.info("Direct transcription completed successfully")
                return jsonify({'text': result['text']})
            except Exception as direct_error:
                logger.error(f"Direct transcription failed: {str(direct_error)}")
                
                # Provide detailed error information
                error_details = {
                    'error': 'Failed to process audio file. Please check your FFmpeg installation.',
                    'ffmpeg_error': str(e),
                    'whisper_error': str(direct_error),
                    'installation_guide': {
                        'windows': 'Download from https://ffmpeg.org/download.html and add to PATH',
                        'macos': 'Run: brew install ffmpeg',
                        'linux': 'Run: sudo apt-get install ffmpeg'
                    },
                    'troubleshooting': [
                        'Make sure FFmpeg is installed and accessible from the command line',
                        'Try running "ffmpeg -version" in a command prompt to verify installation',
                        'Check if the PATH environment variable includes the FFmpeg directory',
                        'Restart the application after installing FFmpeg'
                    ]
                }
                
                return jsonify(error_details), 500
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Transcription error: {str(e)}'}), 500
    finally:
        # Clean up temporary files
        for path in [temp_webm_path, temp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.route('/analyze/questionnaire', methods=['POST'])
@login_required
def analyze_questionnaire():
    try:
        responses = request.get_json()
        if not responses:
            return jsonify({'error': 'No responses provided'}), 400

        # Convert question responses to category responses
        category_responses = {
            'energy_level': responses.get('question_1', ''),
            'thought_patterns': responses.get('question_2', ''),
            'sleep_quality': responses.get('question_3', ''),
            'social_connection': responses.get('question_4', ''),
            'self_relationship': responses.get('question_5', ''),
            'motivation': responses.get('question_6', ''),
            'stress_management': responses.get('question_7', ''),
            'purpose': responses.get('question_8', ''),
            'self_care': responses.get('question_9', ''),
            'life_satisfaction': responses.get('question_10', '')
        }

        # Analyze questionnaire responses
        analysis = text_analyzer.analyze_questionnaire(category_responses)
        
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in questionnaire analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate mental health focused response
        response = generate_mental_health_response(user_message)
        
        return jsonify({
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your message'}), 500

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True) 