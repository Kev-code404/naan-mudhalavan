from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model and tokenizer
try:
    logger.info("Loading DialoGPT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Comprehensive mental health Q&A database
MENTAL_HEALTH_QA = {
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
        return random.choice(MENTAL_HEALTH_QA['crisis']['suicide prevention'])
    
    # Check for conversational prompts
    for category, prompts in CONVERSATIONAL_PROMPTS.items():
        for key, responses in prompts.items():
            if key in input_lower:
                return random.choice(responses)
    
    # Check for specific questions in the Q&A database
    for category, questions in MENTAL_HEALTH_QA.items():
        for question, answer in questions.items():
            if question in input_lower:
                return answer
    
    # Check for category keywords
    for category, questions in MENTAL_HEALTH_QA.items():
        if category in input_lower:
            return random.choice(list(questions.values()))
    
    # If no specific match, return a general supportive response
    return random.choice(GENERAL_SUPPORT)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True) 