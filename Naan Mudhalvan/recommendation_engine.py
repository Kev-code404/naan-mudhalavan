class RecommendationEngine:
    def __init__(self):
        self.recommendation_categories = {
            'immediate_actions': {
                'high_priority': [
                    "Consider scheduling an appointment with a mental health professional",
                    "Reach out to a trusted friend or family member for support",
                    "Contact a mental health crisis hotline if you're having thoughts of self-harm"
                ],
                'medium_priority': [
                    "Practice deep breathing exercises",
                    "Take a short walk or engage in light physical activity",
                    "Write down your thoughts and feelings in a journal"
                ],
                'low_priority': [
                    "Try a mindfulness meditation session",
                    "Listen to calming music",
                    "Take a break from screens and social media"
                ]
            },
            'lifestyle_changes': {
                'high_priority': [
                    "Establish a regular sleep schedule",
                    "Incorporate regular exercise into your routine",
                    "Maintain a balanced diet with proper nutrition"
                ],
                'medium_priority': [
                    "Set aside time for hobbies and activities you enjoy",
                    "Practice good sleep hygiene",
                    "Stay hydrated throughout the day"
                ],
                'low_priority': [
                    "Try a new hobby or creative activity",
                    "Spend more time in nature",
                    "Practice gratitude by keeping a daily gratitude journal"
                ]
            },
            'professional_help': {
                'high_priority': [
                    "Schedule an appointment with a therapist or counselor",
                    "Consider joining a support group",
                    "Consult with a psychiatrist about medication options"
                ],
                'medium_priority': [
                    "Research local mental health resources",
                    "Look into online therapy options",
                    "Consider talking to your primary care physician"
                ],
                'low_priority': [
                    "Explore self-help books and resources",
                    "Try meditation or mindfulness apps",
                    "Join online mental health communities"
                ]
            }
        }

    def get_recommendations(self, analysis):
        # Extract mental health score from analysis
        mental_health_score = analysis.get('mental_health_score', 50)
        
        # Determine priority level based on mental health score
        priority = self._determine_priority(mental_health_score)
        
        # Get recommendations for each category
        recommendations = {
            'immediate_actions': self._get_category_recommendations('immediate_actions', priority),
            'lifestyle_changes': self._get_category_recommendations('lifestyle_changes', priority),
            'professional_help': self._get_category_recommendations('professional_help', priority)
        }
        
        # Add personalized recommendations based on specific indicators
        personalized_recommendations = self._get_personalized_recommendations(analysis)
        recommendations['personalized'] = personalized_recommendations
        
        return recommendations

    def _determine_priority(self, score):
        if score < 30:
            return 'high_priority'
        elif score < 60:
            return 'medium_priority'
        else:
            return 'low_priority'

    def _get_category_recommendations(self, category, priority):
        return self.recommendation_categories[category][priority]

    def _get_personalized_recommendations(self, analysis):
        personalized = []
        
        # Check for specific indicators in the analysis
        if 'mental_health_indicators' in analysis:
            indicators = analysis['mental_health_indicators']
            
            # Add recommendations based on facial expression
            if 'facial_expression' in indicators:
                if indicators['facial_expression'] == 'negative':
                    personalized.append("Practice positive self-talk and affirmations")
                    personalized.append("Try smiling more, even if you don't feel like it")
            
            # Add recommendations based on eye contact
            if 'eye_contact' in indicators:
                if indicators['eye_contact'] == 'poor':
                    personalized.append("Practice maintaining eye contact in conversations")
                    personalized.append("Consider joining a social skills group")
        
        # Add recommendations based on emotions
        if 'emotions' in analysis:
            emotions = analysis['emotions']
            if 'anxiety' in emotions and emotions['anxiety'] > 0.5:
                personalized.append("Try progressive muscle relaxation exercises")
                personalized.append("Practice grounding techniques when feeling anxious")
            if 'depression' in emotions and emotions['depression'] > 0.5:
                personalized.append("Set small, achievable daily goals")
                personalized.append("Try to maintain a regular daily routine")
        
        return personalized

    def format_recommendations(self, recommendations):
        formatted = {
            'summary': "Based on your analysis, here are some recommendations to support your mental health:",
            'categories': {}
        }
        
        for category, items in recommendations.items():
            if items:  # Only include categories with recommendations
                formatted['categories'][category] = {
                    'title': category.replace('_', ' ').title(),
                    'items': items
                }
        
        return formatted 