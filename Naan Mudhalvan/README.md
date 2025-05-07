# Mental Health Analysis System

A comprehensive system for analyzing mental health through text, audio, and visual inputs. The system provides emotional analysis, mental health scoring, and personalized recommendations.

## Features

- Text Analysis: Analyze written content for emotional patterns and mental health indicators
- Audio Analysis: Process speech for tone, emotion, and mental health indicators
- Visual Analysis: Analyze facial expressions and visual cues for emotional state
- Personalized Recommendations: Generate tailored mental health recommendations based on analysis results

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Web browser with JavaScript enabled

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-analysis-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Enter text for analysis
   - Upload audio files for analysis
   - Upload images for analysis

4. View the analysis results and recommendations

## Input Requirements

- Text: Any written content in English
- Audio: WAV, MP3, or other common audio formats
- Images: JPG, PNG, or other common image formats with clear facial features

## System Components

- `app.py`: Main Flask application
- `text_analyzer.py`: Text analysis module
- `audio_analyzer.py`: Audio analysis module
- `visual_analyzer.py`: Visual analysis module
- `recommendation_engine.py`: Recommendation generation module
- `templates/index.html`: Web interface

## Security and Privacy

- All analysis is performed locally
- No data is stored permanently
- Uploaded files are processed in memory and deleted immediately after analysis

## Limitations

- Text analysis is optimized for English language
- Audio analysis requires clear speech
- Visual analysis works best with front-facing facial images
- The system is not a replacement for professional mental health care

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for informational purposes only and is not a substitute for professional mental health advice, diagnosis, or treatment. Always seek the advice of your mental health professional or other qualified health provider with any questions you may have regarding a medical condition. 