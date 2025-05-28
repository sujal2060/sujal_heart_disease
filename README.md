# Heart Disease Prediction System

A Streamlit application that predicts the probability of heart disease based on various health parameters and includes an AI-powered chatbot for heart health information.

## Features

- Heart disease prediction using machine learning
- Interactive user interface
- AI-powered chatbot for heart health information
- Real-time predictions
- Feature importance visualization

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file with:

```
TOGETHER_API_KEY=your_api_key_here
```

4. Run the application:

```bash
streamlit run heart_disease_app.py
```

## Project Structure

- `heart_disease_app.py`: Main Streamlit application
- `chatbot.py`: AI chatbot implementation
- `heart.csv`: Dataset for heart disease prediction
- `requirements.txt`: Python dependencies
- `packages.txt`: System dependencies
- `setup.py`: Package configuration

## Dependencies

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Python-dotenv
- Requests

## License

MIT License
