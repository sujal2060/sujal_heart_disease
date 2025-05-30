import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from chatbot import HeartDiseaseChatbot
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# Initialize chatbot
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
chatbot = HeartDiseaseChatbot(TOGETHER_API_KEY)

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.write("""
This application predicts the probability of heart disease based on various health parameters.
Please fill in your details below to get a prediction.
""")

# Create three columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    # Load and preprocess data
    @st.cache_data
    def load_data():
        data = pd.read_csv('heart.csv')
        return data

    # Train model
    @st.cache_resource
    def train_model():
        data = load_data()
        data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
        data['ChestPainType'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
        data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
        data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
        data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
        
        feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                           'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                           'Oldpeak', 'ST_Slope']
        
        X = data[feature_columns]
        y = data['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, feature_columns

    def get_user_input():
        st.sidebar.header("Input Parameters")
        age = st.sidebar.slider('Age', 20, 100, 50)
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        cp = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
        trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
        chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 250)
        fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.sidebar.selectbox('Resting ECG Results', ['Normal', 'ST', 'LVH'])
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 202, 150)
        exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
        slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ['Up', 'Flat', 'Down'])

        sex = 1 if sex == 'Male' else 0
        cp = ['ATA', 'NAP', 'ASY', 'TA'].index(cp)
        fbs = 1 if fbs == 'Yes' else 0
        restecg = ['Normal', 'ST', 'LVH'].index(restecg)
        exang = 1 if exang == 'Yes' else 0
        slope = ['Up', 'Flat', 'Down'].index(slope)

        user_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': cp,
            'RestingBP': trestbps,
            'Cholesterol': chol,
            'FastingBS': fbs,
            'RestingECG': restecg,
            'MaxHR': thalach,
            'ExerciseAngina': exang,
            'Oldpeak': oldpeak,
            'ST_Slope': slope
        }

        return pd.DataFrame([user_data], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                                  'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                                  'Oldpeak', 'ST_Slope'])

    def main_prediction():
        model, scaler, feature_columns = train_model()
        user_input = get_user_input()

        if st.sidebar.button('Predict'):
            user_input = user_input[feature_columns]
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)
            probability = model.predict_proba(user_input_scaled)

            st.subheader('Prediction Results')
            if prediction[0] == 1:
                st.error('High Risk of Heart Disease')
            else:
                st.success('Low Risk of Heart Disease')

            st.write(f'Probability of Heart Disease: {probability[0][1]:.2%}')

            st.subheader('Feature Importance and Your Values')
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_,
                'Your Value': user_input.iloc[0].values
            })
            feature_importance['Your Value'] = feature_importance.apply(lambda row: 
                f"{row['Your Value']:.1f}" if isinstance(row['Your Value'], (int, float)) 
                else str(row['Your Value']), axis=1)

            graph_col, table_col = st.columns([2, 1])

            with graph_col:
                fig = px.bar(feature_importance, 
                           x='Feature', 
                           y='Importance',
                           title='Feature Importance in Prediction',
                           labels={'Importance': 'Importance Score', 'Feature': 'Health Parameters'},
                           color='Importance',
                           color_continuous_scale='RdBu')
                fig.update_layout(
                    xaxis_title="Health Parameters",
                    yaxis_title="Importance Score",
                    showlegend=False,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                st.markdown("### Your Input Values")
                st.dataframe(
                    feature_importance[['Feature', 'Your Value']].set_index('Feature'),
                    height=500
                )

    main_prediction()

with col2:
    st.markdown("## 🧠 Ask our Health Chatbot")
    st.write("Feel free to ask any health-related or heart disease-related questions.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            bot_response = chatbot.get_response(user_input)
            st.session_state.chat_history.append(("Bot", bot_response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**{sender}:** {message}")
        else:
            st.markdown(f"> 💬 **{sender}:** {message}")

