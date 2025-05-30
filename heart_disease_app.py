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
st.title("Heart Disease Prediction System")
st.write("""
This application predicts the probability of heart disease based on various health parameters.
Please fill in your details below to get a prediction.
""")

# Create three columns with custom widths
col1, col2, col3 = st.columns([1, 2, 1])

if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = True

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
        
        # Convert categorical variables
        data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
        data['ChestPainType'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
        data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
        data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
        data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
        
        # Store column names for later use - using only available columns
        feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                         'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                         'Oldpeak', 'ST_Slope']
        
        X = data[feature_columns]
        y = data['HeartDisease']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, feature_columns

    # Create input fields
    def get_user_input():
        st.sidebar.header("Input Parameters")
        
        # Age
        st.sidebar.markdown("### Age")
        st.sidebar.markdown("""
        - Average: 54 years
        - Healthy Range: 20-65 years
        - Risky: >65 years
        """)
        age = st.sidebar.slider('Age', 20, 100, 50)
        
        # Sex
        st.sidebar.markdown("### Sex")
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        
        # Chest Pain Type
        st.sidebar.markdown("### Chest Pain Type")
        st.sidebar.markdown("""
        - ATA: Typical Angina (chest pain related to heart)
        - NAP: Non-Anginal Pain
        - ASY: Asymptomatic
        - TA: Atypical Angina
        """)
        cp = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
        
        # Resting Blood Pressure
        st.sidebar.markdown("### Resting Blood Pressure")
        st.sidebar.markdown("""
        - Average: 132 mmHg
        - Healthy: <120 mmHg
        - Risky: >140 mmHg
        """)
        trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
        
        # Cholesterol
        st.sidebar.markdown("### Cholesterol")
        st.sidebar.markdown("""
        - Average: 246 mg/dl
        - Healthy: <200 mg/dl
        - Risky: >240 mg/dl
        """)
        chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 250)
        
        # Fasting Blood Sugar
        st.sidebar.markdown("### Fasting Blood Sugar")
        st.sidebar.markdown("""
        - Normal: <120 mg/dl
        - High: >120 mg/dl
        """)
        fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        
        # Resting ECG
        st.sidebar.markdown("### Resting ECG")
        st.sidebar.markdown("""
        - Normal: Normal ECG
        - ST: ST-T wave abnormality
        - LVH: Left ventricular hypertrophy
        """)
        restecg = st.sidebar.selectbox('Resting ECG Results', ['Normal', 'ST', 'LVH'])
        
        # Maximum Heart Rate
        st.sidebar.markdown("### Maximum Heart Rate")
        st.sidebar.markdown("""
        - Average: 149 bpm
        - Healthy: 60-100 bpm (resting)
        - Risky: <60 or >100 bpm (resting)
        """)
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 202, 150)
        
        # Exercise Angina
        st.sidebar.markdown("### Exercise Angina")
        st.sidebar.markdown("""
        - No: No chest pain during exercise
        - Yes: Chest pain during exercise
        """)
        exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        
        # ST Depression
        st.sidebar.markdown("### ST Depression")
        st.sidebar.markdown("""
        - Average: 0.8 mm
        - Healthy: <1.0 mm
        - Risky: >2.0 mm
        """)
        oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
        
        # ST Slope
        st.sidebar.markdown("### ST Slope")
        st.sidebar.markdown("""
        - Up: Normal
        - Flat: Borderline
        - Down: Abnormal
        """)
        slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ['Up', 'Flat', 'Down'])
        
        # Convert categorical variables
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
        
        # Create DataFrame with the same column order as the training data
        return pd.DataFrame([user_data], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                                'Oldpeak', 'ST_Slope'])

    # Main prediction function
    def main_prediction():
        # Load and train model
        model, scaler, feature_columns = train_model()
        
        # Get user input
        user_input = get_user_input()
        
        # Make prediction
        if st.sidebar.button('Predict'):
            st.session_state.show_chatbot = False
            # Ensure columns are in the same order as training data
            user_input = user_input[feature_columns]
            
            # Scale the input
            user_input_scaled = scaler.transform(user_input)
            
            # Make prediction
            prediction = model.predict(user_input_scaled)
            probability = model.predict_proba(user_input_scaled)
            
            # Display results
            st.subheader('Prediction Results')
            if prediction[0] == 1:
                st.error('High Risk of Heart Disease')
            else:
                st.success('Low Risk of Heart Disease')
            
            st.write(f'Probability of Heart Disease: {probability[0][1]:.2%}')
            
            # Display feature importance with user values
            st.subheader('Feature Importance and Your Values')
            
            # Create a DataFrame with feature importance and user values
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_,
                'Your Value': user_input.iloc[0].values
            })
            
            # Format the values for better readability
            feature_importance['Your Value'] = feature_importance.apply(lambda row: 
                f"{row['Your Value']:.1f}" if isinstance(row['Your Value'], (int, float)) 
                else str(row['Your Value']), axis=1)
            
            # Create two columns for the graph and the table
            graph_col, table_col = st.columns([2, 1])
            
            with graph_col:
                # Create the bar chart
                fig = px.bar(feature_importance, 
                           x='Feature', 
                           y='Importance',
                           title='Feature Importance in Prediction',
                           labels={'Importance': 'Importance Score', 'Feature': 'Health Parameters'},
                           color='Importance',
                           color_continuous_scale='RdBu')
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Health Parameters",
                    yaxis_title="Importance Score",
                    showlegend=False,
                    height=500
                )
                
                # Display the graph
                st.plotly_chart(fig, use_container_width=True)
            
            with table_col:
                # Display the table with user values
                st.markdown("### Your Input Values")
                st.dataframe(
                    feature_importance[['Feature', 'Your Value']].set_index('Feature'),
                    height=500
                )

    # Run prediction
    main_prediction()

with col2:
    # Bot icon/button to show chatbot if hidden
    if not st.session_state.show_chatbot:
        if st.button("🤖 Open Chatbot"):
            st.session_state.show_chatbot = True

    if st.session_state.show_chatbot:
        # Chatbot interface with custom styling
        st.markdown("""
        <style>
        .chat-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .chat-header {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #1f77b4;
            text-align: center;
            padding: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .chat-subheader {
            font-size: 16px;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-header">SUJAL HEART DISEASE PREDICTION SYSTEM</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-subheader">Your AI Assistant for Heart Health Information</div>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about heart disease..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                response = chatbot.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    # Additional information or features can be added here
    st.markdown("### Heart Health Tips")
    st.markdown("""
    - Maintain a healthy diet
    - Exercise regularly
    - Get enough sleep
    - Manage stress
    - Regular check-ups
    - Avoid smoking
    - Limit alcohol intake
    """) 
