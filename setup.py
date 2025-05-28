from setuptools import setup, find_packages

setup(
    name="heart-disease-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.31.1",
        "pandas==2.2.0",
        "numpy==1.26.3",
        "scikit-learn==1.4.0",
        "joblib==1.3.2",
        "requests==2.31.0",
        "plotly==5.18.0",
        "python-dotenv==1.0.0",
        "openai==1.12.0"
    ],
    python_requires=">=3.8",
) 