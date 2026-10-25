import os 
from pathlib import Path
from langchain_openai import ChatOpenAI 
from src.ml_agent.schemas.eda_spec import EDAReport
from src.ml_agent.schemas.problem_spec import ProblemSpec
from src.ml_agent.schemas.feature_spec import featurespec
from src.ml_agent.schemas.model_training_spec import modeltrainingspec
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are an experienced Senior Machine Learning Engineer who has epertise in training Machine Learning Models.

INPUTS:
Training Features
Testing Features
Target from training and testing data
Preprocessor Pipeline

RESPONSIBILITIES:
- Train model on training data only
- Test data to be used for testing
- Use multiple algorithms so as to get baseline model
- Perform Hyperparameter tuning on baseline model to improve metrics
- Use approprioate metrics to score test data

RULES:
- Do not use test data for training the model

"""


def model_training_agent(featurespec):
    llm = ChatOpenAI(
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
    
    structured_llm = llm.with_structured_output(modeltrainingspec)

    prompt = f"""
             SystemPrompt : {SYSTEM_PROMPT}
             FeatureEngineeringResult: {featurespec}

            Produce proposed model training plan

              """
    response = structured_llm.invoke(prompt)

    return response