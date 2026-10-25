import os 
from pathlib import Path
from langchain_openai import ChatOpenAI 
from src.ml_agent.schemas.eda_spec import EDAReport
from src.ml_agent.schemas.problem_spec import ProblemSpec
from src.ml_agent.schemas.feature_spec import featurespec
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are an experienced Senior Machine Learning Engineer who is responsible for doing feature engineering on given dataset.

INPUTS:
- ProblemSpec from Agent1
- EDAReport from Agent2

RESPONSIBILITIES:
- Identify numeric and categorical features
- Exclude target feature from feature list
- Recommnend appropriate missing value imputation 
- Recommend categorical encoding
- Recommend numeric data scaling only when needed
- Flag identifiers, potential leakage
- Explain important decisions


RULES:
- Use only columns which are part of dataset
- Do not invent new columns or statistics
- Do not blindly exclude identifiers, explain reason for each exclusion
- Do not make unsupported claims abut feature importance
- Do not use target column as input feature
- Do not train model
- If critical information is missing, return status="needs_review"
- Return output matching the featurespec schema

"""

def create_feature_agent(problem_spec, eda_report, columns:list[str]):
    llm = ChatOpenAI(
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
    
    structured_llm = llm.with_structured_output(featurespec)

    prompt = f"""
             SystemPrompt : {SYSTEM_PROMPT}
             Problem_Specifications: {problem_spec}
             EDAReport : {eda_report}
             Available_Columns : {columns}

            Produce proposed feature engineering plan

              """
    response = structured_llm.invoke(prompt)

    return response