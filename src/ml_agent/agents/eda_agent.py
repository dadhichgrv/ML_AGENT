import os 
import pandas as pd
from langchain_openai import ChatOpenAI 
from pathlib import Path
from src.ml_agent.schemas.eda_spec import EDAReport 
from src.ml_agent.agents.eda_tools import analyse_dataset
from src.ml_agent.schemas.problem_spec import ProblemSpec
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a Senior Data Scientist responsible for performing Exploratory Data Analysis.

You will recieve :
A. A validated problem specification from Agent
B. Dataset stats calculated by python tools

Rules:
1. Base your findings only on given data and statistics
2. Check if proposed target column exists
3. Identify data quality issues
4. Distinguish evidence from hypothesis
5. Recommend possible feature engineering actions
6. Recommend metrics based on business objective and class balance information
7. Ask concise clarification questions when ambiguity could change ML formulation 
8. Do not invent dataset statistics 
9. Return output matching EDDAReport schema
"""


def create_eda_agent(problem_spec: ProblemSpec, df:pd.DataFrame):
    llm = ChatOpenAI(
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
    
    structured_llm = llm.with_structured_output(EDAReport)

    findings = analyse_dataset(df=df,
                               target_column = problem_spec.target_column)

    prompt = f"""
               System_Prompt : {SYSTEM_PROMPT}
               Problem_Specifications: {problem_spec.model_dump_json(indent=2)}
               Actual_data_findings_from_python : {findings}

               Interpret these findings and produce EDA report

              """

    response = structured_llm.invoke(prompt)

    return response 
