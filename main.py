import pandas as pd
from pathlib import Path
import os
from src.ml_agent.agents.problem_agent import create_problem_agent
from src.ml_agent.agents.eda_agent import create_eda_agent
from src.ml_agent.agents.feature_agent import create_feature_agent
from src.ml_agent.agents.feature_eng_tools import engineer_features
from src.ml_agent.workflow import workflow
from langchain_openai import ChatOpenAI 
#from src.ml_agent.schemas.feature_spec import featurespec

llm = ChatOpenAI(
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

problem_statement = """
We want to predict food delivery time for each customer
"""

DATAPATH = Path(__file__).resolve().parent / "data" / "swiggy_input.csv"
df = pd.read_csv(DATAPATH )

result = workflow.invoke(
    {
        "df":df,
        "llm":llm,
        "problem_statement":problem_statement
     })

problem_spec  = result["problemspec"]
#edareport     = result["edareport"]
edaresult     = result["edaresult"]
featurespec   = result["featurespec"]
processeddata = result["processed_data"]
trainingresults = result["trainingresults"]
testingresults = result["testingresults"]

print("\n Problem Specification \n",problem_spec)
#print("EDA Report \n",edareport)
print("\n EDA Result \n",edaresult)
print("\n Feature Spec \n",featurespec)
print("\n Processed Data \n",processeddata)
print("\n Training Results \n",trainingresults)
print("\n Testing Results \n",testingresults)


# problem_spec = create_problem_agent(business_problem=business_problem, dataset= dataset)

# eda_report = create_eda_agent(problem_spec, DATAPATH)

# feature_plan = create_feature_agent(problem_spec, eda_report, dataset.columns.tolist())

# result = engineer_features(
#         df = dataset,
#         target_column = feature_plan.target_column,
#         #numeric_features = feature_plan.numeric_features ,
#         #categorical_features = feature_plan.categorical_features,
#         featurespec = feature_plan
#                             )

# print("FEATURE_PLAN :",feature_plan)
# print("\n")
# print("RESULT",result)
