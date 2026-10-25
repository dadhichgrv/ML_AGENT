import os 
from langchain_openai import ChatOpenAI 
from src.ml_agent.schemas.problem_spec import ProblemSpec , FramingOutput
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a Senior Data Scientist responsible for translating business problems into well defined Machine Learning Problems.


Rules:
1. Do not assume every problem is classification
2. Identify the business objective, prediction unit, target, prediction horizon where possible
3. Use only the given input data to refer column names
4. Do not claim a target column exists unless it appers in given data
5. Distinguish known facts from assumptions
6. Recommend metrics based on business objective and class balance information
7. Ask concise clarification questions when ambiguity could change ML formulation 
8. Do not invent dataset statistics 
9. If target definition or prediction horizon is essential but unclear, use status = "needs_clarification"
10. Use status = "ready_for_eda" only when problem is sufficiently defined to begin exploratory data analysis
11. Return only field defined by requested output schemas
"""


def create_problem_agent(business_problem: str, dataset: dict):
    llm = ChatOpenAI(
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
    
    structured_llm = llm.with_structured_output(ProblemSpec, method="json_schema")

    prompt = f"""
               System_Prompt : {SYSTEM_PROMPT}
               Business_Problemm : {business_problem}
               Dataset : {dataset}

               Translate this into Machine learning problem 

              """

    response = structured_llm.invoke(prompt)

    return response 
