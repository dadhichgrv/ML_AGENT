from typing import TypedDict
import pandas as pd
import os
from src.ml_agent.agents.feature_agent import create_feature_agent
from src.ml_agent.agents.problem_agent import create_problem_agent
from src.ml_agent.agents.eda_agent import create_eda_agent
from src.ml_agent.agents.model_training_agent import model_training_agent
from src.ml_agent.agents.feature_eng_tools import engineer_features
from src.ml_agent.agents.model_training_tools import train_model
from src.ml_agent.agents.model_testing_tools import test_model
from langgraph.graph import START, END, StateGraph


class MLWorkflowState(TypedDict):
    df: pd.DataFrame
    problem_statement : str
    llm: object

    problemspec: object 
    edareport: object 
    edaresult: dict
    featurespec: object
    processed_data: dict
    modeltrainingspec: object
    trainingresults: dict
    testingresults: dict



def problem_analysis_node(state:MLWorkflowState):
    llm = state["llm"]
    problem_statement = state["problem_statement"]
    df = state["df"]
    result = create_problem_agent(business_problem = problem_statement, dataset = df)
    return {"problemspec":result}


def eda_tool_node(state:MLWorkflowState):
    llm = state["llm"]
    df = state["df"]
    problemspec = state["problemspec"]
    result = create_eda_agent(problem_spec = problemspec, df = df ) 
    return {"edaresult":result}

# def eda_analysis_node(state:MLWorkflowState):
#     llm = state["llm"]
#     df = state["df"]
#     problemspec = state["problemspec"]
#     edareport = analyse_dataset(csv_path = df, target_column= problemspec.target_column) 
#     return {"edareport":edareport}

def feature_agent_node(state:MLWorkflowState):
    llm = state["llm"]
    problemspec = state["problemspec"]
    edaresult = state["edaresult"]
    df = state["df"]
    feature_plan = create_feature_agent(problemspec, edaresult, df.columns.tolist())
    return {"featurespec":feature_plan}


def feature_engineering_node(state:MLWorkflowState):
    featurespec = state["featurespec"]
    df = state["df"]
    target_column = featurespec.target_column
    result = engineer_features(df,target_column,featurespec)
    return {"processed_data":result}


def model_training_agent_node(state:MLWorkflowState):
    processed = state["processed_data"]
    summary = {
        "n_train": processed["X_train"].shape[0],
        "n_features": processed["X_train"].shape[1],
        "target": state["featurespec"].target_column,
        "task_type": state["problemspec"].ml_task_type,
                 }
    results = model_training_agent(summary)
    return {"modeltrainingspec":results}


def model_training_node(state:MLWorkflowState):
    modeltrainingspec = state["modeltrainingspec"]
    processed = state["processed_data"]
    problemspec = state["problemspec"]

    results = train_model(
                         X_train_transformed=processed["X_train"],
                         y_train=processed["y_train"],
                         candidate_models=modeltrainingspec.candidate_models,
                         task_type=problemspec.ml_task_type,
                         evaluation_metric=modeltrainingspec.evaluation_metric
                         )
    return {"trainingresults":results}

def model_testing_node(state:MLWorkflowState):
    processed = state["processed_data"]
    trainingresults = state["trainingresults"]
    result = test_model( X_test_transformed=processed["X_test"],
                         y_test=processed["y_test"],
                         model=trainingresults['best_model'],
                         evaluation_metric=trainingresults['evaluation_metric'])
    return {"testingresults":result}

graph = StateGraph(MLWorkflowState)

graph.add_node("problem_analysis_node",problem_analysis_node)
graph.add_node("eda_tool_node",eda_tool_node)
# graph.add_node("eda_analysis_node",eda_analysis_node)
graph.add_node("feature_agent_node",feature_agent_node)
graph.add_node("feature_engineering_node",feature_engineering_node)
graph.add_node("model_training_agent_node",model_training_agent_node)
graph.add_node("model_training_node",model_training_node)
graph.add_node("model_testing_node",model_testing_node)

graph.add_edge(START,"problem_analysis_node")
graph.add_edge("problem_analysis_node","eda_tool_node")
graph.add_edge("eda_tool_node","feature_agent_node")
#graph.add_edge("eda_tool_node","eda_analysis_node")
#graph.add_edge("eda_analysis_node","feature_agent_node")
graph.add_edge("feature_agent_node","feature_engineering_node")
graph.add_edge("feature_engineering_node","model_training_agent_node")
graph.add_edge("model_training_agent_node","model_training_node")
graph.add_edge("model_training_node","model_testing_node")
graph.add_edge("model_testing_node",END)

workflow = graph.compile()



