from typing import Literal , Optional
from pydantic import BaseModel, Field

class ProblemSpec(BaseModel):
    business_objective : str = Field(description = "The business outcome user wants to improve")
    ml_task_type : Literal[
        "binary_classification",
        "regression",
        "forecasting",
        "clustering"
                            ]
    prediction_unit : str = Field(description="What one prediction represents eg. customer or region")
    target_column : str | None = Field(description="Proposed target column else null if unknown")
    target_definition : str = Field(description="What target means and how shud it be labelled")
    prediction_horizon : str = Field(description="When the prediction applies ")
    candidate_features : str = Field(description="Potential input columns based only on given data")
    excluded_features : str = Field(description="Columns that shud not be used with reasons")
    primary_metric : str = Field(description="Recommended primary evaluation metric and why")
    secondary_metric : str = Field(description="Additional useful evaluation metric")
    validation_strategy : Literal ["random_split","stratified_split","temporal_split"]
    assumption: list[str] = Field(description="Assumptions the agent made due to missing information")
    clarification_questions: list[str] = Field(description="Questions that must be answered before proceeding")
    status : Literal["ready_for_eda","needs_clarification"]


class FramingOutput(BaseModel):
    clarification_questions: list[str] = Field(description="Questions that must be answered before proceeding")
    status : Literal["ready_for_eda","needs_clarification"]
    spec: Optional[ProblemSpec] = Field(description="Null if needs_clarification")


    

