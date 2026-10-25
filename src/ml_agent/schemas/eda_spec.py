from pydantic import BaseModel, Field 
from typing import Literal 

class EDAReport(BaseModel):
    target_column: str 
    target_status : Literal["valid","missing","needs_review"]

    data_quality_issues: list[str] = Field(default_factory=list)
    target_findings: list[str] = Field(default_factory=list)
    numerical_feature_findings: list[str] = Field(default_factory=list)
    categorical_feature_findings: list[str] = Field(default_factory=list)
    leakage_warnings: list[str] = Field(default_factory=list)
    recommendations_for_feature_engineering: list[str] = Field(default_factory=list)

    overall_assessment: str 
    status: Literal["ready_for_feature_engineering","needs_review"]