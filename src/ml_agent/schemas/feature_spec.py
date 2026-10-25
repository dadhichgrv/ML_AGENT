from typing import Literal 
from pydantic import BaseModel , Field

class featurespec(BaseModel):
    target_column: str 
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)

    excluded_features: list[str] = Field(default_factory=dict, 
                                             description="Excluded features")
    numeric_imputation: Literal["mean","median"]
    categorical_imputation: Literal["most_frequent","constant"]

    categorical_encoding: Literal["one_hot"]
    scaling_strategy: Literal["none","standard","robust"]

    reasoning: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    status : Literal["ready","needs_review"]

