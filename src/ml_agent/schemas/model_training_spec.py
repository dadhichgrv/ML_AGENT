from typing import Literal 
from pydantic import BaseModel , Field

class modeltrainingspec(BaseModel):
    candidate_models: list[Literal["logistic_regression","linear_regression",
                              "random_forest_classifier","random_forest_regressor"
                             ]
                            ] = Field(description="Candidate algorithm suitable for this task")
    
    evaluation_metric: Literal["accuracy","f1","roc_auc","mae","rmse","r2"]
                            
    reasoning: str = Field(description="Why these algorithms and metric were selected")

