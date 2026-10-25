import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score, root_mean_squared_error
from src.ml_agent.schemas.feature_spec import featurespec


def train_model(
        X_train_transformed,
        y_train,
        candidate_models,
        task_type,
        evaluation_metric):
    
    X_fit,X_val,y_fit,y_val = train_test_split(X_train_transformed,y_train,test_size=0.2,random_state=42)
    
    models = {
        "linear_regression": LinearRegression(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest_classifier": RandomForestClassifier(
                            n_estimators=100,
                            random_state=42
                                                ),
        "random_forest_regressor": RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                                                )

                }


    results = {}
    trained_models={}

    for model_name in candidate_models:
        model = models[model_name]

        #pipeline = featurespec.preprocessor
        model.fit(X_train_transformed,y_train)
        y_val_pred = model.predict(X_val)


        if task_type=="classification":
            if evaluation_metric == "accuracy":
                score = accuracy_score(y_val, y_val_pred)

            elif evaluation_metric=="f1":
                score = f1_score(y_val, y_val_pred,average="weighted")
            elif evaluation_metric=="roc_auc":
                score = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])

        else:
            if evaluation_metric=="mae":
                    score = mean_absolute_error(y_val, y_val_pred)
            elif evaluation_metric=="rmse":
                    score = root_mean_squared_error(y_val, y_val_pred)
            elif evaluation_metric=="r2":
                    score = r2_score(y_val, y_val_pred)


        results[model_name]=score
        trained_models[model_name]=model

    minimize_metrics={"mae","rmse"}
    best_model_name = (
         min(results, key=results.get)
         if evaluation_metric in minimize_metrics
         else max(results, key=results.get)
    )

    best_model = trained_models[best_model_name]
    best_model.fit(X_train_transformed,y_train)

    return {
         "best_model": best_model,
         "best_model_name": best_model_name,
         "validation_scores":results,
         "evaluation_metric":evaluation_metric
    }
            


 


   