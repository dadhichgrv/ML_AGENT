import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score, root_mean_squared_error



def test_model(
        X_test_transformed,
        y_test,
        model,
        evaluation_metric):
    y_test_pred = model.predict(X_test_transformed)

    if evaluation_metric == "accuracy":
        score = accuracy_score(y_test, y_test_pred)
    elif evaluation_metric=="f1":
        score = f1_score(y_test, y_test_pred,average="weighted")
    elif evaluation_metric=="roc_auc":
        score = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:,1])
    elif evaluation_metric=="mae":
        score = mean_absolute_error(y_test, y_test_pred)
    elif evaluation_metric=="rmse":
        score = root_mean_squared_error(y_test, y_test_pred)
    elif evaluation_metric=="r2":
        score = r2_score(y_test, y_test_pred)


    return {
         "test_scores":score
    }
            


 


   