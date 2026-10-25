import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from src.ml_agent.schemas.feature_spec import featurespec


def engineer_features(
        df: pd.DataFrame,
        target_column: str,
        #numeric_features: list[str],
        #categorical_features: list[str],
        featurespec:featurespec):
    
    X = df.drop(columns=[target_column])
    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        .astype(float)
    )
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


    numeric_features = featurespec.numeric_features
    categorical_features = featurespec.categorical_features


    numeric_steps = [
        ('imputer', SimpleImputer(strategy=featurespec.numeric_imputation)),
                                ]
    if featurespec.scaling_strategy=="standard":
        numeric_steps.append(
            ("scaler",StandardScaler())
        )
    elif featurespec.scaling_strategy=="robust":
        numeric_steps.append(
            ("scaler",RobustScaler())
             )

    numeric_pipeline = Pipeline(numeric_steps)


    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=featurespec.categorical_imputation,
                              **(
                                  {"fill_value":"missing"}
                                  if featurespec.categorical_imputation=="constant"
                                  else {}
                              )
                              )),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
                            ])


    transformer = []

    if featurespec.numeric_features:
        transformer.append(("numeric",numeric_pipeline,featurespec.numeric_features))

    if featurespec.categorical_features:
        transformer.append(("categorical",categorical_pipeline,featurespec.categorical_features))

    processor = ColumnTransformer(
        transformers=transformer,
        remainder="drop"
                                )
    
    X_train_transformed = processor.fit_transform(X_train)
    X_test_transformed  = processor.transform(X_test)

    return {
        "X_train" : X_train_transformed,
        "X_test"  : X_test_transformed,
        "y_train" : y_train,
        "y_test"  : y_test,
        "preprocessor" : processor
    }
