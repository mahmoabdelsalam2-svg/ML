# automl_api/routers/tabular_router.py

import uuid
import pandas as pd
from pathlib import Path
import httpx
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from automl_api import settings
from automl_api.logger import loguru_logger as logger

router = APIRouter()
job_registry = {}

base_estimators_clf = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]
base_estimators_reg = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('svr', SVR())
]

stacked_clf = StackingClassifier(estimators=base_estimators_clf, final_estimator=LogisticRegression())
stacked_reg = StackingRegressor(estimators=base_estimators_reg, final_estimator=LinearRegression())

MODEL_MAP = {
    "Classification": {
        "Stacked Ensemble": stacked_clf,
        "Support Vector Machine (SVM)": SVC(probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    },
    "Regression": {
        "Stacked Ensemble": stacked_reg,
        "Linear Regression": LinearRegression(),
        "Support Vector Regressor (SVR)": SVR(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
    }
}

AUTOML_CLASSIFICATION_MODELS = ["Stacked Ensemble", "Random Forest", "Logistic Regression"]

PARAM_GRIDS = {
    "Random Forest": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10]
    },
    "Random Forest Regressor": {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10]
    }
}


class UrlAnalysisRequest(BaseModel):
    url: str
    model_name: str
    problem_type: str
    target_column: str


class AutoMLRequest(BaseModel):
    url: str
    target_column: str


def load_dataframe(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return df


def perform_single_model_analysis(job_id: str, df: pd.DataFrame, model_name: str, problem_type: str,
                                  target_column: str):
    job_registry[job_id]["status"] = "preprocessing"

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                temp_dt = pd.to_datetime(X[col], errors='coerce')
                if temp_dt.notna().sum() / len(temp_dt) > 0.5:
                    logger.info(f"Detected and processing date column: {col}")
                    X[f'{col}_year'] = temp_dt.dt.year
                    X[f'{col}_month'] = temp_dt.dt.month
                    X[f'{col}_day'] = temp_dt.dt.day
                    X = X.drop(columns=[col])
            except (ValueError, TypeError):
                continue

    is_imbalanced = False
    if problem_type.lower() == 'classification':
        class_counts = y.value_counts(normalize=True)
        if class_counts.min() < 0.25:
            is_imbalanced = True
            logger.info("Imbalanced dataset detected. SMOTE will be applied.")

    cols_to_drop = ['alive', 'deck', 'embark_town', 'who', 'adult_male', 'class']
    existing_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    X = X.drop(columns=existing_cols_to_drop)

    if 'name' in X.columns:
        X['title'] = X['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = X['title'].value_counts()[X['title'].value_counts() < 10].index
        X['title'] = X['title'].replace(rare_titles, 'Rare')
        X = X.drop(columns=['name'])

    # Identify different feature types based on data type and cardinality
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    object_features = X.select_dtypes(exclude=np.number).columns.tolist()
    categorical_features = [col for col in object_features if X[col].nunique() < 50]
    text_features = [col for col in object_features if col not in categorical_features]

    logger.info(f"Numeric features identified: {numeric_features}")
    logger.info(f"Categorical features (low cardinality): {categorical_features}")
    logger.info(f"Text features (high cardinality): {text_features}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]

    for col in text_features:
        text_transformer = TfidfVectorizer(stop_words='english', max_features=1000)
        transformers.append((f'text_{col}', text_transformer, col))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        sparse_threshold=0.0
    )

    feature_selector = SelectKBest(score_func=f_classif, k='all')
    if problem_type == 'Regression':
        feature_selector = SelectKBest(score_func=f_regression, k='all')

    try:
        model = MODEL_MAP[problem_type][model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' not found for problem type '{problem_type}'.")

    pipeline_steps = [
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector)
    ]

    if problem_type == 'Classification' and is_imbalanced:
        pipeline_steps.append(('smote', SMOTE(random_state=42)))

    model_step_name = 'classifier' if problem_type == 'Classification' else 'regressor'
    pipeline_steps.append((model_step_name, model))

    pipeline = ImbPipeline(steps=pipeline_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid_key = model_name
    if problem_type == 'Regression' and 'Regressor' not in model_name:
        param_grid_key = f"{model_name} Regressor"

    if param_grid_key in PARAM_GRIDS:
        logger.info(f"Starting randomized search for {model_name}...")
        job_registry[job_id]["status"] = f"tuning_{model_name.replace(' ', '_')}"

        scoring = 'accuracy' if problem_type == 'Classification' else 'r2'

        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=PARAM_GRIDS.get(param_grid_key, {}),
            n_iter=10,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
        best_model_pipeline = random_search.best_estimator_
    else:
        job_registry[job_id]["status"] = f"training_{model_name.replace(' ', '_')}"
        pipeline.fit(X_train, y_train)
        best_model_pipeline = pipeline

    job_registry[job_id]["status"] = "evaluating"
    predictions = best_model_pipeline.predict(X_test)

    metrics = {}
    if problem_type == 'Classification':
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
    else:  # Regression
        metrics['r2_score'] = r2_score(y_test, predictions)
        metrics['mean_squared_error'] = mean_squared_error(y_test, predictions)

    return {"model_name": model_name, "problem_type": problem_type, "target_column": target_column, "metrics": metrics,
            "smote_applied": is_imbalanced}


def run_tabular_analysis_from_file(job_id: str, file_path: Path, model_name: str, problem_type: str,
                                   target_column: str):
    try:
        job_registry[job_id]["status"] = "loading_data"
        df = load_dataframe(file_path)
        results = perform_single_model_analysis(job_id, df, model_name, problem_type, target_column)
        job_registry[job_id].update({"status": "complete", "results": results})
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}", exc_info=True)
        job_registry[job_id].update({"status": "failed", "error": str(e)})


def run_automl_analysis(job_id: str, df: pd.DataFrame, target_column: str):
    try:
        all_results = []
        for name in AUTOML_CLASSIFICATION_MODELS:
            result = perform_single_model_analysis(job_id, df.copy(), name, "Classification", target_column)
            all_results.append(result)

        if not all_results:
            raise ValueError("No models were successfully trained.")

        best_model = max(all_results, key=lambda x: x['metrics']['accuracy'])
        job_registry[job_id].update(
            {"status": "complete", "results": {"best_model": best_model, "comparison": all_results}})
    except Exception as e:
        logger.error(f"AutoML analysis failed for job {job_id}: {e}", exc_info=True)
        job_registry[job_id].update({"status": "failed", "error": str(e)})


async def download_file(url: str, dest_path: Path):
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f: f.write(r.content)


@router.post("/analyze-file")
async def analyze_data_from_file(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                                 model_name: str = Form(...), problem_type: str = Form(...),
                                 target_column: str = Form(...)):
    job_id = str(uuid.uuid4())
    file_path = settings.STORAGE_PATH / f"{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer: buffer.write(await file.read())
    job_registry[job_id] = {"status": "pending"}
    background_tasks.add_task(run_tabular_analysis_from_file, job_id, file_path, model_name, problem_type,
                              target_column)
    return {"job_id": job_id}


@router.post("/analyze-url")
async def analyze_data_from_url(request: UrlAnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    file_path = settings.STORAGE_PATH / f"{job_id}{Path(request.url).suffix or '.csv'}"
    job_registry[job_id] = {"status": "downloading"}
    try:
        await download_file(request.url, file_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to download file: {e}")
    background_tasks.add_task(run_tabular_analysis_from_file, job_id, file_path, request.model_name,
                              request.problem_type, request.target_column)
    return {"job_id": job_id}


@router.post("/automl")
async def analyze_automl(request: AutoMLRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    file_path = settings.STORAGE_PATH / f"{job_id}{Path(request.url).suffix or '.csv'}"
    job_registry[job_id] = {"status": "downloading"}
    try:
        await download_file(request.url, file_path)
        df = load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to process file: {e}")
    background_tasks.add_task(run_automl_analysis, job_id, df, request.target_column)
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in job_registry: raise HTTPException(404, "Job not found")
    return job_registry[job_id]


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in job_registry or job_registry[job_id].get("status") != "complete":
        raise HTTPException(404, "Results not ready or job not found.")
    return job_registry[job_id]["results"]