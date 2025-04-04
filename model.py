import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

train_IDs = train_df['Id']
test_IDs = test_df['Id']

y_train = np.log1p(train_df['SalePrice'])

train_features = train_df.drop(['Id', 'SalePrice'], axis=1)
test_features = test_df.drop(['Id'], axis=1)

all_features = pd.concat([train_features, test_features], axis=0)
print(f"Combined features shape: {all_features.shape}")

numeric_features = all_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = all_features.select_dtypes(include=['object']).columns.tolist()

print(f"Number of numeric features: {len(numeric_features)}")
print(f"Number of categorical features: {len(categorical_features)}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# XGBoost Pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

# LightGBM Pipeline
lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMRegressor(objective='regression', random_state=42))
])

# CatBoost Pipeline
cb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', cb.CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=False))
])

xgb_param_grid = {
    'model__n_estimators': [100, 500],
    'model__learning_rate': [0.01, 0.05],
    'model__max_depth': [3, 5],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

lgb_param_grid = {
    'model__n_estimators': [100, 500],
    'model__learning_rate': [0.01, 0.05],
    'model__num_leaves': [31, 50],
    'model__subsample': [0.8]
}

cb_param_grid = {
    'model__iterations': [100, 500],
    'model__learning_rate': [0.01, 0.05],
    'model__depth': [6, 8]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def run_grid_search(pipeline, param_grid, name):
    print(f"\nTuning {name} model...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kf,
        scoring=rmsle_scorer,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(train_features, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f} RMSLE")

    return grid_search.best_estimator_

xgb_pipeline.set_params(
    model__n_estimators=500,
    model__learning_rate=0.05,
    model__max_depth=5,
    model__subsample=0.8,
    model__colsample_bytree=0.8
)
xgb_scores = cross_val_score(xgb_pipeline, train_features, y_train, cv=kf, scoring=rmsle_scorer)
print(f"XGBoost CV RMSLE: {-np.mean(xgb_scores):.4f} (±{np.std(xgb_scores):.4f})")
xgb_pipeline.fit(train_features, y_train)

lgb_pipeline.set_params(
    model__n_estimators=500,
    model__learning_rate=0.05,
    model__num_leaves=31,
    model__subsample=0.8
)
lgb_scores = cross_val_score(lgb_pipeline, train_features, y_train, cv=kf, scoring=rmsle_scorer)
print(f"LightGBM CV RMSLE: {-np.mean(lgb_scores):.4f} (±{np.std(lgb_scores):.4f})")
lgb_pipeline.fit(train_features, y_train)

cb_pipeline.set_params(
    model__iterations=500,
    model__learning_rate=0.05,
    model__depth=6
)
cb_scores = cross_val_score(cb_pipeline, train_features, y_train, cv=kf, scoring=rmsle_scorer)
print(f"CatBoost CV RMSLE: {-np.mean(cb_scores):.4f} (±{np.std(cb_scores):.4f})")
cb_pipeline.fit(train_features, y_train)

xgb_preds = np.expm1(xgb_pipeline.predict(test_features))
lgb_preds = np.expm1(lgb_pipeline.predict(test_features))
cb_preds = np.expm1(cb_pipeline.predict(test_features))