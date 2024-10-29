import warnings
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import (
    PowerTransformer,
    KBinsDiscretizer,
    TargetEncoder)
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport

#%% 1. Завантаження даних
try:
    data = pd.read_csv('./datasets/kaggle/final_proj_data.csv')
    valid = pd.read_csv('./datasets/kaggle/final_proj_test.csv')
except FileNotFoundError:
    data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
    valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')   
data.info()

#%%

y = data.pop("y")
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

##%%
#report = ProfileReport(train)
#report.to_notebook_iframe()

#%%
model = make_pipeline(
    make_column_transformer(
        (TargetEncoder(random_state=42),
         make_column_selector(dtype_include=object)),
        remainder='passthrough',
    n_jobs=-1),
    SelectKBest(),
    PowerTransformer(),
    SMOTE(random_state=42),
    KBinsDiscretizer(
        encode='onehot-dense',
        strategy='uniform',
        subsample=None,
        random_state=42),
    GradientBoostingClassifier(
        random_state=42))

#%%

# model.get_params().keys()

params = {
    'selectkbest__k': [10, 15],
    'smote__k_neighbors': [7, 9],
    'gradientboostingclassifier__subsample': [0.65, 0.85],
    'gradientboostingclassifier__max_depth': [5, 7]
}

rs = RandomizedSearchCV(
    model,
    params,
    n_jobs=-1,
    refit=False,
    random_state=42,
    verbose=1)

search = rs.fit(data, y)
search.best_params_

#%%

model.set_params(**search.best_params_)

#%%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    cv_results = cross_val_score(
        estimator=model,
        X=train,
        y=target,
        scoring='balanced_accuracy',
        cv=10,
        n_jobs=-1)

cv_results.mean()

#%%
model.fit(data, y)

#%% Формування .csv файлу для результатів
output = pd.DataFrame({'index': valid.index,
                       'y': model.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)