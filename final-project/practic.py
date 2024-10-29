import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import category_encoders as ce

    


# %%

try:
    data = pd.read_csv('./datasets/kaggle/final_proj_data.csv')
    valid = pd.read_csv('./datasets/kaggle/final_proj_test.csv')
except FileNotFoundError:
    data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
    valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')   
data.info()

# Видалення клонок з пропусками більше 30%
columns_to_drop = data.columns[data.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 30%
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)

# Видаляємо ознаки з високою кореляцією (шукав їх в таблиці кореляції)
columns_to_drop = ['Var22', 'Var160', 'Var227', 'Var228', 'Var195', 'Var207', 'Var21'] 
# columns_to_drop = ['Var6', 'Var22', 'Var25', 'Var38', 'Var85', 'Var109', 'Var119', 'Var126', 'Var133', 'Var153', 'Var163', 'Var123', 'Var140', 'Var24', 'Var81', 'Var83', 'Var112'] 
data = data.drop(columns=columns_to_drop)                  
valid = valid.drop(columns=columns_to_drop)

# Видаляємо сатегоріальні ознаки з високим з'язком. (Шукав за хі-квадрат (χ²))
columns_to_drop = ['Var212', 'Var216', 'Var197', 'Var199', 'Var206', 'Var210', 'Var192', 'Var193', 'Var203', 
                   'Var211', 'Var208', 'Var198', 'Var221', 'Var202', 'Var217' , 'Var219', 'Var218', 'Var204', 'Var220', 'Var226'] 
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)


# data = data.dropna()                                   # Видаляємо рядки з пропусками буде 5200
data = data.dropna(thresh=data.shape[1] - 5)           # Видаляємо рядки, в яких кількість пропусків більше ніж 10(9080)
data.info()

# %%
y = data.pop("y")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# %% 

cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        #('encoder', ce.TargetEncoder()) 
    ])

num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('kbins', KBinsDiscretizer(encode='ordinal')),
        ('kbins_str', FunctionTransformer(lambda x: x.astype(int).astype(str))) 
    ])

# %%

col_processor = (ColumnTransformer(
    transformers=[
        ('cat',
         cat_transformer,
         make_column_selector(dtype_include=object)),
        ('num',
         num_transformer,
         make_column_selector(dtype_exclude=object))],
    n_jobs=-1,
    verbose_feature_names_out=False)
    .set_output(transform='pandas'))

# %%

clf_estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
# clf_estimator = SVC(class_weight='balanced',
#           kernel='poly',
#           probability=True,
#           random_state=42)

# %%

clf_pipe_model = (Pipeline(
    steps=[
        #('lod_transformer', LenOfDescriptionTransformer()),
        #('other_transformers', OtherCustomTransformers()),
        ('col_processor', col_processor),
        ('encoder', ce.TargetEncoder().set_output(transform='pandas')),
        ('scaler', StandardScaler().set_output(transform='pandas')),
        ('clf_estimator', clf_estimator)
    ]))


clf_model = clf_pipe_model.fit(X_train, y_train)

# %%

pred_pipe = clf_model.predict(X_test)

# print(confusion_matrix(y_test, pred_pipe))
print(f"Pipe's accuracy is: {accuracy_score(y_test, pred_pipe):.1%}")

# %%

# pet = pd.DataFrame(
#     data={
#         'Type': 'Cat', # Dolphin
#         'Age': 3,
#         'Breed1': 'Tabby',
#         'Gender': 'Male',
#         'Color1': 'Black',
#         'Color2': 'White',
#         'MaturitySize': 'Small',
#         'FurLength': 'Short',
#         'Vaccinated': 'No',
#         'Sterilized': 'No',
#         'Health': 'Healthy',
#         'Fee': 5,
#         'Description': f'{"a"*1500}',
#         'PhotoAmt': 2
#     },
#     index=[0])

# pet_pred_proba = clf_model.predict_proba(pet).flatten()

# print(f'This pet has a {pet_pred_proba[1]:.1%} probability "of getting adopted"')
# print('==========================')

# %%
###X_train_transformed = clf_pipe_model[:-1].transform(X_train)

# %%
###pet_transformed1 = clf_pipe_model[:5].transform(pet)

# %%

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

cv_results = cross_val_score(
    estimator=clf_pipe_model,
    X=X,
    y=y,
    scoring='accuracy',
    cv=5,
    verbose=1)

acc_cv = cv_results.mean()

print(f"Pipe's accuracy on CV: {acc_cv:.1%}")

# %%

parameters = {
    'clf_estimator__max_depth': (None, 5),
    'clf_estimator__max_features': ('sqrt', 0.75)}

search = (GridSearchCV(
    estimator=clf_pipe_model,
    param_grid=parameters,
    scoring='accuracy',
    cv=5,
    refit=False)
    .fit(X, y))

# %%

parameters_best = search.best_params_
clf_pipe_model = clf_pipe_model.set_params(**parameters_best)

model_upd = clf_pipe_model.fit(X, y)

# %%

cv_results_upd = cross_val_score(
    estimator=model_upd,
    X=X,
    y=y,
    scoring='accuracy',
    cv=5)

acc_cv_upd = cv_results_upd.mean()
print(f"Pipe's UPDATED accuracy on CV: {acc_cv_upd:.1%}")



#%% Predict(valid) to CSV

output = pd.DataFrame({'index': valid.index,
                       'y': model_upd.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)