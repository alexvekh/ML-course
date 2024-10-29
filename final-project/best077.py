# import warnings
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import cross_val_score, RandomizedSearchCV
# from sklearn.preprocessing import (
#     PowerTransformer,
#     KBinsDiscretizer,
#     TargetEncoder)
# from sklearn.preprocessing import KBinsDiscretizer, StandardScaler,  PowerTransformer, LabelEncoder, OrdinalEncoder
# from sklearn.compose import make_column_transformer, make_column_selector
# from sklearn.feature_selection import SelectKBest
# from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.pipeline import make_pipeline
# from imblearn.over_sampling import SMOTE
# from ydata_profiling import ProfileReport
# from sklearn.impute import SimpleImputer
# # from imblearn.pipeline import Pipeline
# # from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
# from sklearn.model_selection import StratifiedKFold


import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, StandardScaler, TargetEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer



#%% 1. Завантаження даних
print('Завантаження даних')
# data = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv')
# valid = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv')
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

# data = data.dropna(thresh=data.shape[1] - 5)           # Видаляємо рядки, в яких кількість пропусків більше ніж 10(9080)

\

y = data.pop("y")

# num_imputer = SimpleImputer().set_output(transform='pandas')
# data = num_imputer.fit_transform(data)
# valid = num_imputer.transform(valid) 


#%%
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')


##%%
num_imputer = SimpleImputer().set_output(transform='pandas')
data_num = num_imputer.fit_transform(data_num)
valid_num = num_imputer.transform(valid_num)  # Використовуємо transform замість fit

##%%
selected_columns = ['Var28', 'Var38', 'Var57', 'Var81', 'Var113', 'Var153']
data_number = data_num[selected_columns]
data_numcat = data_num.drop(columns=selected_columns)
valid_number = valid_num[selected_columns]
valid_numcat = valid_num.drop(columns=selected_columns)

##%%
kbin_imputer = KBinsDiscretizer(n_bins=5, encode='ordinal').set_output(transform='pandas')
data_number = kbin_imputer.fit_transform(data_number)
valid_number = kbin_imputer.transform(valid_number)  # Використовуємо transform замість fit

##%%
data_num = pd.concat([data_number, data_numcat], axis=1)
valid_num = pd.concat([valid_number, valid_numcat], axis=1)

##%%
func_imputer = FunctionTransformer(lambda x: x.astype(int).astype(str)).set_output(transform='pandas')
data_num = func_imputer.fit_transform(data_num)
valid_num = func_imputer.transform(valid_num)  # Використовуємо transform замість fit

##%%
data = pd.concat([data_num, data_cat], axis=1)
valid = pd.concat([valid_num, valid_cat], axis=1)


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
    'selectkbest__k': [20, 15],
    'smote__k_neighbors': [13, 9],
    'gradientboostingclassifier__subsample': [0.65, 0.55],
    'gradientboostingclassifier__max_depth': [9, 7]
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
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_val_score(
        estimator=model,
        X=data,
        y=y,
        scoring='balanced_accuracy',
        cv=skf,
        n_jobs=-1)
print(f"Pipe's ballance accuracy on CV: {cv_results.mean()}")
cv_results.mean()



# #%%
# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# print(confusion_matrix(y_test, pred)) 
# print(f"Pipe's accuracy is: {accuracy_score(y_test, pred):.1%}")




#%% Формування .csv файлу для результатів

model.fit(data, y)

output = pd.DataFrame({'index': valid.index,
                       'y': model.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)