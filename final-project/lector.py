import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import (
    PowerTransformer,
    KBinsDiscretizer,
    TargetEncoder)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler,  PowerTransformer, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
# from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.model_selection import StratifiedKFold


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



#%%
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')


cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
data_cat = cat_imputer.fit_transform(data_cat)
valid_cat = cat_imputer.transform(valid_cat)

data_cat.fillna('None', inplace=True)
valid_cat.fillna('None', inplace=True)

cat_imputer = ce.OneHotEncoder(handle_unknown='ignore', ).set_output(transform='pandas')
data_cat = cat_imputer.fit_transform(data_cat)
valid_cat = cat_imputer.transform(valid_cat)


##%%
selected_columns = ['Var38', 'Var57', 'Var81', 'Var113', 'Var153']
data_number = data_num[selected_columns]
data_numcat = data_num.drop(columns=selected_columns)
valid_number = valid_num[selected_columns]
valid_numcat = valid_num.drop(columns=selected_columns)

##%%

number_imputer = SimpleImputer().set_output(transform='pandas')
data_number = number_imputer.fit_transform(data_number)
valid_number = number_imputer.transform(valid_number)  # Використовуємо transform замість fit


##%%

data_numcat.fillna(0, inplace=True)
valid_numcat.fillna(0, inplace=True)

##%%
data_numcat
number_imputer = SimpleImputer().set_output(transform='pandas')
data_numcat = number_imputer.fit_transform(data_numcat)
valid_numcat = number_imputer.transform(valid_numcat)


# data_numcat = data_numcat.astype(int).astype(str)
# valid_numcat = valid_numcat.astype(int).astype(str)


##%%

data_num = pd.concat([data_number, data_numcat], axis=1)
valid_num = pd.concat([valid_number, valid_numcat], axis=1)

##%%
# func_imputer = FunctionTransformer(lambda x: x.astype(int).astype(str)).set_output(transform='pandas')
# data_num = func_imputer.fit_transform(data_number)
# valid_num = func_imputer.transform(valid_number)  # Використовуємо transform замість fit

##%%
data = pd.concat([data_num, data_cat], axis=1)
valid = pd.concat([valid_num, valid_cat], axis=1)




##%%
# smote = SMOTE(k_neighbors=13, random_state=42)
# data = smote.fit_resample(data, y)

#%%

# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
# Застосовуємо SMOTE тільки до тренувальних даних
#smote = SMOTE(k_neighbors=13, random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)




# #%%
# print('Створення preprocessor для обробки даних')
# # report = ProfileReport(train)
# # report.to_notebook_iframe()

# preprocessor = make_column_transformer(
#     # 1. Категоріальні дані
#     (Pipeline(steps=[
#         #('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', TargetEncoder(random_state=42))
#     ]), make_column_selector(dtype_include=object)),
    
#     # 2. Числові стовпці, що не ввійшли у вибрані
#     (Pipeline(steps=[
#         #('imputer', SimpleImputer(strategy='mean')),
#         ('to_str', FunctionTransformer(lambda x: x.astype(int).astype(str))),
#         ('encoder', TargetEncoder(random_state=42))
#     ]), make_column_selector(dtype_include=np.number, pattern="^Var(?!28|38|81|113|153).*")),
    
#     # 3. Вибрані числові стовпці
#     (Pipeline(steps=[
#         #('imputer', SimpleImputer(strategy='mean')),
#         ('kbins', KBinsDiscretizer(n_bins=5, encode='ordinal')),
#         ('to_str', FunctionTransformer(lambda x: x.astype(int).astype(str))),
#         ('encoder', TargetEncoder(random_state=42))
#     ]), ['Var28', 'Var38', 'Var81', 'Var113', 'Var153']),
    
#     remainder='passthrough'
# )

#%% 
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
model = make_pipeline(
    # make_column_transformer(
    #     (TargetEncoder(random_state=42),
    #      make_column_selector(dtype_include=object)),
    #     remainder='passthrough',
    # n_jobs=-1),
    SelectKBest(),  
    # StandardScaler(),
    PowerTransformer(),
    # PCA(n_components=0.96),
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
    'selectkbest__k': [30, 40],
    'smote__k_neighbors': [15, 13],
    'gradientboostingclassifier__subsample': [0.57, 0.59],
    'gradientboostingclassifier__max_depth': [15, 20]
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