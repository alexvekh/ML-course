# # Best pipeline: 
#     BernoulliNB(
#         XGBClassifier(
#             XGBClassifier(
#                 input_matrix, 
#                 learning_rate=0.5, 
#                 max_depth=3, 
#                 min_child_weight=8, 
#                 n_estimators=100, 
#                 n_jobs=1, 
#                 subsample=1.0, 
#                 verbosity=0), 
#             learning_rate=0.5, 
#             max_depth=3, 
#             min_child_weight=8, 
#             n_estimators=100, 
#             n_jobs=1, 
#             subsample=1.0, 
#             verbosity=0), 
#         alpha=1.0, 
#         fit_prior=False)
# #Точність на тестових даних: 0.991


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
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier


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

y = data.pop("y")
# Видалення клонок з пропусками більше 30%




# common_columns = data.columns.intersection(valid.columns)
# columns_to_drop = common_columns[data[common_columns].isna().mean() > 0.3] 


columns_to_drop = valid.columns[valid.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 30%
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)

#%%
# # Видаляємо ознаки з високою кореляцією (шукав їх в таблиці кореляції)
# columns_to_drop = ['Var22', 'Var160', 'Var227', 'Var228', 'Var195', 'Var207', 'Var21'] 
# # columns_to_drop = ['Var6', 'Var22', 'Var25', 'Var38', 'Var85', 'Var109', 'Var119', 'Var126', 'Var133', 'Var153', 'Var163', 'Var123', 'Var140', 'Var24', 'Var81', 'Var83', 'Var112'] 
# data = data.drop(columns=columns_to_drop)                  
# valid = valid.drop(columns=columns_to_drop)

# # Видаляємо сатегоріальні ознаки з високим з'язком. (Шукав за хі-квадрат (χ²))
# columns_to_drop = ['Var212', 'Var216', 'Var197', 'Var199', 'Var206', 'Var210', 'Var192', 'Var193', 'Var203', 
#                    'Var211', 'Var208', 'Var198', 'Var221', 'Var202', 'Var217' , 'Var219', 'Var218', 'Var204', 'Var220', 'Var226'] 
# data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
# valid = valid.drop(columns=columns_to_drop)

# data = data.dropna(thresh=data.shape[1] - 5)           # Видаляємо рядки, в яких кількість пропусків більше ніж 10(9080)

##%%
# y = data.pop("y")



#%%
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')



# # fill nan
# cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
# data_cat = cat_imputer.fit_transform(data_cat)
# valid_cat = cat_imputer.transform(valid_cat)

# data_cat.fillna('None', inplace=True)
# valid_cat.fillna('None', inplace=True)


mean_imputer = SimpleImputer(strategy='constant', fill_value='missing').set_output(transform='pandas')
data_cat = mean_imputer.fit_transform(data_cat)
valid_cat = mean_imputer.transform(valid_cat)

#%%

# # Визначення колонок, які краще заповнювати середнім значенням, які медіаною
# desc = data.describe()

# mean_columns = []
# median_columns = []

# for col in desc.columns:
#     # Перевірка на викиди за допомогою IQR (межі)
#     Q1 = desc[col]['25%']
#     Q3 = desc[col]['75%']
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     # Якщо є викиди, заповнюємо медіаною, інакше середнім
#     if (data[col] < lower_bound).any() or (data[col] > upper_bound).any():
#         median_columns.append(col)
#     else:
#         mean_columns.append(col)

# print("Колонки для заповнення середнім:", mean_columns)
# print("Колонки для заповнення медіаною:", median_columns)

mean_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
data_num = mean_imputer.fit_transform(data_num)
valid_num = mean_imputer.transform(valid_num)  # Використовуємо transform замість fit

##%%
# mean_columns = ['Var38', 'Var57', 'Var81', 'Var113', 'Var153']

# data_mean = data_num[mean_columns]
# data_median = data_num.drop(columns=mean_columns)
# valid_mean = valid_num[mean_columns]
# valid_median = valid_num.drop(columns=mean_columns)

##%%
# mean_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
# data_mean = mean_imputer.fit_transform(data_mean)
# valid_mean = mean_imputer.transform(valid_mean)  # Використовуємо transform замість fit

# ##%%
# #encode parameter are: 'onehot-dense', 'ordinal', 'onehot'.
# #strategy: This can be 'uniform', 'quantile', or 'kmeans'.
# kbin_imputer = KBinsDiscretizer(n_bins=20, encode='ordinal').set_output(transform='pandas')
# data_number = kbin_imputer.fit_transform(data_number)
# valid_number = kbin_imputer.transform(valid_number)  # Використовуємо transform замість fit
# # #%%
# func_imputer = FunctionTransformer(lambda x: x.astype(int).astype(str)).set_output(transform='pandas')
# data_number = func_imputer.fit_transform(data_number)
# valid_number = func_imputer.transform(valid_number)  # Використовуємо transform замість fit


##%%
# data_numcat.fillna(0, inplace=True)
# valid_numcat.fillna(0, inplace=True)
# median_imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
# data_median = median_imputer.fit_transform(data_median)
# valid_median = median_imputer.transform(valid_median)  # Використовуємо transform замість fit

# # # #%%
# kbin_imputer = KBinsDiscretizer(n_bins=5, encode='ordinal').set_output(transform='pandas')
# data_numcat = kbin_imputer.fit_transform(data_numcat)
# valid_numcat = kbin_imputer.transform(valid_numcat)  # Використовуємо transform замість fit

# func_imputer = FunctionTransformer(lambda x: x.astype(int).astype(str)).set_output(transform='pandas')
# data_numcat = func_imputer.fit_transform(data_numcat)
# valid_numcat = func_imputer.transform(valid_numcat)  # Використовуємо transform замість fit


##%%
# data_num = pd.concat([data_mean, data_median], axis=1)
# valid_num = pd.concat([valid_mean, valid_median], axis=1)




data = pd.concat([data_num, data_cat], axis=1)
valid = pd.concat([valid_num, valid_cat], axis=1)

from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier)
# #%% 

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
    XGBClassifier(
        learning_rate=0.5,
        max_depth=3,
        min_child_weight=8,
        n_estimators=100,
        n_jobs=1,
        subsample=1.0,
        verbosity=0
    )
    # або
    # BernoulliNB(alpha=1.0, fit_prior=False)
)


# Підготовка даних (X_train, y_train)
# model.fit(X_train, y_train)

# Перевірка точності на тестових даних
# accuracy = model.score(X_test, y_test)
# print(f"Точність на тестових даних: {accuracy:.3f}")

#%%

# model.get_params().keys()

params = {
    'selectkbest__k': [20, 40],
    'smote__k_neighbors': [13, 9],
    #'gradientboostingclassifier__subsample': [0.65, 0.55],
    #'gradientboostingclassifier__max_depth': [9, 7]
    #'bernoullinb__alpha': [1.0, 2.0],
    #'xgbclassifier__n_estimators': [100, 10]
}







rs = RandomizedSearchCV(
    model,
    params,
    n_jobs=-1,
    refit=False,
    random_state=42,
    n_iter=4,
    verbose=1)

search = rs.fit(data, y)
search.best_params_

# #%%

model.set_params(**search.best_params_)



# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#%% Навчаємо модель на тренувальному наборі
# grid_search.fit(X_train, y_train)

# #%% Оцінка на валідаційному наборі
# y_pred = grid_search.predict(X_valid)
# accuracy = accuracy_score(y_valid, y_pred)
# print(f'Accuracy on validation set: {accuracy:.4f}')

# #%% 11. Пере-навчання моделі на всіх даних (тренувальна + тестова вибірки)
# X_full = pd.concat([X_train, X_valid])
# y_full = pd.concat([y_train, y_valid])
# final_model = grid_search.best_estimator_.fit(X_full, y_full)

#%% 12. Прогноз для валідаційного набору і формування .csv файлу
# X_test = pd.read_csv('test_data.csv')  # Завантаження тестових даних
# y_test_pred = final_model.predict(X_test)




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