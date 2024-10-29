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
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, StandardScaler, TargetEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier)
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import ADASYN

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

#y = data.pop("y")
# Видалення клонок з пропусками більше 30%
columns_to_drop = data.columns[data.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 30%
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

# data = data.dropna(thresh=data.shape[1] - 10)           # Видаляємо рядки, в яких кількість пропусків більше ніж 10(9080)


##%%

y = data.pop("y")

#%%
# data_num = data.select_dtypes(include=np.number)
# data_cat = data.select_dtypes(include='object')
# valid_num = valid.select_dtypes(include=np.number)
# valid_cat = valid.select_dtypes(include='object')


# cat_imputer = SimpleImputer(strategy='constant', fill_value='missing').set_output(transform='pandas')
# data_cat = cat_imputer.fit_transform(data_cat)
# valid_cat = cat_imputer.transform(valid_cat)


# num_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
# data_num = num_imputer.fit_transform(data_num)
# valid_num = num_imputer.transform(valid_num)  # Використовуємо transform замість fit


# data = pd.concat([data_num, data_cat], axis=1)
# valid = pd.concat([valid_num, valid_cat], axis=1)






#%%
# #%% 
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.svm import SVC
from sklearn.decomposition import PCA

model = make_pipeline(
    make_column_transformer(
        # Для категорійних даних
        (Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target_encoder', TargetEncoder(random_state=42))
        ]), make_column_selector(dtype_include=object)),
        
        # Для числових даних
        (SimpleImputer(strategy='constant', fill_value=0), make_column_selector(dtype_include=np.number)),

        remainder='passthrough',
        n_jobs=-1
    ),
    SelectKBest(k=67),
    PowerTransformer(),
    SMOTE(k_neighbors=13, sampling_strategy=0.9, random_state=42),
    KBinsDiscretizer(encode='onehot-dense', strategy='uniform', subsample=None, random_state=42),
    # StandardScaler(),
    # PCA(n_components=0.96),
    # PolynomialFeatures(degree=2, interaction_only=False, include_bias=False, order='C'),
                # дуже довго

    # GradientBoostingClassifier(max_depth=7, random_state=42, subsample=0.65), # 0.74

    # RandomForestClassifier(
    #     n_estimators=200,          # кількість дерев
    #     max_depth=10,            # максимальна глибина дерева
    #     min_samples_split=3,       # мінімальна кількість зразків для поділу
    #     min_samples_leaf=3,        # мінімальна кількість зразків у листі
    #     max_features=0.5,       # кількість ознак для розгляду при поділі
    #     class_weight='balanced',   # ваги класів для балансування
    #     random_state=42            # фіксація випадковості
    #     )      
    # {'smote__k_neighbors': 13, 'selectkbest__k': 40, 'randomforestclassifier__min_samples_split': 3, 'randomforestclassifier__max_features': 0.5}
    # Pipe's ballance accuracy on CV: 0.8416038195201805
    # {'smote__k_neighbors': 15, 'selectkbest__k': 67, 'randomforestclassifier__min_samples_split': 3, 'randomforestclassifier__max_features': 0.5}
    # Pipe's ballance accuracy on CV: 0.8421414077320168



    # LogisticRegression(penalty='l2', class_weight='balanced', max_iter=800) # 0.8438
        # {'smote__k_neighbors': 15, 'logisticregression__solver': 'lbfgs', 'logisticregression__max_iter': 800} - 0.8320759497575345
        # {'smote__k_neighbors': 13, 'selectkbest__k': 65, 'logisticregression__max_iter': 800}   - 0.8403038494425769

        
    # SVC(class_weight='balanced', kernel='linear', probability=True, random_state=42)    
        # {'svc__kernel': 'rbf', 'svc__class_weight': None, 'selectkbest__k': 65}  # 0.8084

    # BaggingClassifier(estimator=KNeighborsClassifier(),
    #                   max_features=0.75, max_samples=0.75,
    #                   n_jobs=-1, random_state=42)

    # BaggingClassifier(estimator=LogisticRegression(penalty='l2', class_weight='balanced', max_iter=800),
    #                   max_features=0.75, max_samples=0.75,
    #                   n_jobs=-1, random_state=42)  # 0.8365192044575757

    # BaggingClassifier(
    #     RandomForestClassifier(
    #             n_estimators=200,          # кількість дерев
    #             max_depth=10,            # максимальна глибина дерева
    #             min_samples_split=3,       # мінімальна кількість зразків для поділу
    #             min_samples_leaf=3,        # мінімальна кількість зразків у листі
    #             max_features=0.5,       # кількість ознак для розгляду при поділі
    #             class_weight='balanced',   # ваги класів для балансування
    #             random_state=42            # фіксація випадковості
    #             ),
    #     max_features=0.75, 
    #     max_samples=0.75,
    #     n_jobs=-1, 
    #     random_state=42)  # 0.8365192044575757
    
    # KNeighborsClassifier(
    #     n_neighbors=35,          # Кількість сусідів
    #     weights='distance',      # Стратегія зважування сусідів
    #     algorithm='auto',       # Алгоритм для пошуку сусідів
    #     leaf_size=30,           # Розмір листа для алгоритму
    #     metric='minkowski',     # Метрика для обчислення відстані
    #     p=2,                    # Параметр для minkowski (1: мангетенська, 2: евклідова)
    #     metric_params=None,     # Додаткові параметри для метрики
    #     n_jobs=-1)               # Кількість потоків (-1: усі доступні процесори) # 0.7848

    # AdaBoostClassifier(algorithm='SAMME', random_state=42) # 0.8182
        # 'adaboostclassifier__n_estimators': 500, 'adaboostclassifier__learning_rate': 1.5

    # GaussianNB(priors=None, var_smoothing=1e-09)
    
    # StackingClassifier(
    #     estimators=[
    #         ('logreg', LogisticRegression(penalty='l2', class_weight='balanced', max_iter=800)),
    #         ('rf', RandomForestClassifier(
    #         n_estimators=200,          # кількість дерев
    #         max_depth=10,              # максимальна глибина дерева
    #         min_samples_split=3,       # мінімальна кількість зразків для поділу
    #         min_samples_leaf=3,        # мінімальна кількість зразків у листі
    #         max_features=0.5,          # кількість ознак для розгляду при поділі
    #         class_weight='balanced',   # ваги класів для балансування
    #         random_state=42            # фіксація випадковості
    #     ))                              
    #         ],
    #     final_estimator=GradientBoostingClassifier(subsample=0.75, max_features='sqrt', random_state=42)
    # )
    
    VotingClassifier(
        estimators=[
                ('logreg', LogisticRegression(penalty='l2', class_weight='balanced', max_iter=800)),
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=3, min_samples_leaf=3,
                             max_features=0.5, class_weight='balanced', random_state=42)),
                ('rf2', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=3, min_samples_leaf=3,
                             max_features=0.5, class_weight='balanced', random_state=42))  
                ],
        voting='soft') # Pipe's ballance accuracy on CV: 0.8469382809074666
)
    
       
    
    
    
    
    
# #%%

# model.get_params().keys()

params = {
    'selectkbest__k': [67, 40],
    'smote__k_neighbors': [15, 19],
    #'pca__n_components': [0.96, 0.94],
    #'borderlinesmote__m_neighbors': [5, 3],
    #'borderlinesmote__k_neighbors': [20, 30],
    #'logisticregression__max_iter': [800, 1000],
    #'logisticregression__solver': ['lbfgs', 'balanced'],
    #'gradientboostingclassifier__subsample': [0.85, 0.95],
    #'gradientboostingclassifier__max_depth': [9, 11]
    #'randomforestclassifier__max_features': ['auto', 0.5],
    #'randomforestclassifier__max_depth': [20, 10],
    #'randomforestclassifier__n_estimators': [200, 250]
    # 'polynomialfeatures__interaction_only': [False, True],
    # 'polynomialfeatures__include_bias': [False, True]
    #'svc__class_weight': ['balanced', None],
    #'svc__kernel': ['linear', 'rbf']
    # 'kneighborsclassifier__n_neighbors': [35, 13],
    # 'kneighborsclassifier__leaf_size': [2, 30]
    #  'kneighborsclassifier__max_samples': [0.75, 0.5]
    #'adaboostclassifier__n_estimators': [500, 200],
    #'adaboostclassifier__learning_rate': [2.0, 1.5]
    'votingclassifier__voting': ['soft', 'hard']
}

rs = RandomizedSearchCV(
    model,
    params,
    n_jobs=-1,
    refit=False,
    random_state=42,
    verbose=1,
    n_iter=4
    )

search = rs.fit(data, y)
search.best_params_
print(search.best_params_)
# #%%

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