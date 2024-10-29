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
from sklearn.ensemble import (StackingClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

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

# Видалення клонок з пропусками більше 30% позитивно врливає на модель. (15% або 50% - вде не так)
columns_to_drop = data.columns[data.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 30%
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)



#%%
# # Видалення ознак з високою кореляцією, знайдені в таблиці кореляцій, а також ознак з високим з'язком (за хі-квадрат (χ²))погіршували результат моделі
# corr_columns = ['Var22', 'Var160', 'Var227', 'Var228', 'Var195', 'Var207', 'Var21'] 
# xi_columns = ['Var212', 'Var216', 'Var197', 'Var199', 'Var206', 'Var210', 'Var192', 'Var193', 'Var203', 
#                    'Var211', 'Var208', 'Var198', 'Var221', 'Var202', 'Var217' , 'Var219', 'Var218', 'Var204', 'Var220', 'Var226'] 
# data = data.drop(columns=corr_columns+xi_columns)                  
# valid = valid.drop(columns=columns_to_drop)

# Видалення рядків, в яких кількість пропусків більше ніж 10 або 5 таж погіршувало результат
# data = data.dropna(thresh=data.shape[1] - 10)

y = data.pop("y")

#%%
# Обробка даних
# Розділяэмо на числові та категоріальні
# Пропуски в числових колонках заповрюємо середнім значенням (виявилося дещо крашим ніж медіаною)
# Пробуски в категоріальних заповнюємо окремою категорією "missing" (трохи краще ніж надавати найбільш пощирене значення)

# - SelectKBest() - Відбір ознак, ознак на основі статистичних тестів (підібрано вибір всіх 67 озгнак).
# - Покращує продуктивність моделі, відсіюючи менш важливі ознаки.

# PowerTransformer():
#   - Трансформує дані для підвищення нормальності розподілу та зменшення впливу викидів.
#   - Використовується для стабілізації дисперсії та покращення ефективності моделі.

# SMOTE -балансування класів (Створює нові зразки менш представленого класу.

# KBinsDiscretizer(encode='onehot-dense', strategy='uniform', random_state=42):
#   - Дискретизує числові ознаки, розділяючи їх на однакові інтервали (кількість бінів).
#   - Розкладає дані на кількісні категорії з подальшим кодуванням (one-hot).

# Спроба мастабування StandardScaler() результат не покращувала

# PCA(n_components=0.96):
#   - Метод зниження розмірності зі збереженням 96%, або 92% дисперсії даних троги пощіршувала модель.
# PolynomialFeatures(degree=2, interaction_only=False, include_bias=False, order='C'):
#   - Генерація поліноміальниї ознак другого ступеня PolynomialFeatures(degree=2) дуже довго відбувалася і бкз користі

# Для навчаня моделі застосовувалися різні алгоритми. Найкраще себе показали  RandomForestClassifier та LogisticRegression, які були включені до фінального ансамблю моделей (RandomForestClassifier двічи), дозволило ще трохі пвдпищити метрику
 
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

    #LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000) # 0.8438
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
    #     RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=3, min_samples_leaf=3,
    #                           max_features=0.5, class_weight='balanced', random_state=42),
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
      
    # XGBClassifier(learning_rate=0.5, max_depth=3, min_child_weight=8, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)   #0.8190
    
    # BernoulliNB(alpha=1.0, fit_prior=False)  # 0.7669

    # VotingClassifier(estimators=[
    #     ('nb', BernoulliNB(alpha=1.0, fit_prior=False)),
    #     ('xgb', XGBClassifier(learning_rate=0.5, max_depth=3, min_child_weight=8, n_estimators=100, random_state=42))
    # ], voting='soft')                

    VotingClassifier(
        estimators=[
                ('logreg', LogisticRegression(solver='newton-cholesky', penalty='l2', class_weight='balanced', max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_split=3, min_samples_leaf=3,
                             max_features=1.0, class_weight='balanced', random_state=42)),
                # ('rf2', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=3, min_samples_leaf=3,
                #              max_features=0.5, class_weight='balanced', random_state=42))  
                ],
        voting='soft')        # Pipe's ballance accuracy on CV: 0.8508 (0.8649 om kaggle)
    #     # 'votingclassifier__rf__n_estimators': 200, 'votingclassifier__rf__max_features': 0.5, 'votingclassifier__rf__max_depth': 10, 'selectkbest__k': 40
)
    
 #%% Пошук найкращих параметрів RandomizedSearch

# model.get_params().keys()

# params = {
#     'selectkbest__k': [67, 40],
#     #'smote__k_neighbors': [15, 13],
#     #'pca__n_components': [0.96, 0.94],
#     #'borderlinesmote__m_neighbors': [5, 3],
#     #'borderlinesmote__k_neighbors': [20, 30],
#     #'logisticregression__max_iter': [2000, 1000],
#     #'logisticregression__solver': ['lbfgs', 'newton-cholesky'],
#     #'logisticregression__class_weight': ['balanced', None],
#     #'gradientboostingclassifier__subsample': [0.85, 0.95],
#     #'gradientboostingclassifier__max_depth': [9, 11]
#     #'randomforestclassifier__max_features': ['auto', 0.5],
#     #'randomforestclassifier__max_depth': [20, 10],
#     #'randomforestclassifier__n_estimators': [200, 250]
#     # 'polynomialfeatures__interaction_only': [False, True],
#     # 'polynomialfeatures__include_bias': [False, True]
#     #'svc__class_weight': ['balanced', None],
#     #'svc__kernel': ['linear', 'rbf']
#     # 'kneighborsclassifier__n_neighbors': [35, 13],
#     # 'kneighborsclassifier__leaf_size': [2, 30]
#     # 'kneighborsclassifier__max_samples': [0.75, 0.5]
#     #'adaboostclassifier__n_estimators': [500, 200],
#     #'adaboostclassifier__learning_rate': [2.0, 1.5]
#     #'votingclassifier__voting': ['soft', 'hard']
#     # 'xgbclassifier__n_estimators': [100, 200]
#     'votingclassifier__rf__max_features': ['auto', 0.5],
#     'votingclassifier__rf__max_depth': [20, 10],
#     'votingclassifier__rf__n_estimators': [200, 250]
# }

##%%
# rs = RandomizedSearchCV(
#     model,
#     params,
#     n_jobs=-1,
#     refit=False,
#     random_state=42,
#     verbose=1,
#     n_iter=4
#     )

# search = rs.fit(data, y)
# search.best_params_
# print(search.best_params_)

# model.set_params(**search.best_params_)

#%% Пошук найкращих параметрів GridSearch

parameters = {
    'smote__k_neighbors': [15, 13],
    'smote__sampling_strategy': [0.8, 0.9],
    #'randomforestclassifier__max_depth': (None, 10), 
    # 'randomforestclassifier__max_features': (0.5, 0.75, 1.0),
    # 'randomforestclassifier__n_estimators': (200, 250),
    # 'randomforestclassifier__min_samples_leaf': (3, 5)
    #'votingclassifier__rf__max_depth': (7, 10), 
    #'votingclassifier__rf__max_features': (0.5, 0.75, 1.0),
    #'votingclassifier__rf__n_estimators': (200, 250),
    #'votingclassifier__rf__min_samples_leaf': (3, 5),
    #'votingclassifier__logreg__solver': ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
    'votingclassifier__logreg__C': [0.01, 0.1, 1, 10]
    }

grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='balanced_accuracy', cv=3, n_jobs=-1)
grid_search.fit(data, y)

print(grid_search.best_params_)
print("GridSearchCV Best Balanced Accuracy Score:", grid_search.best_score_)


model.set_params(**grid_search.best_params_)


 #%%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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