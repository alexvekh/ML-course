import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler,  PowerTransformer, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import category_encoders as ce
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # Використовуйте Pipeline з imblearn
from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

# pd.set_option("future.no_silent_downcasting", True)
# from sklearn import set_config
# set_config(transform_output="pandas")
## .set_output(transform='pandas')


#%% 1. Завантаження даних
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


# data = data.dropna()                                   # Видаляємо рядки з пропусками буде 5200
data = data.dropna(thresh=data.shape[1] - 5)           # Видаляємо рядки, в яких кількість пропусків більше ніж 10(9080)


# from scipy.stats import zscore
# # 3.1. Проведіть очистку від викидів для колонок 
# features_of_interest = data.select_dtypes(include=np.number).columns

# # Обчислюємо z-score для кожної колонки
# df_zscore = data[features_of_interest].apply(zscore, nan_policy='omit')

# # Видаляємо рядки, де z-score більше 3 або менше -3 (тобто є викидом)
# data = data[(df_zscore < 3).all(axis=1) & (df_zscore > -3).all(axis=1)]
# data[features_of_interest].describe()


# Розділення дпних 
y = data.pop("y")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)


 #%% # Видаляємо ознаки з високою кореляцією.

# # Розрахунок кореляції
# data_num = data.select_dtypes(include=np.number)

# subset = pd.concat([data_num, y], axis=1)
# corr_mtx = subset.corr()
# mask_mtx = np.zeros_like(corr_mtx)
# np.fill_diagonal(mask_mtx, 1)
# fig, ax = plt.subplots(figsize=(28, 24))
# sns.heatmap(subset.corr(),
#             cmap='coolwarm',
#             center=0,
#             annot=True,
#             fmt='.2f',
#             linewidth=0.5,
#             square=True,
#             mask=mask_mtx,
#             ax=ax)
# plt.show()

#%% MI (Mutual Information)
# from sklearn.feature_selection import mutual_info_classif

# # data_cat = data.select_dtypes(include='object')
# # Відновлення пропущених категоріальних значень

# cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
# X_cat = cat_imputer.fit_transform(data_cat)
# encoder = TargetEncoder()
# data_cat_encoded = encoder.fit_transform(X_cat, y)


# # Використання взаємної інформації
# # Обчислення v-значення для кожної пари змінних
# for col1 in data_cat.columns:
#     for col2 in data_cat.columns:
#         if col1 != col2:
#             mi = mutual_info_classif(data_cat_encoded[col1], data_cat_encoded[col2], discrete_features=True)
#             print(f'Mutual Information для {col1} і {col2}: {mi}')
# # Значення варіюються від 0 (немає інформації або залежності) до 1 (максимальна залежність).



# #%% Cramer’s V
# import pandas as pd
# from scipy.stats import chi2_contingency
# import numpy as np

# data_cat = data.select_dtypes(include='object')

# def cramer_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     r, k = confusion_matrix.shape
#     return np.sqrt(chi2 / (n * (min(k-1, r-1))))

# # Обчислення v-значення для кожної пари змінних
# for col1 in data_cat.columns:
#     for col2 in data_cat.columns:
#         if col1 != col2:
#             cramer_v = cramer_v(data_cat[col1], data_cat[col2])
#             print(f'cramer_v-значення для {col1} і {col2}: {cramer_v}')

# # # Cramer’s V — це міра асоціації між двома категоріальними змінними. Сила зв’язку між двома категоріями.
# # # Значення Cramer’s V варіюються від 0 (немає залежності) до 1 (сильна залежність).



    # %% хі-квадрат (χ²)
# Спроба знайти залежність між категоріальними ознаками
# from scipy.stats import chi2_contingency

# data_cat = data.select_dtypes(include='object')
# # Функція для обчислення Chi-Square тесту
# def chi_square_test(col1, col2):
#     confusion_matrix = pd.crosstab(col1, col2)
#     chi2, p, dof, expected = chi2_contingency(confusion_matrix)
#     return p  # p-значення тесту

# # Обчислення p-значення для кожної пари змінних
# for col1 in data_cat.columns:
#     for col2 in data_cat.columns:
#         if col1 != col2:
#             p_value = chi_square_test(data_cat[col1], data_cat[col2])
#             print(f'P-значення для {col1} і {col2}: {p_value}')
# Chi-Square тест показує статистичну значущість залежності, де низьке p-значення (наприклад, менше 0.05) свідчить про наявність залежності між змінними.


#%%
# Дивимось у яких ознак мало унікальних категорій (щоб обробляти OneHotEncoder) 
# X_train.select_dtypes(include='object').apply(lambda x: x.unique()[:10])    # категоріальні ознаки

# %% Pipelines

# cat_transformer = Pipeline(
#     steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent').set_output(transform='pandas')),
#         ('encoder', ce.TargetEncoder().set_output(transform='pandas')) 
#         # ('encoder', ce.OneHotEncoder().set_output(transform='pandas')) 
#         # # encoder = LabelEncoder()  
#     ])


cat_transformer_littleUniq = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent').set_output(transform='pandas')),
        ('encoder', ce.OneHotEncoder().set_output(transform='pandas'))  # OneHotEncoder для невеликої кількості унікальних значень
    ])
cat_transformer_alotUniq = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent').set_output(transform='pandas')),
        ('encoder', ce.TargetEncoder().set_output(transform='pandas'))  # TargetEncoder для великої кількості унікальних значень
    ])
 
 
little_uniq_features =  ['Var196', 'Var205']# 'Var203', , 'Var208', 'Var211', 'Var218'
alot_uniq_features = [col for col in data.select_dtypes(include='object').columns if col not in little_uniq_features]

cat_transformer = ColumnTransformer(
    transformers=[
        ('cat_littleUniq', cat_transformer_littleUniq, little_uniq_features),  # Стовпці з малою кількістю унікальних значень
        # ('cat_alotUniq', cat_transformer_alotUniq, ['Var192', 'Var193', 'Var195', 'Var198', 'Var202', 'Var204', 
        #                                             'Var207', 'Var217', 'Var219', 'Var220', 
        #                                             'Var221', 'Var222', 'Var223', 'Var226', 'Var227', 'Var228'])  # Стовпці з великою кількістю унікальних значень
        ('cat_alotUniq', cat_transformer_alotUniq, alot_uniq_features)  # Стовпці з великою кількістю унікальних значень
    ],
    n_jobs=-1,
    verbose_feature_names_out=False
).set_output(transform='pandas')

num_transformer = Pipeline(
    steps=[
        # ('imputer', SimpleImputer(strategy='mean').set_output(transform='pandas')),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0).set_output(transform='pandas')),
        ('power_transformer', PowerTransformer(method='yeo-johnson').set_output(transform='pandas')),
        ('scaler', StandardScaler().set_output(transform='pandas')),
        
        # ('kbins', KBinsDiscretizer(encode='ordinal').set_output(transform='pandas')),
        # ('kbins_str', FunctionTransformer(lambda x: x.astype(int).astype(str)).set_output(transform='pandas'))
    ])

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


pca = PCA(n_components=0.96)  # Зберігаємо 95% дисперсії

smote = SMOTE(random_state=42)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import numpy as np


# %% Estimator

lnr = (LogisticRegression(penalty='l2', class_weight=None, max_iter=1000)) # 0.713

knn = KNeighborsClassifier(
    n_neighbors=11,          # Кількість сусідів
    weights='distance',      # Стратегія зважування сусідів
    algorithm='auto',       # Алгоритм для пошуку сусідів
    leaf_size=30,           # Розмір листа для алгоритму
    metric='minkowski',     # Метрика для обчислення відстані
    p=2,                    # Параметр для minkowski (1: мангетенська, 2: евклідова)
    metric_params=None,     # Додаткові параметри для метрики
    n_jobs=-1               # Кількість потоків (-1: усі доступні процесори)
)    # 0.707

rf = RandomForestClassifier(
    n_estimators=200,          # кількість дерев
    max_depth=10,            # максимальна глибина дерева
    min_samples_split=3,       # мінімальна кількість зразків для поділу
    min_samples_leaf=3,        # мінімальна кількість зразків у листі
    max_features=0.5,       # кількість ознак для розгляду при поділі
    class_weight='balanced',   # ваги класів для балансування
    random_state=42            # фіксація випадковості
    )      # 0.722

# FILED  
# svc = SVC(class_weight='balanced', kernel='linear', probability=True, random_state=42)





# Будуємо й оцінюємо ансамбль моделей за принципом bagging, використовуючи алгоритм kNN. У нашому ансамблі кожна модель 
# навчатиметься на випадкових підмножинах, що складаються з 75% об'єктів та ознак тренувальної вибірки.
bgg = (BaggingClassifier(
     KNeighborsClassifier(),
     max_samples=0.75,
     max_features=0.75,
     # n_jobs=-1,
     random_state=42))


# # Побудова AdaBoostClassifier. Будуємо й оцінюємо ансамбль моделей для класифікації за методом Ada Boost:
ada = (AdaBoostClassifier(algorithm='SAMME', random_state=42))

       

# # Побудова GradientBoostingClassifier. гіперпараметри subsample=0.75 і max_features='sqrt':
#     #- subsample: частка спостережень, на якій будуть навчатися базові моделі. 
#         # Якщо частка менше 1.0, то алгоритм починає працювати за методом Stohastic Gradient Boost; 
#     #- max_features: кількість ознак, які слід враховувати для пошуку найкращого розбиття при навчанні дерева рішень (базової моделі).
# # Визначення параметрів 0.0 < subsample < 1.0 і max_features < n_features приводить до зменшення дисперсії прогнозів (low variance), 
# # але потенційно може збільшити похибку моделі (high bias):

gb = (GradientBoostingClassifier(
     learning_rate=0.3,
     subsample=0.75,
     max_features='sqrt',
     random_state=42))

    

# ## %%
# # Побудова StackingClassifier (за принципом stacking). Для цього ми використовуємо той самий набір базових моделей, 
# # але додатково створюємо над ними метамодель типу GradientBoostingClassifier. (метамодель у нашому ансамблі сама є ансамблем моделей!)

gbc = GradientBoostingClassifier(
    subsample=0.75,
    max_features='sqrt',
    random_state=42)

estimators = [('lnr', lnr),
              ('knn', knn),
              ('rfc', rf)]

stc = StackingClassifier(
    estimators=estimators,
    final_estimator=gbc)




# ## %% ----  All the 20 fits failed. # 0.744
# clf1 = (LogisticRegression(penalty='l2', class_weight='balanced')) # 0.713

# clf2 = KNeighborsClassifier(
#     n_neighbors=35,          # Кількість сусідів
#     weights='distance',      # Стратегія зважування сусідів
#     algorithm='auto',       # Алгоритм для пошуку сусідів
#     leaf_size=30,           # Розмір листа для алгоритму
#     metric='minkowski',     # Метрика для обчислення відстані
#     p=2,                    # Параметр для minkowski (1: мангетенська, 2: евклідова)
#     metric_params=None,     # Додаткові параметри для метрики
#     n_jobs=-1               # Кількість потоків (-1: усі доступні процесори)
# ) 

# clf3 = RandomForestClassifier(
#     n_estimators=500,          # кількість дерев
#     max_depth=10,              # максимальна глибина дерева
#     min_samples_split=5,       # мінімальна кількість зразків для поділу
#     min_samples_leaf=3,        # мінімальна кількість зразків у листі
#     max_features='sqrt',       # кількість ознак для розгляду при поділі
#     class_weight='balanced',   # ваги класів для балансування
#     random_state=42            # фіксація випадковості
#     )
# ## Побудова VotingClassifier (за принципом soft voting, використовуючи три моделі різних типів 
# # # з їхніми базовими налаштуваннями в пакеті sklearn (LogisticRegression, KNeighborsClassifier, GaussianNB)):

clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = GaussianNB()

estimators = [('lnr', lnr),
              ('knn', knn),
              ('rfc', rf)]

vot = VotingClassifier(
    estimators=estimators,
    voting='soft')#.fit(X_res, y_res)  


clf_estimator = lnr


# %% Main pipeline
clf_pipe_model = (Pipeline(
    steps=[
        ('col_processor', col_processor),
        # ('encoder', ce.TargetEncoder().set_output(transform='pandas')),
        # ('scaler', StandardScaler().set_output(transform='pandas')),
        # ('outlier_removal', outlier_remover),  # Очищення від викидів
        ('smote', smote),  # Балансування класів
        #('pca', pca),  # Можна вимкнути PCA, якщо не потрібно
        ('clf_estimator', clf_estimator)
    ]))

#%% Fit 80%
clf_model = clf_pipe_model.fit(X_train, y_train)

# %% Predict(X-test)

pred_pipe = clf_model.predict(X_test)

print(confusion_matrix(y_test, pred_pipe)) 
print(f"Pipe's accuracy is: {accuracy_score(y_test, pred_pipe):.1%}")


## %%
# X_train_transformed = clf_pipe_model[:-1].transform(X_train)
## %%
# pet_transformed1 = clf_pipe_model[:5].transform(pet)

# transformed_data = col_processor.fit_transform(X_train, y_train)
# print(transformed_data.head())

# %% Об'єднання даних та кросвалідація

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)


cv_results = cross_val_score(
    estimator=clf_pipe_model,
    X=X,
    y=y,
    scoring='balanced_accuracy',
    cv=5,
    verbose=1)

acc_cv = cv_results.mean()

print(f"Pipe's ballance accuracy on CV: {acc_cv:.1%}")

# %% Grid Search best parameters

parameters = {
    #Voting
    # 'clf_estimator__voting': ('hard', 'soft'),
    # 'clf_estimator__weights': ('float', 'int'),
    #NearestNeighbors
    # 'clf_estimator__n_neighbors': (5, 17),
    # 'clf_estimator__radius': (1.0, 3.0),
    # 'clf_estimator__algorithm': ('ball_tree', 'kd_tree')
    # RandomForest
    # 'clf_estimator__n_neighbors': (5, 17),
    # 'clf_estimator__weights': ('uniform', 'distance'),
    # 'clf_estimator__algorithm': ('auto', 'brute')
    # RandomForest
    #'clf_estimator__max_depth': (None, 10), 
    #'clf_estimator__max_features': (0.5, 0.75)
    #LogisticRegression
    'clf_estimator__solver': ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
    }

search = (GridSearchCV(
    estimator=clf_pipe_model,
    param_grid=parameters,
    scoring='balanced_accuracy',
    # scoring='accuracy',
    cv=5,
    refit=False)
    .fit(X, y))

print(f"Best parameters: {search.best_params_}")
print(f"Best balanced accuracy: {search.best_score_:.4f}")

# %% Set best parameters
parameters_best = search.best_params_
clf_pipe_model = clf_pipe_model.set_params(**parameters_best)

#%% Fit 100%

model_upd = clf_pipe_model.fit(X, y)

# %%  Кросвалідація з best parameters

cv_results_upd = cross_val_score(
    estimator=model_upd,
    X=X,
    y=y,
    scoring='balanced_accuracy',
    # scoring='accuracy',
    cv=5)

acc_cv_upd = cv_results_upd.mean()
print(f"Pipe's UPDATED accuracy on CV: {acc_cv_upd:.1%}")

## %%
# clf_pipe_model.named_steps['lod_transformer']

#%% Predict(valid) to CSV

output = pd.DataFrame({'index': valid.index,
                       'y': model_upd.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)