# from pycaret.classification import *

# import warnings
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
# from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, StandardScaler, TargetEncoder
# from sklearn.compose import make_column_transformer, make_column_selector
# from sklearn.feature_selection import SelectKBest
# from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.pipeline import make_pipeline
# from imblearn.over_sampling import SMOTE
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import FunctionTransformer


# #%% 1. Завантаження даних
# print('Завантаження даних')
# # data = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv')
# # valid = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv')
# try:
#     data = pd.read_csv('./datasets/kaggle/final_proj_data.csv')
#     valid = pd.read_csv('./datasets/kaggle/final_proj_test.csv')
# except FileNotFoundError:
#     data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
#     valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')   
# data.info()
# #%%

# # Ініціалізація PyCaret, вказуючи назву колонки з міткою
# clf_setup = setup(data=data, target='your_target_column', session_id=123)

# # Порівняння моделей
# best_model = compare_models()

# # Налаштування та тренування найкращої моделі
# tuned_model = tune_model(best_model)

# # Збереження налаштованої моделі
# save_model(tuned_model, 'best_classification_model')


#%%  TPOT
# conda install -c conda-forge tpot
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
import category_encoders as ce 

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # приклад датасету, замініть на свій

# Завантажте свій датасет
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

#%%
columns_to_drop = valid.columns[valid.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 30%
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')

none_imputer = SimpleImputer(strategy='constant', fill_value='missing').set_output(transform='pandas')
data_cat = none_imputer.fit_transform(data_cat)
valid_cat = none_imputer.transform(valid_cat)

## ADDED 
encoder = ce.TargetEncoder()
# encoder = LabelEncoder()  # not works
data_cat = encoder.fit_transform(data_cat, y)
valid_cat = encoder.transform(valid_cat)

mean_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
data_num = mean_imputer.fit_transform(data_num)
valid_num = mean_imputer.transform(valid_num)  # Використовуємо transform замість fit

data = pd.concat([data_num, data_cat], axis=1)
valid = pd.concat([valid_num, valid_cat], axis=1)

data.info()
#%%


X = data
# Розділіть дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізуйте TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
# Пояснення параметрів:
    # generations — кількість поколінь для генетичного алгоритму (кожне покоління вдосконалює моделі).
    # population_size — кількість моделей у кожному поколінні.
    # verbosity — рівень деталізації виводу (0 — без повідомлень, 1 — тільки прогрес, 2 — повний вивід).
#Після запуску, TPOT збереже код для найкращої моделі в файлі best_model_pipeline.py, який можна використовувати окремо без TPOT.




# Навчайте модель на тренувальних даних
tpot.fit(X_train, y_train)

# Оцініть якість моделі на тестових даних
print("Точність на тестових даних:", tpot.score(X_test, y_test))

# Збережіть найкращий знайдений код для моделі
tpot.export('best_model_pipeline.py')