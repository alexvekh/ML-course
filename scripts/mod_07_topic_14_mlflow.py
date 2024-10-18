# Практика використання MLFlow Tracking. Огляд структури файлу
# Відкритий фреймворк MLflow дозволяє ефективно керувати життєвим циклом моделей машинного навчання, 
# починаючи з їх розробки й закінчуючи впровадженням в експлуатацію.
# здійснює збереження моделі та логування параметрів на вимогу користувача, прописану безпосередньо у скрипті, 
# який відповідає за побудову/навчання моделі.
# Під час виконання MLflow Tracking записує в “журнал” створеного користувачем "експерименту" визначені ним метадані (метрики, параметри, 
# час виконання тощо), а також “артефакти” (наприклад, моделі, зразки вхідних даних тощо), щодо яких необхідно вести моніторинг.
# MLflow Tracking підтримує багато сценаріїв розгортання для різних розробницьких процесів. 

# розглянемо базовий сценарій, який передбачає реєстрацію даних експерименту та моделей у локальні файли. 
# pip install mlflow

import warnings
from tempfile import mkdtemp
import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# %%

# Завантаження й логування базового ML-конвеєра

# Створимо новий “експеримент”, у рамках якого будемо вести журнал за допомогою MLFlow:
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
mlflow.set_experiment('MLflow Tracking')

# %%

# Завантажимо набір даних, попередню очистку якого ми виконали в темі 13“Вступ до побудови ML-конвеєрів. Крос-валідація”.
data = pd.read_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')

X, y = (data.drop(['Item_Identifier',
                   'Item_Outlet_Sales'],
                  axis=1),
        data['Item_Outlet_Sales'])

# %%
# Далі завантажимо розроблений нами ML-конвеєр для прогнозування продажів для цього набору даних. 
# Конвеєр був збережений до навчання, тому ми не зможемо відразу його використовувати як підготовлену модель для отримання прогнозів.
with open('../models/mod_07_topic_13_mlpipe.joblib', 'rb') as fl:
    pipe_base = joblib.load(fl)

# %%

# Тут проведемо крос-валідацію базової моделі (RandomForestRegressor) в конвеєрі і збережемо відповідну метрику, як в темі 13.
# Цього разу крос-валідація виконуватиметься на всьому наборі даних, а не лише на тренувальній вибірці (80%), тому метрика може відрізнятися.
cv_results = cross_val_score(
    estimator=pipe_base,
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)

rmse_cv = np.abs(cv_results).mean()

# %%
# Далі виконаємо навчання базового ML-конвеєра:
model_base = pipe_base.fit(X, y)

# %%
# Збережемо в окрему змінну параметри навченої моделі (у складі ML-конвеєра), які на наступному кроці запишемо в журнал “експерименту” за допомогою MLFlow:
params_base = pipe_base.named_steps['reg_estimator'].get_params()

# %%

# Цей блок коду використовується для запису метаданих у журнал створеного нами “експерименту” і збереження навченої моделі.
# У журнал будуть внесені гіперпараметри моделі, її метрика на крос-валідації, інформація про структуру вхідних даних, 
# які очікує конвеєр (sіgnature), а також сама модель у форматі, який дозволяє її безпосереднє використання для отримання прогнозів.

# Start an MLflow run
with mlflow.start_run(run_name='rfr'):
    # Log the hyperparameters
    mlflow.log_params(params_base)
    # Log the loss metric
    mlflow.log_metric('cv_rmse_score', rmse_cv)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag('Model', 'RandomForest for BigMart')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Infer the model signature
        signature = mlflow.models.infer_signature(
            X.head(),
            model_base.predict(X.head()))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model_base,
        artifact_path='model_base',
        signature=signature,
        input_example=X.head(),
        registered_model_name='model_base_tracking')

# %%

# Модифікація й підбір гіперпараметрів ML-конвеєра

# Далі ми на основі базового ML-конвеєра створимо аналогічний, але цього разу встановимо останнім кроком інший оцінювач із бібліотеки sklearn, 
# а саме GradientBoostingRegressor.
pipe_upd = Pipeline(
    steps=pipe_base.steps[:-1] +
    [('reg_model',
      GradientBoostingRegressor(random_state=42))],
    memory=mkdtemp())

# %%
# Ефективність цього алгоритму суттєво залежить від правильно підібраної комбінації гіперпараметрів.
# Оптимізація гіперпараметрів — це завдання, яке полягає у виборі найкращих значень гіперпараметрів для алгоритму машинного навчання. 
# Гіперпараметри визначають процес навчання моделі й задаються користувачем, модель їх не “вивчає” в ході тренування. 
# Різні моделі мають власні гіперпараметри, які залежать від типу задач, які розв’язує модель, математичних/статистичних методів, 
# які вона використовує для навчання, а також конкретних варіантів реалізації моделі в різних пакетах Python.

# Традиційний метод підбору гіперпараметрів — це так званий “пошук по сітці”, який передбачає повний перебір заданого набору гіперпараметрів 
# із переліку доступних для налаштування. Під час пошуку оцінюється ефективність моделі за визначеною метрикою для кожної комбінації гіперпараметрів 
# з використанням крос-валідації або на тестовому наборі.

# Замість повного перебору всіх комбінацій гіперпараметрів можна виконати їх випадковий вибір для навчання й оцінки моделі. 
# “Випадковий пошук” в окремих випадках може перевершити “пошук по сітці”, особливо, якщо лише невелика кількість гіперпараметрів 
# суттєво впливає на ефективність алгоритму.

# Підбір гіперпараметрів у пакеті sklearn можна ефективно здійснити за допомогою об'єктів GridSearchCV (”пошук по сітці”) 
# та RаndomizedSearchCV (”випадковий пошук”).

# Задамо набір гіперпараметрів для пошуку по сітці й виконаємо їх оптимізацію для моделі в модифікованому ML-конвеєрі:
parameters = {
    'reg_model__learning_rate': (0.1, 0.3),
    'reg_model__subsample': (0.75, 0.85),
    'reg_model__max_features': ('sqrt', 'log2')}

search = (GridSearchCV(
    estimator=pipe_upd,
    param_grid=parameters,
    scoring='neg_root_mean_squared_error',
    cv=5,
    refit=False)
    .fit(X, y))

# %%
# Далі виконаємо навчання моделі з оптимальними гіперпараметрами:
parameters_best = search.best_params_
pipe_upd = pipe_upd.set_params(**parameters_best)

model_upd = pipe_upd.fit(X, y)

# %%

cv_results_upd = cross_val_score(
    estimator=pipe_upd,
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)

rmse_cv_upd = np.abs(cv_results_upd).mean()

# %%
# Логування модифікованого ML-конвеєра
# Зробимо відповідний запис у “журнал” поточного експерименту для модифікованої версії ML-конвеєра, як для його базової версії:
with mlflow.start_run(run_name='gbr'):
    mlflow.log_params(pipe_upd.named_steps['reg_model'].get_params())
    mlflow.log_metric('cv_rmse_score', rmse_cv_upd)
    mlflow.set_tag('Model', 'GradientBoosting model for BigMart')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        signature = mlflow.models.infer_signature(
            X.head(),
            model_upd.predict(X.head()))

    model_info = mlflow.sklearn.log_model(
        sk_model=model_upd,
        artifact_path='model_upd',
        signature=signature,
        input_example=X.head(),
        registered_model_name='model_upd_tracking')

# %%
#Тепер локально запустимо сервіс MLFlow Tracking, завдяки якому будемо здійснювати моніторинг.
# Для цього в терміналі виконаємо навігацію до папки проєкту й запустимо команду: mlflow server --host 127.0.0.1 --port 8080
# Паралельно в іншому вікні термінала запускаємо скрипт mod_07_topic_14_mlflow.py
# Після завершення скрипта можемо відкрити у браузері адресу http://127.0.0.1:8080/ і побачити журнал створеного нами “експерименту”. 
# В інтерактивному режимі тут доступна інформація за кожною з моделей, відповідні гіперпараметри, метрики, 
# документація щодо запуску моделей для отримання прогнозів тощо.

# верніть увагу, як зменшилась помилка на крос-валідації після заміни моделі в ML-конвеєрі та його навчання з оптимізованими гіперпараметрами 
# (1085.78 проти 1114.43).


#%%

# Ми можемо отримувати інформацію від MLFlow Tracking не лише через графічний інтерфейс користувача, але й за допомогою програмних засобів.

# best_run = (mlflow
#             .search_runs(
#                 experiment_names=['MLflow Tracking'],
#                 order_by=['metrics.cv_rmse_score'],
#                 max_results=1))

# best_run[['tags.Model', 'metrics.cv_rmse_score']]
