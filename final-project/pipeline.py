import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import set_config
set_config(transform_output="pandas")
# .set_output(transfor='pandas')
#%% 1. Завантаження даних
data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')
# data = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv')
# valid = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv')
data.info()

#%%

y = data.pop("y")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

#%% Розбиваємо на числові та категоріальні підмножини:
X_train_num = X_train.select_dtypes(include=np.number)
X_train_cat = X_train.select_dtypes(include='object')
X_test_num = X_test.select_dtypes(include=np.number)
X_test_cat = X_test.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')


#%% 2. Визначаємо типи ознак
num_features = data.select_dtypes(include=[np.number]).columns
cat_features = data.select_dtypes(include=[object]).columns




#%% 3. Формування трансформерів числових і категоріальних ознак
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Заповнюємо пропуски середнім
    ('scaler', StandardScaler())  # Нормалізуємо ознаки
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заповнюємо пропуски найпоширенішими значеннями
    ('encoder', TargetEncoder())  # Target Encoding для категоріальних ознак
])

# 4. Об'єднання всіх трансформерів у ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 5. Балансування класів за допомогою SMOTE 

# 6. Модель RandomForestClassifier
random_forest_classifier = RandomForestClassifier(random_state=42)

# 7. Зменшення розмірності (PCA) – за бажанням, якщо даних дуже багато
pca = PCA(n_components=0.95)  # Зберігаємо 95% дисперсії

# 8. Створення ML-конвеєра
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),  # Можна вимкнути PCA, якщо не потрібно
    ('smote', SMOTE(random_state=42)),  # Балансування класів
    ('classifier', random_forest_classifier)
])

#%% 9. Поділ даних на тренувальний і валідаційний набори
X = data.drop(columns=['y'])  # Замініть 'target' на ім'я цільової змінної
y = data['y']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#%% 10. Підбір гіперпараметрів з GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],.
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'pca__n_components': [0.90, 0.95, 0.99]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#%% Навчаємо модель на тренувальному наборі
grid_search.fit(X_train, y_train)

#%% Оцінка на валідаційному наборі
y_pred = grid_search.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f'Accuracy on validation set: {accuracy:.4f}')

#%% 11. Пере-навчання моделі на всіх даних (тренувальна + тестова вибірки)
X_full = pd.concat([X_train, X_valid])
y_full = pd.concat([y_train, y_valid])
final_model = grid_search.best_estimator_.fit(X_full, y_full)

#%% 12. Прогноз для валідаційного набору і формування .csv файлу
# X_test = pd.read_csv('test_data.csv')  # Завантаження тестових даних
# y_test_pred = final_model.predict(X_test)

#%% Формування .csv файлу для результатів
output = pd.DataFrame({'index': valid.index,
                       'y': final_model.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)