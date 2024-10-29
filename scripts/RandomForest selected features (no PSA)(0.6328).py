# Метод k-means. Ієрархічний кластерний аналіз

# Встановимо необхідні бібліотеки у створене раніше середовище conda:
# pip install yellowbrick
# conda install kneed

import warnings
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering 
import category_encoders as ce 
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier

 
# %%
# 1. Завантажте набір даних.
data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')
# data = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv')
# valid = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv')
data.info()

# %%
# data.isna().mean().sort_values(ascending=False)           # процент пропусків за кожною колонкою.
# data.isna().mean(axis=1).sort_values(ascending=False)     # процент пропусків за кожним рядком.
# valid.isna().sum(axis=1)                                  # кількість пропусків за кожним рядком.
# valid.isna().mean(axis=1).sort_values(ascending=False)    # процент пропусків за кожним рядком.
# data.isna().sum(axis=1)                                   # кількість пропусків за кожним рядком.


columns_to_drop = data.columns[data.isna().mean() > 0.3]    # Створюємо список колонок, у яких частка пропусків більше 10%
data = data.drop(columns=columns_to_drop)                   # Видаляємо ці колонки з датафреймів
valid = valid.drop(columns=columns_to_drop)

# data = data.dropna()                                   # Видаляємо рядки з пропусками # 5200
data = data.dropna(thresh=data.shape[1] - 10)           # Видаляємо рядки, в яких кількість пропусків більше ніж 5(9080)

# data.dtypes

# Відсоткове співвідношення 1 та 0 цільової мітки
# percentages = data['y'].value_counts(normalize=True) * 100
# print(percentages)


# %%
# Розподіли ознак

# melted = data.melt(var_name='feature', value_name='value')
# melted['y'] = y.repeat(data.shape[1]).values

# g = sns.FacetGrid(melted,
#                   col='feature',
#                   col_wrap=4,
#                   sharex=False,
#                   sharey=False,
#                   aspect=1.25)

# g.map(sns.histplot, 'value')

# g.set_titles(col_template='{col_name}')

# g.tight_layout()

# plt.show()
# %%
y = data.pop("y")

# %%
# Розбиваємо на числові та категоріальні підмножини:
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')


#%%
# Відновлення пропущених числових значень
num_imputer = SimpleImputer().set_output(transform='pandas')
data_num = num_imputer.fit_transform(data_num)
valid_num = num_imputer.transform(valid_num)  # Використовуємо transform замість fit

# Відновлення пропущених категоріальних значень
cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
data_cat = cat_imputer.fit_transform(data_cat)
valid_cat = cat_imputer.transform(valid_cat)  # Використовуємо transform замість fit

# %%
# Нормалізація числових ознак за допомогою об'єкта StandardScaler або PowerTransformer з пакету sklearn
# scaler = StandardScaler().set_output(transform='pandas')
scaler = PowerTransformer().set_output(transform='pandas')
X_train_num = scaler.fit_transform(data_num)
valid_num = scaler.transform(valid_num)

# Кодування категоріальних ознак (наприклад, за допомогою об’єктів OneHotEncoder / TargetEncoder з пакета category_encoders).
# encoder = ce.OneHotEncoder()
encoder = ce.TargetEncoder()

X_train_cat = encoder.fit_transform(data_cat, y)
valid_cat = encoder.transform(valid_cat)

# %%

# 7. Об'єднання підмножини з числовими і категоріальними ознаками (після кодування) в одну:
X = pd.concat([X_train_num, X_train_cat], axis=1)
valid = pd.concat([valid_num, valid_cat], axis=1)

#%%
from imblearn.over_sampling import SMOTE
# Застосування SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


# %%
# Спроба знайти залежність між категоріальними ознаками
# from scipy.stats import chi2_contingency

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

# subset = pd.concat([X, y], axis=1)

# corr_mtx = subset.corr()

# mask_mtx = np.zeros_like(corr_mtx)
# np.fill_diagonal(mask_mtx, 1)

# fig, ax = plt.subplots(figsize=(7, 6))

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




#%%
# Вибір ознак на основі важливості
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Оцінка важливості ознак
feature_importances = rf.feature_importances_

# Вибір ознак на основі важливості
threshold = 0.05  # Ви можете вибрати інший поріг
selected_features = X.columns[feature_importances > threshold]
X_selected = X[selected_features]



#%%
# # 3.1. Проведіть очистку від викидів для колонок 
# # features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']

# # Обчислюємо z-score для кожної колонки
# df_zscore = data[features_of_interest].apply(zscore, nan_policy='omit')

# # Видаляємо рядки, де z-score більше 3 або менше -3 (тобто є викидом)
# data = data[(df_zscore < 3).all(axis=1) & (df_zscore > -3).all(axis=1)]

# data[features_of_interest].describe()

#%%
#%% 
# PSA
# розрахунок головних компонент
pca = PCA(random_state=42).fit(X)
pve = pca.explained_variance_ratio_

#%%
# Пошук оптимальноъ кількості головних компонетн за "правилом ліктя” на графіку дисперсії (змінності) даних
sns.set_theme()

kneedle = KneeLocator(
    x=range(1, len(pve) + 1),
    y=pve,
    curve='convex',
    direction='decreasing')

kneedle.plot_knee()

plt.show()

#%% 

# Визначимо й візуалізуємо на графіку кумулятивну частку змінності даних, яку пояснюють 9 перших головних компонент:
n_components = kneedle.elbow

ax = sns.lineplot(np.cumsum(pve))

ax.axvline(x=n_components,
           c='black',
           linestyle='--',
           linewidth=0.75)

ax.axhline(y=np.cumsum(pve)[n_components],
           c='black',
           linestyle='--',
           linewidth=0.75)

ax.set(xlabel='number of components',
       ylabel='cumulative explained variance')

plt.show()

# %%
# Отже, після зменшення розмірності даних ми зберігаємо приблизно 97% “інформативності” вхідного. (дуже високий показник)

# Зменшуємо розмірність даних за допомогою PCA:
X = pca.transform(X)[:, :n_components]
valid = pca.transform(valid)[:, :n_components]

#  X = pca.transform(X)[:, :4]


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42)

#%%
#    RandomForest

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train , y_train)


# Отримання важливості ознак
# feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
# print(feature_importances)

#y_pred = rf.predict(X_test)

#%%
y_pred = pd.Series(data=rf.predict(X_test), index=X_test.index)

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Розрахунок метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import balanced_accuracy_score

# Розрахунок збалансованої точності
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced Accuracy: {balanced_accuracy:.2%}')

print(f'Accuracy: {accuracy:.2%}')
print(f'Precision: {precision:.2%}')
print(f'Recall: {recall:.2%}')
print(f'F1-score: {f1:.2%}')

# Побудова матриці похибок
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# print("y_pred:", y_pred.values[:30])
# print("y_test:", y_test.values[:30])



# %%

#    RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X , y)

# Отримання важливості ознак
# feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
# print(feature_importances)

# y_pred = rf2.predict(X_test)

#%%

output = pd.DataFrame({'index': valid.index,
                       'y': rf.predict(valid)})

output.to_csv('final_proj_sample_submission.csv', index=False)







#%%













# Навчаємо модель і отримуємо прогнози:
  # class_weight='balanced' - невелюємо дисбаланс між позитивним і негативним класами (Ваги обчислюються автоматично під час тренування моделі.)
  # solver='lbfgs', liblinear, newton-cg, newton-cholesky, sag, saga (майже не впливають на результат)
from sklearn.neighbors import KNeighborsRegressor
knn = (KNeighborsRegressor(n_neighbors=20, 
                             weights='uniform', 
                             algorithm='auto', 
                             leaf_size=12, 
                             p=1, 
                             metric='minkowski', 
                             metric_params=None, 
                             n_jobs=None)
       .fit(X_train, y_train))

y_pred = knn.predict(X_test)

# %%

# # Обробка VALID dataset
# y_true = valid.pop("Salary")

# # # Зміна типу на int
# # valid['Role'] = valid['Role'].map(role_mapping).astype(int)
# # valid['Qualification'] = valid['Qualification'].map(qualification_mapping).astype(int)
# # valid['University'] = valid['University'].map(u_mapping).astype(int)

# # 3.2.Розбиваємо датасет на підмножини
# valid_num = valid.select_dtypes(include=np.number)
# valid_cat = valid.select_dtypes(include='object')

# # Нормалізація числових, кодування категоріальних
# X_test_num = scaler.transform(valid_num)

# valid_cat = valid_cat.drop(['Phone_Number', "Name", "Date_Of_Birth", "Qualification"], axis=1)
# X_test_cat = encoder.transform(valid_cat)

# # 7. Об'єднання підмножини
# X_test = pd.concat([X_test_num, X_test_cat], axis=1)
# X_train.shape


# %%
# Результат

#y_pred = knn.predict(X_test)

y_pred = pd.Series(knn.predict(X_test), index=X_test.index, dtype='int64')
#%%
threshold = 0.5
y_pred = (y_pred >= threshold).astype(int)
y_pred = pd.Series(y_pred, index=X_test.index).astype('int64')


#%%
# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")



#%%

# threshold = 0.5
# y_pred = (y_pred >= threshold).astype(int)
# y_pred = pd.Series(y_pred)



#%% 






#%%
# 3. Нормалізуйте набір даних за допомогою об’єкта StandardScaler з пакета sklearn для подальшої кластеризації.
X = StandardScaler().fit_transform(concrete)

#%% 
# 4. Визначте оптимальну кількість кластерів за допомогою об'єкта KElbowVisualizer з пакета yellowbrick.

# 4a). розрахунок головних компонент
pca = PCA(random_state=42).fit(X)
pve = pca.explained_variance_ratio_


#%%
# 4b). Пошук оптимальноъ кількості головних компонетн за "правилом ліктя” на графіку дисперсії (змінності) даних

sns.set_theme()

kneedle = KneeLocator(
    x=range(1, len(pve) + 1),
    y=pve,
    curve='convex',
    direction='decreasing')

kneedle.plot_knee()

plt.show()

# Визначена візуально за “правилом ліктя” оптимальна кількість компонент дорівнює 6.
# Проте цікаво спробувати й 4, оскільки це ближче до правила 1:10 (4:427) 


#%% 

# 4c). Визначимо й візуалізуємо на графіку кумулятивну частку змінності даних, яку пояснюють 9 перших головних компонент:
n_components = kneedle.elbow

ax = sns.lineplot(np.cumsum(pve))

ax.axvline(x=n_components,
           c='black',
           linestyle='--',
           linewidth=0.75)

ax.axhline(y=np.cumsum(pve)[n_components],
           c='black',
           linestyle='--',
           linewidth=0.75)

ax.set(xlabel='number of components',
       ylabel='cumulative explained variance')

plt.show()

# %%

# Отже, після зменшення розмірності даних ми зберігаємо приблизно 97% “інформативності” вхідного. (дуже високий показник)
# У випадку 4 - 92%


# 4d). Зменшуємо розмірність даних за допомогою PCA:
X = pca.transform(X_train)[:, :n_components]

#  X = pca.transform(X)[:, :4]



# %%
# 4e). Кластеризація набору даних. Визначення кількості кластерів за допомогою KMeans

# Далі виконаємо кілька варіантів кластеризації, використовуючи трансформований набір даних і задаючи різну кількість кластерів для пошуку.
# На кожному етапі обчислимо сумарну міру щільності та побудуємо графік її залежності від заданої кількості кластерів. 
# Потім за допомогою "правила ліктя" визначимо оптимальну кількість кластерів. (за допомогою функціоналу пакета yellowbrick)

model_kmn = KMeans(random_state=42)

visualizer = KElbowVisualizer(
    model_kmn,
    k=(2, 10),
    timings=False)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    visualizer.fit(X)

visualizer.show()
# Визначена оптимальна кількість кластерів для цього набору даних за “правилом ліктя” дорівнює 5.


#%%

# 5. Проведіть кластеризацію методом k-середніх і отримайте мітки для кількості кластерів, визначеної на попередньому кроці.

k_best = visualizer.elbow_value_
# k_best = 3
model_kmn = KMeans(n_clusters=k_best, random_state=42).fit(X2)

labels_kmn = pd.Series(model_kmn.labels_, name='k-means')


#%% 


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Зменшуємо кількість вимірів до 2 за допомогою PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X2)

# Візуалізуємо кластери
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmn, cmap='viridis', s=50)
plt.title(f'Visualization of {k_best} clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster label')
plt.show()

#%% 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Зменшуємо кількість вимірів до 3 за допомогою PCA
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X)
X_pca = X
# Створюємо 3D-графік
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Візуалізуємо кластери
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                     c=labels_kmn, cmap='viridis', s=50)

# Додаємо заголовок та підписи осей
ax.set_title(f'3D Visualization of {k_best} Clusters')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Додаємо кольорову шкалу
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Cluster label')

plt.show()

#%%
concrete.mean()

#%%

# 6. Використайте оригінальний набір вхідних даних для розрахунку описової статистики кластерів («звіту»): 
    # розрахуйте медіани для кожної ознаки, включаючи підрахунок кількості компонент по кожному кластеру за допомогою методу groupby.

data = concrete.copy()
data['Cluster'] = labels_kmn

cluster_medians = data.groupby('Cluster').median()
print("Cluster medians:")
print(cluster_medians)

cluster_counts = data.groupby('Cluster').size().rename('Count')
print("Cluster counts:")
print(cluster_counts)


report = pd.concat([cluster_medians, cluster_counts], axis=1)

report

#%%

# 7. Додайте до звіту кількість об'єктів (рецептур) у кожному з кластерів.

report.columns = [f'Median {col}' for col in report.columns[:-1]] + ['Count']

#%%

report.head()

#%%

# melted = report.melt()
report_reset = report.reset_index()

# Melt the DataFrame to convert it into a long format
melted = report_reset.melt(id_vars=['Cluster'])

g = sns.FacetGrid(melted, 
                  col='variable', 
                  col_wrap=3, 
                  sharex=False, 
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    g.map(sns.barplot, 'Cluster', 'value')

# Set the titles and layout
g.set_titles(col_template='{col_name}')
g.tight_layout()

plt.show()

