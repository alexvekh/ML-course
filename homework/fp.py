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

# %%
# 1. Завантажте набір даних.
data = pd.read_csv('../datasets/kaggle/final_proj_data.csv')
valid = pd.read_csv('../datasets/kaggle/final_proj_test.csv')
# data = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv')
# valid = pd.read_csv('/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv')

data.info()

# %%
# процент пропусків за кожною колонкою.
data.isna().mean().sort_values(ascending=False)

# %%
# 3.1. Видаляємо із набору ознаки з великою кількістю пропущених значень.
# data2 = data[data.columns[data.isna().mean().lt(0.3)]]

# %%
# кількість пропусків за кожним рядком.
data2.isna().sum(axis=1)
#%%
# Видаляємо рядки з пропусками
data2 = data2.dropna()
# data2 = data2[data.columns[data.isna().mean().lt(0.01)]]
data2.isna().mean().sort_values(ascending=False)
# або
# Видаляємо рядки, в яких кількість пропусків більше ніж 8
# data_cleaned = data.dropna(thresh=data.shape[1] - 8)
# %%
data2.dtypes

#%%
# Відсоткове співвідношення 1 та 0 цільової мітки
percentages = data2['y'].value_counts(normalize=True) * 100
print(percentages)


#%%
# Візуалізація за допомогою парних графіків
sns.pairplot(data2, hue='y')
plt.show()
 

# %%
y = data2.pop("y")

# %%
# Розбиваємо на числові та категоріальні підмножини:
data_num = data2.select_dtypes(include=np.number)
data_cat = data2.select_dtypes(include='object')



# %%

from imblearn.combine import SMOTEENN
# Балансування даних
smote_enn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smote_enn.fit_resample(data2, y)




#%%
# ----Зміна категоріальних ознак на числові покращення результату не дало 
# # Зміна 'Role' на int
# role_mapping = {'Junior': 1,'Mid': 2,'Senior': 3}
# data['Role'] = data['Role'].map(role_mapping).astype(int)

# # Зміна 'Qualification' на int
# qualification_mapping = {'Bsc': 1, 'Msc': 2, 'PhD': 3}
# data['Qualification'] = data['Qualification'].map(qualification_mapping).astype(int)

# # Зміна 'University' на int
# u_mapping = {'Tier1': 1, 'Tier2': 2, 'Tier3': 3}
# data['University'] = data['University'].map(u_mapping).astype(int)

# %%
# Розподіли ознак

# melted = data3.melt()

# g = sns.FacetGrid(melted,
#                   col='variable',
#                   col_wrap=4,
#                   sharex=False,
#                   sharey=False,
#                   aspect=1.25)

# g.map(sns.histplot, 'value')

# g.set_titles(col_template='{col_name}')

# g.tight_layout()

# plt.show()


# %%
# ----Відновлення пропущених значень не потрібно, оскільки ми їх видалили всі
# Відновлення пропущених числових значень
# num_imputer = SimpleImputer().set_output(transform='pandas')
# data_num = num_imputer.fit_transform(data_num)

# Відновлення пропущених категоріальних значень:
# cat_imputer = SimpleImputer(
#     strategy='most_frequent').set_output(transform='pandas')
# data_cat = cat_imputer.fit_transform(data_cat)


#%%
# ----Отримання коду з номеру телефону, який міг би вказати про регіон проживання не допомогло (майже всі різні)
# data_cat['Phone_Code'] = data_cat['Phone_Number'].apply(lambda x: x.split('-')[0])
#valid_cat = valid_cat.drop('Phone_Number'], axis=1)

# %%
# Нормалізація числових ознак за допомогою об'єкта StandardScaler або PowerTransformer з пакету sklearn
# scaler = StandardScaler().set_output(transform='pandas')
scaler = PowerTransformer().set_output(transform='pandas')
X_train_num = scaler.fit_transform(data_num)

#%% 
correlation = data_num.corr()y.sort_values(ascending=False)
print(correlation)
#%%
# наліз на основі графіків розподілу
import seaborn as sns
import matplotlib.pyplot as plt

for column in data_cat.columns:
    sns.boxplot(x='y', y=column, data=data)
    plt.title(f'Boxplot of {column} vs y')
    plt.show()
#%%

# 4. Використання дерев рішень для оцінки важливості ознак
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X , y)

# Отримання важливості ознак
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

#%%
# 3.1. Проведіть очистку від викидів для колонок 
# AveRooms, AveBedrms, AveOccup та Population

features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']


# Обчислюємо z-score для кожної колонки
df_zscore = data[features_of_interest].apply(zscore, nan_policy='omit')

# Видаляємо рядки, де z-score більше 3 або менше -3 (тобто є викидом)
data = data[(df_zscore < 3).all(axis=1) & (df_zscore > -3).all(axis=1)]

data[features_of_interest].describe()


# %%



# Спроба знайти залежність між категоріальними ознаками
from scipy.stats import chi2_contingency

# Функція для обчислення Chi-Square тесту
def chi_square_test(col1, col2):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    return p  # p-значення тесту

# Обчислення p-значення для кожної пари змінних
for col1 in data_cat.columns:
    for col2 in data_cat.columns:
        if col1 != col2:
            p_value = chi_square_test(data_cat[col1], data_cat[col2])
            print(f'P-значення для {col1} і {col2}: {p_value}')
# Chi-Square тест показує статистичну значущість залежності, де низьке p-значення (наприклад, менше 0.05) свідчить про наявність залежності між змінними.

# %%
# Категоріальні ознаки. 
# Видаляємо 'Phone_Number', "Name", "Date_Of_Birth" які всі різні і як категорії не підходять
# видалення "Qualification" підбрано,- найкраше покращує результат (з Qualification та University, які по Chi-Squareу залежні)
# data_cat = data_cat.drop(['Phone_Number', "Name", "Date_Of_Birth", "Qualification"], axis=1)

# %%
# Кодування категоріальних ознак (наприклад, за допомогою об’єктів OneHotEncoder / TargetEncoder з пакета category_encoders).
encoder = ce.OneHotEncoder()
# encoder = ce.TargetEncoder()

X_train_cat = encoder.fit_transform(data_cat, y_train)

# %%
# 7. Об'єднання підмножини з числовими і категоріальними ознаками (після кодування) в одну:
X_train = pd.concat([X_train_num, X_train_cat], axis=1)


# %%

# Навчаємо модель і отримуємо прогнози:
  # class_weight='balanced' - невелюємо дисбаланс між позитивним і негативним класами (Ваги обчислюються автоматично під час тренування моделі.)
  # solver='lbfgs', liblinear, newton-cg, newton-cholesky, sag, saga (майже не впливають на результат)

model = (KNeighborsRegressor(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=12, p=1, metric='minkowski', metric_params=None, n_jobs=None)
       .fit(X_train, y_train))


# %%

# Обробка VALID dataset
y_true = valid.pop("Salary")

# # Зміна типу на int
# valid['Role'] = valid['Role'].map(role_mapping).astype(int)
# valid['Qualification'] = valid['Qualification'].map(qualification_mapping).astype(int)
# valid['University'] = valid['University'].map(u_mapping).astype(int)

# 3.2.Розбиваємо датасет на підмножини
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')

# Нормалізація числових, кодування категоріальних
X_test_num = scaler.transform(valid_num)

valid_cat = valid_cat.drop(['Phone_Number', "Name", "Date_Of_Birth", "Qualification"], axis=1)
X_test_cat = encoder.transform(valid_cat)

# 7. Об'єднання підмножини
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
X_train.shape


# %%
# Результат
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f'Validation MAPE: {mape:.2%}')
# Validation MAPE: 4.54%
# %%
print("y_pred:", y_pred)
print("y_true:", y_true.values)

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
X = pca.transform(X)[:, :n_components]

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
model_kmn = KMeans(n_clusters=k_best, random_state=42).fit(X)

labels_kmn = pd.Series(model_kmn.labels_, name='k-means')


#%% 


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Зменшуємо кількість вимірів до 2 за допомогою PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

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

#%%
# 8. Проаналізуйте звіт та зробіть висновки.

# Спроба без зменшення розмірності датасету а також спроба зменшення розмірності до 4 головних компонент замість 6 (що теж візуально виглядало допустимим за "правилом ліктя") помітних змін не привнесло. 
# В будь якому випадку k-means визначав найбільш опримальним 5 кластерів

# Візуально здається, що кластеризація відбулася на сама вдала, чітких кордонів між кластерами не спостерігається. Кластери перемішуються і зачіпають простір інших, шо може говорити про занадто високу кількість кластерів.

# На діарамах добре видно що приміс шлаку відсутній у цементі кластерів 0 та 4, А зола відсутня у кластрах 1, 3 та 4. 
# Крым того Суперпластифікатор відсутній у кластарі 4 
# І оскільки решта ознак більш менг рівномірні, схоже, це основна логіка розділення на кластари.
# Так якщо цемент не мамє шлаку, але має золу = 0 кластер
#      не має золи, клаку та суперпластифікатора = 4
#      має шлак і золу  = 2
#      а 1 та 3, які мають шлак та не мають золи, треба шукати у інших ознаках:
#          таких як кількість цементу, в 1 біля 200, а в 3 - біля 400 
#          та Міцність на стиск, в 1 до 40, а в 3 - під 60
#          тобто решта рецептур ділиться на 2 кластери по міцності (більше цементу дає більше міцності, менше цементу - менше міцності )
# Тому і бачимо, що найменге компонентів у кластері 4 (4 компонета), найбільше у кластері 2 (7 компонентів). Цікаво що середня кількість компонентів з 60 - 120 спостережень виявиляся цілим числом, що говорить про чітке попадання рецептур з 4 або 7 компонентами до своїх кластерів, що підтверджує залежність кластеризаціє саме від компонентів.   

# Варто сказати також про можливість кластеризації на 3 кластери (виявлено хляхом спроб). Достатьо чіткі кордони, та на 3D візуалізаціє видно що компонети шикуються в рядочки, немов би рецептури формувались штучно (наприклад, додаємо компоненту 120 грам, в наступний 130, 140, 150 і так далі, наким чином отримуємо серію рецептів)
# 3 кластери чіткорозподіляються на 0 містить, але не золу, 1: Ні шлаку ні золи (і суперкласівкатора тут нема); 2: І шлак і золу (до речі, вполовині менше сементу тоді треба) 