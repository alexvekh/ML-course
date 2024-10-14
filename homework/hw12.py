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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%

# 1. Завантажте набір даних Concrete.

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

concrete = datasets['concrete']
concrete.info()


#%% 

# 2. Використайте прийом підрахунку кількості для створення нової ознаки Components, 
# яка вказуватиме на кількість задіяних складових у різних рецептурах бетону.

components = ['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']

concrete['Components'] = concrete[components].gt(0).sum(axis=1)
    # .greater than 0 .sum(axis=1): підсумовує результат по рядках.

concrete[components + ['Components']].head(10)

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