import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Завантаження даних
data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')
data.shape

# %%
data.head()

# %%
data.dtypes

# %%
# процент пропусків за кожною колонкою.
data.isna().mean().sort_values(ascending=False)
# Sunshine, Evaporation, Cloud3pm, Cloud9am мають понад 35% пропущених значень. 

# %%
# 3.1. Видаляємо із набору ознаки з великою кількістю пропущених значень.

data = data[data.columns[data.isna().mean().lt(0.35)]]

# Також видалимо спостереження, для яких відсутня цільова мітка (пропуски в колонці RainTomorrow).
data = data.dropna(subset='RainTomorrow')
data.shape

# %%
# 3.2.Розбиваємо датасет на підмножини набору даних із числовими та категоріальними ознаками.:
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%
# Ррозподіли числових ознак (більшість нормльні)
melted = data_num.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=4,
                  sharex=False,
                  sharey=False,
                  aspect=1.25)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# %%
# Категоріальні ознаки. 
# Кількість унікальних категорій за кожною колонкою:
data_cat.nunique()

# %%
# категорії в нашому датасеті:
data_cat.apply(lambda x: x.unique()[:5])

# %%
# 3.3. Зміна типу колонки Date на тип datetimeі та створення з неї додаткових колонок Year та Month.

data_cat['Date'] = pd.to_datetime(data['Date'])

data_cat[['Year', 'Month']] = (data_cat['Date']
                               .apply(lambda x:
                                      pd.Series([x.year, x.month])))

data_cat.drop('Date', axis=1, inplace=True)

data_cat[['Year', 'Month']] = data_cat[['Year', 'Month']].astype(str)

data_cat[['Year', 'Month']]

# %%
# 3.4. Переміщення створеної колонки Year з підмножини набору із категоріальними ознаками до підмножини із числовими ознаками.

# Копіювання до числового набору зі зміною типу
data_num['Year'] = data_cat['Year'].astype(int)

# Видалення з категоріального нобору
data_cat.drop('Year', axis=1, inplace=True)

# %%
# перевірка чи вийшло
print("-> data_cat:", data_cat.columns)
print("-> data_num:", data_num.columns)

# %%

# Розподіли котегоріальних ознак.
melted = data_cat.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=4,
                  sharex=False,
                  sharey=False,
                  aspect=1.25)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show() 

# %%
# 3.5. Розбиваємо підмножини на тренувальну і тестову вибірки

# останній рік
last_year = data_num['Year'].max()

# Тренувальні дані: всі роки, окрім останнього
X_train_num = data_num[data_num['Year'] < last_year].copy()
X_test_num = data_num[data_num['Year'] == last_year].copy()

X_train_cat = data_cat[data_num['Year'] < last_year].copy()
X_test_cat = data_cat[data_num['Year'] == last_year].copy()

# Видаляємо колонку 'Year' з числових даних, щоб вона не використовувалась як ознака
X_train_num.drop('Year', axis=1, inplace=True)
X_test_num.drop('Year', axis=1, inplace=True)

# Беремо цільову змінну 'RainTomorrow'
y_train = X_train_cat['RainTomorrow']
y_test = X_test_cat['RainTomorrow']

# Видаляємо 'RainTomorrow' з категоріальних даних, щоб не було вхідною ознакою
X_train_cat.drop('RainTomorrow', axis=1, inplace=True)
X_test_cat.drop('RainTomorrow', axis=1, inplace=True)

# %%
# Перевіримо відповідність розмірів
print("X_train_num ", X_train_num.shape)
print("X_train_cat ", X_train_cat.shape)
print("y_train     ", y_train.shape)
print("X_test_num  ", X_test_num.shape)
print("X_test_cat  ", X_test_cat.shape)
print("y_test      ", y_test.shape)


# %%
# Чи є пусті значення
print("X_train_num ", X_train_num.isna().any().any())
print("X_train_cat ", X_train_cat.isna().any().any())
print("y_train     ", y_train.isna().any().any())
print("X_test_num  ", X_test_num.isna().any().any())
print("X_test_cat  ", X_test_cat.isna().any().any())
print("y_test      ", y_test.isna().any().any())

# %%
# 4. Відновлення пропущених даних за допомогою об'єкта SimpleImputer з пакету sklearn

num_imputer = SimpleImputer().set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

pd.concat([X_train_num, X_test_num]).isna().sum()

# %%
# Повторимо те саме для категоріальних ознак:
cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

pd.concat([X_train_cat, X_test_cat]).isna().sum()

# %%
# 5. Нормалізація числових ознак за допомогою об'єкта StandardScaler з пакету sklearn
scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%

# 6. Кодування категоріальних ознак за допомогою об’єкта OneHotEncoder з пакету sklearn.
    # Логістична регресія, потребує числового представлення категоріальних ознак.
    # Один із способів — це перетворення їх на багатовимірні бінарні вектори (onehot encoding).
    # Кількість вимірів у векторі = кількістю категорій (для кожної категорії створюється вимір),
    # а кожна категорія отримує вектор, де елемент, який відповідає категорії, дорівнює 1, а всі інші — 0.

encoder = (OneHotEncoder(drop='if_binary',
                         sparse_output=False)
           .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train_cat.shape

# %%
# 7. Об'єднання підмножини з числовими і категоріальними ознаками (після кодування) в одну:

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

X_train.shape

# %%
# Перевіримо розподіл нашої цільової змінної:
y_train.value_counts(normalize=True)

# %%

# Навчаємо модель і отримуємо прогнози:
  # class_weight='balanced' - невелюємо дисбаланс між позитивним і негативним класами (Ваги обчислюються автоматично під час тренування моделі.)
  # solver='lbfgs', liblinear, newton-cg, newton-cholesky, sag, saga (майже не впливають на результат)

clf = (LogisticRegression(solver='liblinear',   
                          class_weight='balanced',
                          max_iter=10,
                          random_state=42)
       .fit(X_train, y_train))

pred = clf.predict(X_test)

# %%
threshold = 0.50

y_pred_proba = pd.Series(clf.predict_proba(X_test)[:,1])
print(y_pred_proba)

y_pred = y_pred_proba.apply(lambda x: 'Yes' if x > threshold else 'No')
print(y_pred)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(classification_report(y_test, y_pred))

df_proba = pd.DataFrame()
df_proba['y_pred_proba'] = y_pred_proba
df_proba['y_test'] = y_test

target_class = 'Yes'

df_proba[
    (df_proba['y_pred_proba'] < threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=20, color='red');

df_proba[
    (df_proba['y_pred_proba'] >= threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=20, color='green');

plt.title(f'class = {target_class}');


# %%
# результати прошгнозів
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()

# %%
# 8. Розраховуємо метрики нової моделі за допомогою методу classification_report()
print(classification_report(y_test, y_pred))

# Ми отримали класифікатор, який прогнозує, чи буде завтра дощ, чи ні, в залежності від локації та показників погоди, зафіксованих протягом поточного дня.
# Його точність (precision) складає 51% (4822/(4579+4822)⋅100), що вказує на середню ефективність прогнозування.
# Однак чутливість (recall) становить 76% (4822/(1519+4822)⋅100), що свідчить про здатність моделі виявляти більшість випадків дощу.

# Це означає, що в одному з чотирьох випадків, коли йтиме дощ, ми опинимося на вулиці без парасольки (класифікатор правильно передбачив 4822 із 6341, тобто 3 з 4 дощових днів, які були в тестовому наборі). 
# Але при цьому ми носитимемо її з собою вдвічі частіше, ніж можна було б очікувати (класифікатор помилково спрогнозував 4579 дощових днів на додаток до 4822, які правильно ідентифікував у тестовому наборі.

# Зверніть увагу, що обраний нами метод розділення набору вхідних даних і кодування часових ознак не дозволяє об'єктивно оцінити здатність моделі до прогнозування дощу в умовному «майбутньому», 
# тобто періоді, якого модель не бачила під час навчання. Виправити це ми спробуємо пізніше в рамках домашнього завдання до цього модуля. 
