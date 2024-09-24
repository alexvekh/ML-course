import warnings
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import category_encoders as ce 
    # conda install category_encoders
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# застосування kNN та Naive Bayes. EDA
# на наборі даних Bank Marketing, який є результатом промокампанії (прямих дзвінків клієнтам), проведеної одним із банків Португалії. 
# Задача класифікації — передбачити, чи погодиться клієнт відкрити строковий депозит у банку.

# Завантажуємо дані та знайомимось із описом ознак.
data = pd.read_csv('../datasets/mod_04_topic_07_bank_data.csv', sep=';')
data.head()

# Ознаки клієнта:
  # default: чи має клієнт прострочений кредит?
  # housing: чи має клієнт кредит на житло?
  # loan: чи має споживчий кредит?
# Ознаки, які описують характер контактів із клієнтом:
  # contact: тип зв'язку з клієнтом (мобільний / стаціонарний телефон) у рамках поточної кампанії;
  # month: місяць, у якому відбувся останній контакт у рамках поточної кампанії;
  # day_of_week: день тижня, у який відбувся останній контакт у рамках поточної кампанії;
  # duration: тривалість останньої розмови з клієнтом у рамках поточної кампанії;
  # campaign: кількість контактів із клієнтом у рамках поточної кампанії;
  # pdays: кількість днів із дня останнього контакту в рамках попередньої кампанії (999 означає, що з клієнтом раніше не контактували);
  # previous: кількість контактів із клієнтом до поточної кампанії;
  # poutcome: результат попередньої маркетингової кампанії.
# Макроекономічні ознаки:
  # emp.var.rate: зміна коефіцієнта зайнятості (квартальний показник);
  # cons.price.idx: індекс споживчих цін (місячний показник);
  # cons.conf.idx: індекс споживчих настроїв (місячний показник);
  # euribor3m: 3-місячна ставка euribor (щоденний індикатор);
  # nr.employed: кількість зайнятих працівників (квартальний показник).
# %%
# Увага! ризик витоку даних: ознака duration сильно пов'язана з цільовою змінною. 
# Важливо розуміти, що час тривалості розмови невідомий до моменту дзвінка, а після завершення дзвінка ми вже знаємо цільову мітку (клієнт у ході розмови озвучує своє рішення щодо відкриття депозиту). 
# Отже, для прогнозів до початку дзвінка ми не можемо використовувати ознаку duration для навчання моделі.

data.drop('duration', axis=1, inplace=True)

# %%
# Розподіл ознак
data.describe()
# більшість клієнтів мають від 1 до 3 контактів (campaign) у рамках поточної кампанії. Однак є випадок, де це значення становить 56-аномалія
# %%
# Раніше ми використовували гістограми для дослідження розподілу ознак. 
# Але можна методом skew об’єкта DataFrame, який повертає значення асиметрії (skew / skeweness) для числових ознак набору даних, 
# даючи розуміння, наскільки сильно вони відхиляються від свого середнього значення.

data.skew(numeric_only=True)
# висновок, що ознаки в наборі даних, для яких показник асиметрії значно відрізняється від 0, мають зміщений (асиметричний) розподіл

# %%

# Розподіл ознак для порівняння.(добавлено)
melted = data.melt()

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
# етапи є бажані (+), а які необов'язкові (-) з точки зору впливу на ефективність відповідного алгоритму.
#         									            kNN	  Naive Bayes
# Очистка від викидів						             +	   +
# Видалення ознак, тісно пов`язаних з іншими ознаками 	 -	   +
# Нормалізація / стандартизація ознак				     +	   -
# Зменшення асиметрії (power transform) розподілу ознак	 +	   -
# Балансування класів						             -	   +

# Power transform — це група методів, які використовують ступеневі закони для зміни статистичних характеристик ознаки.
# Для поліпшення симетрії розподілу цієї ознаки. Раніше ми використовували просте логарифмування, але існують більш складні методи, які дозволять працювати і з від’ємними значеннями.

# %%
# Очистка від викидів
data = data[zscore(data['campaign']).abs().lt(2)]

# %%

# Видалення ознак, тісно пов`язаних з іншими ознаками
# Будуємо матрицю кореляції вхідних змінних між собою, щоб визначити ознаки, які можна видалити з набору даних.
mtx = data.drop('y', axis=1).corr(numeric_only=True).abs()

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(mtx,
            cmap='crest',
            annot=True,
            fmt=".2f",
            linewidth=.5,
            mask=np.triu(np.ones_like(mtx, dtype=bool)),
            square=True,
            cbar=False,
            ax=ax)

plt.show()

# %%

# Для кожної пари із взаємною кореляцією близькою до ~0.7 видаляємо одну (будь-яку) з ознак:
data.drop(
    ['emp.var.rate',
     'cons.price.idx',
     'nr.employed'],
    axis=1,
    inplace=True)

# %%
# Огляд категоріальних ознак
# Виводимо кількість унікальних значень за кожною з категоріальних ознак, 
# приймаємо рішення застосувати до них один із методів кодування після розбивки набору даних на тренувальну та тестову вибірки.

data.select_dtypes(include='object').nunique()

# %%
# категоріальні ознаки (добавлено)
data.select_dtypes(include='object').apply(lambda x: x.unique()[:5])
# %%
# Змінюємо тип цільової змінної на числовий для коректної роботи методів кодування категоріальних ознак на наступних кроках:

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    data['y'] = data['y'].replace({'no': 0, 'yes': 1})

# %%
# Розбиття на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('y', axis=1),
        data['y'],
        test_size=0.2,
        random_state=42))

# %%
# Кодування категоріальних змінних
# Окрім методу Onehot кодування, я існують інші методи, які враховують інформацію про цільову мітку.
# Зазвичай для кодування категорії певної ознаки використовують описові статистики або характеристики розподілу 
# (та виведені метрики на їх основі) підмножини цільової змінної, яка асоціюється із відповідною категорією визначеної ознаки.

# Наразі використаємо метод кодування Weight Of Evidence, який часто застосовується у задачах кредитного скорингу.
# Цей метод кодує категорію ознаки шляхом розрахунку логарифма співвідношення часток об'єктів класу 0 і 1 (співвідношення шансів 
# odds ratio, яке ми згадували в темі "Логістична регресія") в підмножині цільової змінної, яка утворюється відповідною категорією.

# використовуємо пакет categorical_encoders зі sklearn. Спочатку відбираємо категоріальні ознаки, а потім навчаємо наш екземпляр 
# об’єкта WOEEncoder (енкодер) та використовуємо його для трансформації категоріальних ознак у тренувальному та тестовому наборах даних.

cat_cols = X_train.select_dtypes(include='object').columns
cat_cols

# %%

encoder = ce.WOEEncoder(cols=cat_cols)

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %%
# Нормалізація змінних. Зменшення асиметрії (після кодування всі ознаки числові) і одночасно нормалізувати їх можна за допомогою об’єкта PowerTransformer пакета sklearn.

power_transform = PowerTransformer().set_output(transform='pandas')

X_train = power_transform.fit_transform(X_train)
X_test = power_transform.transform(X_test)

# %%
# Звертаємо увагу, як асиметрія зменшилась і наблизилась до 0, що означає, що їх розподіли тепер більше нагадують нормальні.
X_train.skew()

# %%
# Балансування класів 

y_train.value_counts(normalize=True)

# %%
# дивимось на розподіл класів цільової змінної і приймаємо рішення про балансування тренувального набору шляхом генерації 
# додаткових спостережень класу 0 із використанням об’єкта SMOTE пакета imblearn.

sm = SMOTE(random_state=42, k_neighbors=50)
X_res, y_res = sm.fit_resample(X_train, y_train)

# %%
# kNN-класифікатор
# Створюємо екземпляр об’єкта KNeighborsClassifier (реалізація kNN-класифікатора в пакеті sklearn) з параметром n_neighbors=15, 
# навчаємо його, отримуємо прогнози для тестового набору даних і розраховуємо метрику збалансованої точності (balanced_accuracy_score).
knn_mod = KNeighborsClassifier(n_neighbors=7, n_jobs=-1).fit(X_res, y_res)

knn_preds = knn_mod.predict(X_test)

knn_score = balanced_accuracy_score(y_test, knn_preds)

print(f'KNN model accuracy: {knn_score:.1%}')

# %%
# Naive Bayes класифікатор
# Аналогічні кроки для Naive Bayes класифікатора (об’єкт GaussianNB — реалізацію гаусівського наївного баєсівського класифікатора в пакеті sklearn).
gnb_mod = GaussianNB().fit(X_res, y_res)

gnb_preds = gnb_mod.predict(X_test)

gnb_score = balanced_accuracy_score(y_test, gnb_preds)

print(f'GNB model accuracy: {gnb_score:.1%}')

# %%
# при використанні різних моделей на тих самих даних ми отримали приблизно однакову точність прогнозів, на рівні ~70%. 
# Щоб глибше розібратися, як моделі відрізняються у своїх прогнозах, слід проаналізувати спостереження, 
# на яких моделі зробили помилки, але зараз ми цього не будемо робити.
# %%
# оцінимо ефективність моделей
confusion_matrix(y_test, gnb_preds)

# Тепер банк може ефективніше планувати свої майбутні промокампанії. 
# Використовуючи навіть такий простий класифікатор для відбору "перспективних" клієнтів для дзвінків 
# (тобто клієнтів із прогнозованим класом 1), достатньо буде звернутися (подзвонити) лише до ~21% клієнтів 
# ((1123+522)/7963⋅100, щоб охопити ~58% (522/(384+522))⋅100  тих, хто потенційно зацікавлений у відкритті депозиту.