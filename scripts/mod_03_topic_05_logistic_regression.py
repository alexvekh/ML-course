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

data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')
data.shape

# %%

data.head()

# %%

data.dtypes

# Бачимо, що значення колонки Date (календарні дати) не відповідають її заявленому типу, тому це потрібно буде виправити згодом.

# %%
# Перевіримо датасет на наявність пропущених значень, 
# розрахувавши частку пропусків за кожною колонкою.

data.isna().mean().sort_values(ascending=False)

# %%
# У цьому наборі даних деякі колонки (Sunshine, Evaporation, Cloud3pm, Cloud9am) 
# мають значний відсоток пропущених значень (понад 35%). 
# Необхідно визначити, чи це пов'язано з роботою конкретних метеорологічних станцій. 
# Для цього порахуємо та візуалізуємо частку пропущених значень вхідних ознак для різних локацій.

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tmp = (data
           .groupby('Location')
           .apply(lambda x:
                  x.drop(['Location', 'Date'], axis=1)
                  .isna()
                  .mean()))

plt.figure(figsize=(9, 13))

ax = sns.heatmap(tmp,
                 cmap='Blues',
                 linewidth=0.5,
                 square=True,
                 cbar_kws=dict(
                     location="bottom",
                     pad=0.01,
                     shrink=0.25))

ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90)

plt.show()

# %%
# Поки що приймаємо рішення відкинути колонки, в яких частка відсутніх даних перевищує 0.35. 
# Це стандартна практика, бо спроба відновлення 30% і більше пропусків простими статистичними 
#  методами (без глибокого розуміння контексту) може лише призвести до додаткового зашумлення даних.

data = data[data.columns[data.isna().mean().lt(0.35)]]

# Також видалимо спостереження, для яких відсутня цільова мітка (пропуски в колонці RainTomorrow).
data = data.dropna(subset='RainTomorrow')

# %%
# Для подальшого аналізу розіб’ємо датасет на окремі вибірки в залежності від типів вхідних даних (числові й категоріальні):
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%
# Розглянемо розподіли числових ознак та звернемо увагу на їхню форму, яка схожа на розподіл випадкової величини. 
# Це є очікуваним, оскільки ми маємо справу з даними, які відображають природні явища.
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
# Тепер перейдемо до розгляду категоріальних ознак. 
# Рахуємо кількість унікальних категорійза кожною колонкою:
data_cat.nunique()

# %%
# Ось приклади категорій, які ми маємо в нашому датасеті:
data_cat.apply(lambda x: x.unique()[:5])

# %%
# Повернемось до розгляду змінної Date. 
# Оскільки ми будуємо “базовий” класифікатор, то не маємо наміру використовувати саме 
# послідовність спостережень (за декілька попередніх років) для прогнозу цільової змінної в майбутньому. 
# Проте, щоб використати ознаку часу (дати) ефективно в моделі, ми виділимо з неї 
# дві додаткові категорійні ознаки: рік і місяць проведення спостережень. 
# Це дозволить нам врахувати в моделі певні сезонні варіації (річні цикли) 
# та циклічні закономірності природних явищ, із якими ми маємо справу.

data_cat['Date'] = pd.to_datetime(data['Date'])

data_cat[['Year', 'Month']] = (data_cat['Date']
                               .apply(lambda x:
                                      pd.Series([x.year, x.month])))

data_cat.drop('Date', axis=1, inplace=True)

data_cat[['Year', 'Month']] = data_cat[['Year', 'Month']].astype(str)

data_cat[['Year', 'Month']].head()

# %%
# Розбиття на тренувальну й тестову вибірки
# train_test_split розіб'є на тренувальні й тестові набори для обох підвибірок 
# із числовими та категоріальними даними:
X_train_num, X_test_num, X_train_cat,  X_test_cat, y_train, y_test = (
    train_test_split(
        data_num,
        data_cat.drop('RainTomorrow', axis=1),
        data['RainTomorrow'],
        test_size=0.2,
        random_state=42))

# %%
# Відновлення пропущених значень — важливий крок
# Для пропущених числових ознак, можна використовувати середнє значення цієї ознаки (зберігаючи статистичні властивості її розподілу).
# Для категоріальних ознак пропущені можна замінити на моду (тобто категорію щр найчастіше зустрічається в цій ознаці).

# Інструмент - об’єкт SimpleImputer у бібліотеці scikit-learn. Важливо навчати SimpleImputer 
# на тренувальних даних, щоб уникнути витоку даних (не давати моделі “заглядати” в майбутнє), 
# а потім використовувати навчений екземпляр об’єкта для трансформації як тренувальних, 
# так і тестових (а згодом і нових) даних.
# Методи fit і transform об’єкта SimpleImputer забезпечують необхідну функціональність.

num_imputer = SimpleImputer().set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

pd.concat([X_train_num, X_test_num]).isna().sum()

# %%
# Повторимо ті самі дії для категоріальних ознак:

    cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

pd.concat([X_train_cat, X_test_cat]).isna().sum()

# %%
# Нормалізація змінних
scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%
# Onehot кодування категоріальних змінних
# Логістична регресія, потребує числового представлення категоріальних ознак.
# У датасеті категоріальні ознаки є номінальними, тобто не мають природного порядку. 
# Один із способів — це перетворення їх на багатовимірні бінарні вектори (onehot encoding).
# Кількість вимірів у векторі = кількістю категорій (для кожної категорії створюється вимір),
# а кожна категорія отримує вектор, де елемент, який відповідає категорії, дорівнює 1, а всі інші — 0.

# Для кодування категоріальних змінних використаємо об’єкт OneHotEncoder:

encoder = (OneHotEncoder(drop='if_binary',
                         sparse_output=False)
           .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train_cat.shape

# %%
# Тепер ми можемо об’єднати числові й закодовані категоріальні змінні в один набір даних, який і будемо використовувати для тренування моделі:

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

X_train.shape

# %%
## Навчання і оцінка моделі
# Перевіримо розподіл нашої цільової змінної:
y_train.value_counts(normalize=True)

# %%

# існує дисбаланс між позитивним і негативним класами. 
# Щоб ефективно тренувати модель з таким набором даних, можемо використовувати 
# параметр class_weight при створенні екземпляра об'єкта LogisticRegression.
# що дозволяє враховувати помилки, зроблені на спостереженнях класу [i] з вагою class_weight[i], 
# яка буде більшою для менш представленого у вибірці класу. 
# Це дозволяє моделі ефективніше враховувати менш численні класи при навчанні і точніше прогнозувати. 
# Ваги класів обчислюються автоматично під час тренування моделі.

# Навчаємо модель і отримуємо прогнози:

    clf = (LogisticRegression(solver='liblinear',
                          class_weight='balanced',
                          random_state=42)
       .fit(X_train, y_train))

pred = clf.predict(X_test)

# %%
# Оцінювання точності моделі
# Оскільки в нас є дисбаланс класів, метрика точності (accuracy) може бути неінформативною 
# для оцінки ефективності нашого класифікатора. Тому будуємо confusion matrix:

ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

# %%

print(classification_report(y_test, pred))

# Ми отримали класифікатор, який прогнозує, чи буде завтра дощ, чи ні, в залежності від локації та показників погоди, зафіксованих протягом поточного дня.
# Його точність (precision) складає 51% (4822/(4579+4822)⋅100), що вказує на середню ефективність прогнозування.
# Однак чутливість (recall) становить 76% (4822/(1519+4822)⋅100), що свідчить про здатність моделі виявляти більшість випадків дощу.

# Це означає, що в одному з чотирьох випадків, коли йтиме дощ, ми опинимося на вулиці без парасольки (класифікатор правильно передбачив 4822 із 6341, тобто 3 з 4 дощових днів, які були в тестовому наборі). 
# Але при цьому ми носитимемо її з собою вдвічі частіше, ніж можна було б очікувати (класифікатор помилково спрогнозував 4579 дощових днів на додаток до 4822, які правильно ідентифікував у тестовому наборі.

# Зверніть увагу, що обраний нами метод розділення набору вхідних даних і кодування часових ознак не дозволяє об'єктивно оцінити здатність моделі до прогнозування дощу в умовному «майбутньому», 
# тобто періоді, якого модель не бачила під час навчання. Виправити це ми спробуємо пізніше в рамках домашнього завдання до цього модуля. 