# ML-pipe

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import estimator_html_repr
from mod_07_topic_13_transformer import VisRatioEstimator

# %%
# Цей набір містить інформацію про продажі 1559 товарів у 10 магазинах мережі в різних містах протягом 2013 року. 
#  товар і магазин описані певними ознаками.
# Нашою метою є побудова моделі, яка зможе прогнозувати продажі товарів у конкретному магазині.
data = pd.read_csv('../datasets/mod_07_topic_13_bigmart_data.csv')
data.sample(10, random_state=42)

# Отже, набір даних містить такі змінні:
#   - Item_Identifier — унікальний ідентифікатор товару,
#   - Item_Weight — вага товару,
#   - Item_Fat_Content — категорія жирності продукту,
#   - Item_Visibility — відсоток від загальної площі викладки всіх товарів у магазині, виділений для конкретного товару,
#   - Item_Type — категорія, до якої належить товар,
#   - Item_MRP — максимальна роздрібна ціна (прайс-лист) товару,
#   - Outlet_Identifier — унікальний ідентифікатор магазину,
#   - Outlet_Establishment_Year — рік заснування магазину,
#   - Outlet_Size — розмір (площа) магазину.
#   - Outlet_Location_Type — тип населеного пункту, в якому знаходиться магазин,
#   - Outlet_Type — тип магазину (звичайний продуктовий магазин чи супермаркет).
#   - Item_Outlet_Sales — цільова змінна, продажі товару в конкретному магазині.


# %%
# Перевірка типів даних і відсутніх значень
data.info()
# Набір даних складається з числових і категоріальних змінних, при цьому в обох типах змінних є пропуски. 
# У контексті машинного навчання перед нами стоїть задача регресії, тобто передбачення значень неперервної числової змінної.


# %%
# Змінна Outlet_Establishment_Year вказує на рік відкриття магазину.
# Ефективніше буде розрахувати загальний час функціонування кожного магазину на момент збору даних (2013 рік).
# Припущення в тому, що магазини з більшим терміном роботи мають більше постійних клієнтів, отже, 
# така похідна ознака може стати більш інформативною для прогнозування продажів.
data['Outlet_Establishment_Year'] = 2013 - data['Outlet_Establishment_Year']

# %%
# Item_Visibility вказує на відсоток площі викладки, виділеної для конкретного товару. 
# Наявність нульових значень для цієї колонки є дещо дивною, оскільки товар, який продається в магазині, не може займати нульову площу.
# Це може бути помилкою у вхідних даних. Тому ми заміняємо нульові значення в цій колонці на середнє значення для групи "категорія товару — магазин".

# припущення, що товари, які займають більше місця в торговій залі, мають кращі продажі (бо більш помітні для покупців).

# Обчислимо додатковий коефіцієнт, який допоможе визначити, наскільки товар краще або гірше представлений у своїй групі. 
# Іншими словами, ми розрахуємо співвідношення площі, яку займає конкретний товар у магазині, 
# до середньої площі, яку займають товари у відповідній групі в тому самому магазині.

# Для цього ми спочатку обчислюємо додаткову колонку Item_Visibility_Avg, далі використовуємо її для заміни нульових значень 
# у колонці Item_Visibility, а потім — для розрахунку відносного “індексу помітності товару” в торговій залі (Item_Visibility_Ratio).

data['Item_Visibility'] = (data['Item_Visibility']
                           .mask(data['Item_Visibility'].eq(0), np.nan))

data['Item_Visibility_Avg'] = (data
                               .groupby(['Item_Type',
                                         'Outlet_Type'])['Item_Visibility']
                               .transform('mean'))

data['Item_Visibility'] = (
    data['Item_Visibility'].fillna(data['Item_Visibility_Avg']))

data['Item_Visibility_Ratio'] = (
    data['Item_Visibility'] / data['Item_Visibility_Avg'])

data[['Item_Visibility', 'Item_Visibility_Ratio']].describe()

# %%
# Тепер детальніше розглянемо ознаку Item_Fat_Content і перевіримо перелік унікальних категорій у цій колонці:
data['Item_Fat_Content'].unique()

 # %%
# Неузгоджені позначення категорій жирності товарів,- використання різних варіантів записів ('low fat' / 'LF') для позначення категорії "Low Fat".
# Ми можемо виправити цю ситуацію, замінивши текстові значення відповідної колонки:
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'})

# %%
# Також розглянемо ознаку Item_Identifier. Наразі ми не будемо включати її безпосередньо в модель.
# По-перше, ця ознака має занадто велику кількість унікальних значень (має високу кардинальність), 
# і ми не зможемо ефективно застосувати знайомі нам методи кодування без наслідків у вигляді витоку даних або збільшення їх розмірності. 

# По-друге, зберігаючи цей ідентифікатор як вхідну змінну в моделі, ми обмежуємо її придатність для практичного застосування, 
# оскільки не зможемо прогнозувати продажі товарів, яких немає у вхідних даних. 
# Тобто нові унікальні ідентифікатори, що надаватимуться новим товарам у майбутньому, не міститимуть жодної корисної інформації для моделі.

# Проте, якщо звернути увагу на структуру самого ідентифікатора, то можна помітити, що він складається з комбінації літер і цифр. 
# Літери повторюються, що, ймовірно, вказує на приналежність товару до певної категорії в товарній ієрархії. 
# Загалом є три унікальні комбінації літер: FD, DR, NC — які, наприклад, можуть означати категорії Food, Drinks, Non-Consumable. 
# Припускаємо, що додавши в модель змінну для позначення більш високого рівня в ієрархії товарів, ми можемо покращити її точність.

# Вводимо нову похідну ознаку (Item_Identifier_Type), для створення якої використовуємо перші два символи з рядка ідентифікатора. 
data['Item_Identifier_Type'] = data['Item_Identifier'].str[:2]

data[['Item_Identifier', 'Item_Identifier_Type', 'Item_Type']].head()

# %%
# Після попередньої очистки даних і створення додаткових ознак на основі наших припущень переходимо до подальших типових методів обробки даних,
# які неодноразово використовували раніше.

# Ці методи включають розбиття набору даних на тренувальний і тестовий, відновлення пропущених значень для числових і категоріальних ознак
# за допомогою стандартних інструментів бібліотеки sklearn, а також кодування категоріальних змінних.

# Оскільки наш план включає побудову моделі-прототипа на основі RandomForest (ансамбль дерев рішень), 
# наразі в процесі підготовки даних ми можемо не виконувати їх нормалізацію.

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%

X_train_num, X_test_num, X_train_cat,  X_test_cat, y_train, y_test = (
    train_test_split(
        data_num.drop(['Item_Outlet_Sales',
                       'Item_Visibility_Avg'], axis=1),
        data_cat.drop('Item_Identifier', axis=1),
        data['Item_Outlet_Sales'],
        test_size=0.2,
        random_state=42))

# %%
# Відновлення відсутніх значень
num_imputer = SimpleImputer().set_output(transform='pandas')

X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

# %%

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

# %%
# Кодування категоріальних змінних
enc_auto = TargetEncoder(random_state=42).set_output(transform='pandas')

X_train_cat = enc_auto.fit_transform(X_train_cat, y_train)
X_test_cat = enc_auto.transform(X_test_cat)

# %%
# Об’єднуємо обидві окремі групи числових і категоріальних ознак, які обробляли паралельно у відповідні вибірки (тренувальну і тестову), 
# які будемо використовувати для навчання й оцінки якості прогнозування моделі.
X_train_concat = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_concat = pd.concat([X_test_num, X_test_cat], axis=1)

X_train_concat.head()

# %%
# Побудова базової моделі
  # Використовуємо алгоритм Random Forest для побудови моделі. 
  # Оскільки задача прогнозування — це регресія, то для оцінки моделі обираємо одну з відповідних метрик, наприклад, Root Mean Square Error (RMSE).
clf = (RandomForestRegressor(
    n_jobs=-1,
    random_state=42)
    .fit(X_train_concat,
         y_train))

pred_clf = clf.predict(X_test_concat)

rmse_clf = root_mean_squared_error(y_test, pred_clf)

print(f"Prototype's RMSE on test: {rmse_clf:.1f}")

# %%
# Важливість ознак. Відбір ознак у конвеєр
  # Проведемо оцінку важливості ознак і перевіримо наші припущення й наскільки корисними стали для моделі додаткові змінні, згенеровані на їх основі. Для цього ми будуємо графік важливості ознак, подібно до того, як ми це робили, наприклад, у темі “Зменшення розмірності даних. Метод PCA”.
sns.set_theme()

(pd.Series(
    data=clf.feature_importances_,
    index=X_train_concat.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()

# На графіку можна помітити, що ознака Item_Visibility_Ratio отримала високий рівень значимості, що свідчить про її важливість для моделі. 
# Схоже, що введення цієї ознаки покращило прогнозування. З іншого боку, ознака Item_Identifier_Type має один з найнижчих рівнів значимості, 
# тому її збереження в майбутніх розрахунках може бути недоцільним.
# %%

#  Усі ці дії є важливими, але також витратними за часом і зусиллями. За допомогою конвеєрів ми можемо автоматизувати ці кроки.

# ML-конвеєр дозволяє виконати всю сукупність дій обробки даних, навчання моделі з подальшим її використанням 
# для отримання прогнозів на нових даних, усього лише за два кроки, через виклик методів fit() і predict().

# вимоги до створення нестандартного трансформера:
    # По-перше, трансформер повинен бути реалізований як Python-клас, який успадковує методи від класів BaseEstimator і TransformerMixin.
    # Це дозволяє забезпечити сумісність трансформера з іншими об'єктами в екосистемі sklearn.
    # По-друге, у трансформері мають бути реалізовані три базові методи: init(), fit() і transform(). (ініціалізація класу, навчання моделі, обробка даних.


# Для нестандартного трансформера закладаємо логіку розрахунку середніх значень за групами "категорія товару — магазин" на тренувальному наборі, 
# тобто в методі fit(), а потім будемо використовувати ці збережені середні значення для обчислення нової ознаки під час виклику методу transform().

from sklearn.base import BaseEstimator, TransformerMixin

class VisRatioEstimator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.vis_avg = None

    def fit(self, X, y=None):
        vis_avg = (X
                   .groupby(['Item_Type', 'Outlet_Type'])['Item_Visibility']
                   .mean())
        self.vis_avg = vis_avg
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Item_Visibility_Ratio'] = (
            X
            .groupby(['Item_Type', 'Outlet_Type'])['Item_Visibility']
            .transform(lambda x:
                       x / self.vis_avg[x.name]))
        return X

#%%
# Наш нестандартний трансформер розширює набір вхідних даних, додаючи до нього нову колонку.
# Щоб переконатися в коректності його роботи, проведемо перевірку.
# У наборі даних уже існує розрахована раніше вручну колонка з назвою Item_Visibility_Ratio, яку ми перейменуємо в Item_Visibility_Ratio_prev. 
# Далі застосуємо трансформер, який додасть розраховану колонку Item_Visibility_Ratio. 
# Потім порівняємо значення колонки, доданої трансформером, зі значеннями, які ми обчислювали раніше під час підготовки даних для моделі-прототипу.

vis_est = VisRatioEstimator()

data = (data.rename(columns={
    'Item_Visibility_Ratio': 'Item_Visibility_Ratio_prev'}))

data = vis_est.fit_transform(data)

(data[['Item_Visibility_Ratio_prev', 'Item_Visibility_Ratio']]
 .sample(10, random_state=42))

# Як бачимо, значення в колонках Item_Visibility_Ratio_prev і Item_Visibility_Ratio збігаються. 
# Тепер, коли ця перевірка успішно завершена, ми маємо всі необхідні блоки для створення конвеєра.
# %%
# У пакеті sklearn конвеєр будується (описується) шляхом передачі об’єкту Pipeline списку пар “ключ-значення”. 
# ключ — це довільна назва етапу, а значення — це об'єкт-трансформер або оцінювач із заданими параметрами.
# Pipeline часто поєднується з об’єктами ColumnTransformer або FeatureUnion, які дозволяють об'єднати результати трансформерів у єдиний набір ознак.
# ColumnTransformer — це об'єкт, який дозволяє окремо (паралельно) трансформувати різні ознаки або їх групи, а потім об'єднати їх у єдиний набір. 
# Це корисно для табличних даних, де різні типи ознак можуть потребувати різної обробки (наприклад, числові категоріальні),
# і результати цих обробок потрібно знову об'єднати разом.

# Використовуємо Pipeline, щоб відтворити архітектуру конвеєра й логіку обробки вхідних даних для нашої конкретної задачі:

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(random_state=42))])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer())])

preprocessor = (ColumnTransformer(
    transformers=[
        ('cat',
         cat_transformer,
         make_column_selector(dtype_include=object)),
        ('num',
         num_transformer,
         make_column_selector(dtype_include=np.number))],
    n_jobs=-1,
    verbose_feature_names_out=False)
    .set_output(transform='pandas'))

model_pipeline = Pipeline(steps=[
    ('vis_estimator', VisRatioEstimator()),
    ('pre_processor', preprocessor),
    ('reg_estimator', RandomForestRegressor(
        n_jobs=-1,
        random_state=42))])

# %%

with open('../derived/mod_07_topic_13_mlpipe.html', 'w', encoding='utf-8') as fl:
    fl.write(estimator_html_repr(model_pipeline))

with open('../models/mod_07_topic_13_mlpipe.joblib', 'wb') as fl:
    joblib.dump(model_pipeline, fl)

# %%
# Відновлення даних. Розбивка на тренувальну і тестову вибірки

# Тепер ми можемо скористатися створеним нами конвеєром.
# Але спочатку відновимо вхідні дані, видаливши колонки з проміжними розрахунками та новими ознаками, 
# які були створені на етапі побудови моделі-прототипу.
    # Важливо зазначити, що вхідний набір міститиме результати очищення даних в окремих колонках, тобто ми симулюємо ситуацію, 
    # в якій до конвеєра потрапляють частково підготовлені й оброблені дані (як ми це обговорювали раніше).

data.drop([
    'Item_Visibility_Avg',
    'Item_Visibility_Ratio_prev',
    'Item_Visibility_Ratio',
    'Item_Identifier_Type'],
    axis=1,
    inplace=True)

data.sample(10, random_state=42)

# %%

data.to_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')

# %%
# Розбиваємо дані на тренувальну і тестову вибірки. (Зверніть увагу, що тепер це набори із “сирими” даними.)

X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop(['Item_Identifier',
                   'Item_Outlet_Sales'],
                  axis=1),
        data['Item_Outlet_Sales'],
        test_size=0.2,
        random_state=42))

X_train.head(10)

# %%
# Навчання й оцінка моделі в конвеєрі

# Навчання моделі в конвеєрі відбувається за допомогою методу fit(), отримання прогнозів за допомогою методу predict(). 
# Ми навчаємо модель, отримуємо прогнози й оцінюємо результат.

model = model_pipeline.fit(X_train, y_train)

pred_pipe = model.predict(X_test)
rmse_pipe = root_mean_squared_error(y_test, pred_pipe)

print(f"Pipe's RMSE on test: {rmse_pipe:.1f}")

# Зауважимо, що метрика RMSE не дуже відрізняється від значення моделі-прототипа (1047,4 проти 1048,5). 
# Відмінність з’являється за рахунок того, що в моделі-прототипі ми використовували додаткову ознаку Item_Identifier_Type.
# Тому можемо вважати, що наш конвеєр правильно відтворює всі кроки обробки даних, виконані в рамках побудови моделі-прототипа.

# %%
# Крос-валідація
# Крос-валідація (cross-validation/перехресна перевірка) — полягає в багаторазовому розділенні даних на N груп (тренувальний і тестовий набори):
    # Кожна з цих груп використовується по черзі для тестування, а решта — для навчання. 
    # Після кожного прогону тестовий набір стає частиною тренувальних даних у наступному циклі, а один із тренувальних наборів стає тестовим. 
    # На кожному етапі крос-валідації виконується повний комплекс операцій з підготовки даних для навчання моделі, та обробки тестових перед прогнозуванням. 
    # Практично кожна ітерація передбачає навчання та оцінку нової моделі.
    # Без концепції конвеєра реалізація такого повторюваного процесу була б доволі складною.
# Крос-валідація допомагає уникнути випадкового формування “нерепрезентативного” (занадто простого або складного) набору тестових даних, 
# що може призвести до неточних висновків щодо ефективності моделі. 
# Метрика, усереднена за всіма етапами крос-валідації, дає оцінку якості моделі в умовах, наближених до реальних.

# Крос-валідація моделі в конвеєрі
# Маючи конвеєр, ми можемо зручно виконати 5-кратну крос-валідацію моделі за допомогою методу cross_val_score() пакета sklearn:
cv_results = cross_val_score(
    estimator=model_pipeline,
    X=X_train,
    y=y_train,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1)

rmse_cv = np.abs(cv_results).mean()

print(f"Pipe's RMSE on CV: {rmse_cv:.1f}")

# Середня помилка на крос-валідації виявилася більшою, ніж помилка, отримана на відкладеному тестовому наборі (1048,5). 
# Частково це може бути пояснено тим, що при крос-валідації ми використовували щоразу лише 80% доступних для навчання даних 
# (на кожній ітерації модель навчалася на 4/5 тренувальних даних і перевірялася на 1/5).
# З іншого боку, для оцінки якості моделі ми використали 5 різних тестових наборів, тому результат, отриманий на перехресній перевірці, 
# слід вважати більш об'єктивним і наближеним до реальних умов використання моделі.
