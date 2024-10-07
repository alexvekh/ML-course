# feature engineering
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# 1. Завантаження набору даних Autos

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

data = datasets['autos']
data.info()

# %%

# Добавимо відношення довжини ходу поршня stroke до діаметра циліндра borе (відношення S/D: Stroke/Diameter)
data['stroke_ratio'] = data['stroke'] / data['bore']
data[['stroke', 'bore', 'stroke_ratio']].head()


# %%
# пеевірка на на наявність пропусків
data.isna().sum()
        # data.isna().mean().sort_values(ascending=False)  # процент пропусків за кожною колонкою.
        # data = data.dropna()

#%%
# визначенням типів змінних
data.info()

# %%
# визначення кількості унікальних значень для категоріальних змінних
data.select_dtypes(include='object').nunique()

# %%
# категоріальні ознаки
data.select_dtypes(include='object').apply(lambda x: x.unique()[:5])

# %%
# Огляд розподілів
melted = data.select_dtypes(include='number').melt()

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

#%%

from scipy.stats import skew

# Обчислення коефіцієнта асиметрії
data_num = data.select_dtypes(include='number')

for col in data_num.columns:
    skew_value = skew(data_num[col], nan_policy='omit')  # обробка пропущених значень
    print(f"{col:<18}: {skew_value:>5.2f}")
    
# Асиметрія показує, наскільки розподіл числової змінної є симетричним відносно середнього значення.
# 0 = симетрія, до 0.5 низька асиметрія, до 1 - помірна, більше 1 - висока.


# %%
# 2. Визначте перелік дискретних ознак (в широкому розумінні) для подальшого розрахунку показника взаємної інформації.

# Виконаємо розрахунок МІ на прикладі набору даних Autos. (використаємо метод mutual_info_regression() пакета sklearn, 
# оскільки в ньому цільова змінна (price) є неперервною числовою величиною.)
# Особливістю реалізації методу є необхідність явного визначення "дискретних" ознак у наборі даних.
# "дискретні" тут значить більш точні "категоріальні"  Наприклад, інтенсивність пікселів на зображенні є дискретною (але не категоріальною). 
# Аналогічно, у наборі даних Autos числові ознаки, такі як num_of_doors і num_of_cylinders, дискретні (автомобіль не може мати 4.5 двері).

# Щодо інших явних категоріальних ознак, спочатку ми їх перетворюємо на числові шляхом надання унікальним категоріям довільних 
# числових міток за допомогою методу factorize(), а потім зазначаємо в переліку дискретних для подальшого використання в алгоритмі.

X = data.copy()
y = X.pop('price')

cat_features = X.select_dtypes('object').columns

for colname in cat_features:
    X[colname], _ = X[colname].factorize()

# %%

# Розраховані показники MI перетворюємо на об’єкт Series і сортуємо:  
mi_scores = mutual_info_regression(X, y,
    discrete_features=X.columns.isin(
        cat_features.to_list() +
        ['num_of_doors',
         'num_of_cylinders']),
    random_state=42)

mi_scores = (pd.Series( 
    mi_scores,
    name='MI Scores',
    index=X.columns)
    .sort_values())

mi_scores.sample(5)


# %%

discrete_features=X.columns.isin(
    cat_features.to_list() +
    ['num_of_doors',
     'num_of_cylinders'])

# Розраховані показники MI перетворюємо на об’єкт Series і сортуємо:  
mi_scores = mutual_info_regression(X, y,
    discrete_features=discrete_features,
    random_state=42)

mi_scores = (pd.Series( 
    mi_scores,
    name='MI Scores',
    index=X.columns)
    .sort_values())

mi_scores.sample(5)

# %%
# Тепер візуалізуємо отримані результати: 
plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')

plt.show()

# %%
# Тепер, коли в нас є "рейтинг корисності" вхідних ознак за метрикою MI, ми можемо розглянути деякі з них більш докладно.
# Наприклад, curb_weight (споряджена маса автомобіля) є лідером цього рейтингу. Це означає, що через спостереження за цією ознакою 
# можна отримати (відносно) “багато інформації” про цільову змінну, тому що між цими двома величинами існує значна взаємна залежність.

# Саме таку чітку залежність (у цьому випадку нелінійну) ми й бачимо на графіку цих двох змінних:

sns.regplot(data=data, x='curb_weight', y='price', order=2)

plt.show()

# %%

# З іншого боку, ознака fuel_type має досить низький показник MI, проте, згідно з наведеним нижче рисунком, 
# вона чітко розділяє дві групи з різними ціновими трендами за показником horsepower.

# Як відомо, дизельне паливо має більшу щільність за бензин (в кожному літрі більше енергії). Тому за однакової потужності 
# дизельні двигуни більш вигідні. Цим можна пояснити динаміку цін - дизельні автомобілі мають тенденцію коштувати дорожче.

sns.lmplot(data=autos,
           x='horsepower',
           y='price',
           hue='fuel_type',
           facet_kws={'legend_out': False})

plt.show()

# Це ілюструє, що ознака може залишатися корисною в контексті взаємодії з іншими ознаками, незважаючи на низький показник 
# взаємної інформації (із цільовою змінною). Побачимо в рамках домашнього завдання. 
# Часто наявність знань у відповідній предметній галузі може допомогти виявити такі приклади ефективної взаємодії між ознаками.



#%% 
# 3. Розрахуйте показник взаємної інформації для вхідних ознак і цільової змінної price за допомогою методу mutual_info_regression() з пакета sklearn.
target = autos.pop("price")

mis = mutual_info_regression(autos.reshape(-1, 1), target)[0]
# cor = np.corrcoef(x, y)[0, 1]

print(f'MI score: {mis:.2f} | Cor index: {cor:.2f}')

# %%
# 4. Побудуйте регресійну модель / ансамбль (наприклад, за допомогою обєкта RandomForestRegressorабоGradientBoostingRegressorз пакетаsklearn`) для ефективної оцінки важливості вхідних ознак, подібно до того, як ми це робили у темі «Дерева рішень. Важливість ознак в моделі» в розділі «Важливість ознак у моделі».
    # Для побудови моделі потрібно коректно виконати кодування категоріальних / дискретних ознак в наборі даних.
    
# Розділяємо дані на тренувальну та тестову вибірки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
# Ініціалізація та навчання моделі RandomForestRegressor
rfr_model = RandomForestRegressor(random_state=42)
rfr_model.fit(X, y)

# Отримання важливості ознак з GradientBoostingRegressor
rfr_feature_importances = pd.Series(
    rfr_model.feature_importances_,
    name='RFR Feature Importances',
    index=X.columns
).sort_values()

# Вибірка 5 випадкових результатів для демонстрації
rfr_feature_importances.sample(5)

#%% 
import numpy as np
import matplotlib.pyplot as plt

# Довжина даних для візуалізації
indices = np.arange(len(mi_scores))

# Візуалізація результатів
plt.figure(figsize=(8, 8))

# Взаємна інформація (mi_scores)
plt.barh(indices + 0.2, mi_scores, height=0.4, label='Mutual Information Scores')

# Важливості ознак з Random Forest (rfr_feature_importances)
plt.barh(indices - 0.2, rfr_feature_importances, height=0.4, color='red', label='RF Feature Importances')

# Підпис осей
plt.yticks(indices, mi_scores.index)
plt.title('Comparison of Mutual Information Scores and RF Feature Importances')

# Легенда
plt.legend()

# Відображення графіку
plt.show()

#%%
# Ініціалізація та навчання моделі GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X, y)

# Отримання важливості ознак з GradientBoostingRegressor
gbr_feature_importances = pd.Series(
    gbr_model.feature_importances_,
    name='GBR Feature Importances',
    index=X.columns
).sort_values()

# Вибірка 5 випадкових результатів для демонстрації
gbr_feature_importances.sample(5)

#%% 





# # Важливість ознак для RandomForest
# rf_feature_importances = rf_model.feature_importances_

# # Важливість ознак для GradientBoosting
# gb_feature_importances = gb_model.feature_importances_



# # Виведення важливості ознак
# print("Важливість ознак (RandomForest):")
# for feature, importance in zip(X.columns, rf_feature_importances):
#     print(f"{feature}: {importance:.4f}")

# print("\nВажливість ознак (GradientBoosting):")
# for feature, importance in zip(X.columns, gb_feature_importances):
#     print(f"{feature}: {importance:.4f}")
    
    
#%% 
import numpy as np
import matplotlib.pyplot as plt

# Довжина даних для візуалізації
indices = np.arange(len(mi_scores))

# Візуалізація результатів
plt.figure(figsize=(8, 8))

# Взаємна інформація (mi_scores)
plt.barh(indices + 0.2, mi_scores, height=0.4, label='Mutual Information Scores')

# Важливості ознак з Random Forest (gbr_feature_importances)
plt.barh(indices - 0.2, gbr_feature_importances, height=0.4, color='brown', label='Gradient Boosting Feature Importances')

# Підпис осей
plt.yticks(indices, mi_scores.index)
plt.title('Comparison of Mutual Information Scores and Gradient Boosting Feature Importances')

# Легенда
plt.legend()

# Відображення графіку
plt.show()


# %%
# 5. Масштабуйте / уніфікуйте різні за своєю природою показники взаємної інформації та важливості ознак у моделі за допомогою методу .rank(pct=True) об’єкта DataFrame з пакета pandas.


# Деталі щодо використання методу дивіться на сторінці документації.

# %%
# 6. Побудуйте візуалізацію типу grouped barsplots для порівняння обох наборів за допомогою методу catplot() з пакета seaborn.

# 💡 Для цього може знадобитися відповідним чином переформатувати дані за допомогою методу .melt() з пакета pandas.

# %%
#7. Проаналізуйте візуалізацію та зробіть висновки.




# %%

# Математичні перетворення включають у себе нормалізацію (масштабування) ознак та їх трансформацію за допомогою степенів або логарифмів. 
# Ці прийоми ми вже використовували (наприклад, для корекції асиметрії розподілу ознак у темі “Баєсівська класифікація. Метод kNN”.) 

# Тут розглянемо розподіл швидкості вітру в наборі даних про аварії в США (датасет Accidents), який є дуже асиметричним. 
# Просте використання логарифма допоможе змінити розподіл цієї ознаки, зробивши її набагато “інформативнішою” для машинного навчання:

accidents = datasets['accidents']

accidents['LogWindSpeed'] = accidents['WindSpeed'].apply(np.log1p)

sns.set_theme()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1])

plt.show()

# %%
# Прийоми генерації ознак. Підрахунок кількості
# У наборах може бути низка однотипних ознак, які вказуватимуть на наявність або відсутність в об’єкта певних 
# (поєднуваних або взаємовиключних) характеристик.

# Наприклад, у наборі, де захворювання описуються через наявність або відсутність у пацієнта характерних симптомів. 
# Для покращення роботи моделі можна створити нову змінну, яка рахуватиме загальну кількість ознак, наявних в об'єкта, з певного переліку.
# Такі ознаки зазвичай мають бінарний (1/0) або булевий (True/False) тип. У Python можна обробляти булеві значення так само, як і цілі числа. 
# Наприклад, набір даних Accidents містить ознаки, які вказують, чи був на дорозі поруч із місцем аварії певний дорожній знак. 
# Щоб підрахувати загальну кількість знаків поблизу місця аварії, можна скористатися методом sum():

roadway_features = ['Amenity',
                    'Bump',
                    'Crossing',
                    'GiveWay',
                    'Junction',
                    'NoExit',
                    'Railway',
                    'Roundabout',
                    'Station',
                    'Stop',
                    'TrafficCalming',
                    'TrafficSignal']

accidents['RoadwayFeatures'] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ['RoadwayFeatures']].head(10)

# %%
# Також можна використовувати вбудовані методи Pandas для створення булевих ознак на основі числових значень.
# Наприклад, набір даних про бетон (датасет Concrete) містить відомості про вміст визначених компонентів у різних рецептурних формулах бетону. 
# Серед іншого, формули відрізняються відсутністю одного чи декількох компонентів (тобто їх вміст дорівнює 0).
# Попередньо перетворивши числові ознаки на булеві за допомогою методу gt() ("greater than”), 
# можна підрахувати загальну кількість компонентів у формулі.

concrete = datasets['concrete']

components = ['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']

concrete['Components'] = concrete[components].gt(0).sum(axis=1)

concrete[components + ['Components']].head(10)

# Ознаки, створені шляхом підрахунку кількості, можуть стати особливо корисними для моделей/ансамблів на основі “дерев рішень”, 
# оскільки цей алгоритм не має “вбудованого” механізму для одночасного агрегування інформації за кількома ознаками.

# %%
# Часто може бути доцільним розбити ознаки з текстовими рядками на окремі частини для отримання додаткової потенційно корисної інформації.
#  Ось кілька типових прикладів текстових ознак:
    # - Ідентифікаційні номери: '123-45-6789'.
    # - Номери телефонів: '(ХХХ) 987-65-43'.
    # - Поштові адреси: '1234 Unknown St, Unnamed City, NV'.
    # Штрих-коди товарів: '0 36000 29145 2'.
    # Дата і час: 'Mon Sep 30 07:06:05 2013'.
# Зазвичай такі дані мають певну структуру, розуміння якої можна використати для створення нових ознак. 
# Наприклад, у номерах телефонів перші цифри (XXX) - код міста, що може допомогти визначити місцезнаходження абонента. 
# split() (доступний у Pandas через аксесор .str) текст легко розбивати на компоненти безпосередньо в колонках. 
# Наприклад, у наборі даних про клієнтів (датасет Сustomer), можна розділити ознаку Policy (страховий поліс) на типи й рівні покриття:

customer = datasets['customer']

customer[['Type', 'Level']] = (
    customer['Policy']
    .str
    .split(' ', expand=True))

customer[['Policy', 'Type', 'Level']].head(10)

# %%
# Подібний прийом ми використовували для розбиття ознаки Date на окремі змінні для позначення року й місяця в темі “Логістична регресія”.
# Так само можемо об'єднати окремі ознаки в нову комбіновану ознаку, якщо вважаємо, що це краще відображатиме неявні зв’язки між змінними.
# Повернемося до датасету Autos і продемонструємо це на прикладі ознак make і body_style:

autos['make_and_style'] = autos['make'] + '_' + autos['body_style']
autos[['make', 'body_style', 'make_and_style']].head()

# Моделі/ансамблі на основі “дерев рішень” можуть ефективно прогнозувати за допомогою майже будь-якої комбінації ознак.
# Однак, якщо конкретна комбінація ознак є особливо важливою, її явне створення й додавання в модель може покращити
# результати прогнозування, особливо за умови навчання моделі на (відносно) невеликих наборах даних.

# %%
# Групові перетворення дозволяють нам об'єднувати (агрегувати) інформацію за декількома об'єктами, які належать до однієї категорії. 
# Це відкриває можливості для створення таких нових похідних ознак, як, наприклад, "середній рівень доходу за місцем проживання".

# Якщо ми припускаємо, що між об'єктами різних категорій існують певні неявні зв'язки, то групові перетворення можуть стати 
# зручним інструментом як для подальшого дослідження таких зв'язків, так і для їх явного використання в моделі.

# метод groupby() і його використання для розрахунку ознаки AverageIncome на прикладі датасету Сustomer:

customer['AverageIncome'] = (customer
                             .groupby('State')['Income']
                             .transform('mean'))

ratio = customer["Income"] / customer["AverageIncome"]
ratio.hist()

customer[['State', 'Income', 'AverageIncome']].head(10)

# %%
# рупові перетворення можна використати для простого кодування категоріальних ознак, 
# наприклад, підрахувавши частоти їх зустрічання в наборі даних.

customer = (customer
            .assign(StateFreq=lambda x:
                    x.groupby('State')['State']
                    .transform('count') /
                    x['State'].count()))

customer[['State', 'StateFreq']].head(10)

# %%

# На практиці, щоб уникнути витоку даних, рекомендується проводити групові перетворення/розрахунки на тренувальному наборі даних, 
# а потім використовувати отримані результати для кодування ознак у тестовому наборі.

# Для демонстрації цього прийому спочатку розділимо набір даних Customer на тренувальну (c_train) і тестову (c_test) вибірки. 
# Потім за допомогою групування розрахуємо на тренувальній вибірці середній розмір страхового відшкодування (ClaimAmount) за різними 
# видами страхових договорів (Coverage). Далі об'єднаємо результати групового перетворення з тестовою вибіркою за полем Coverage, 
# реалізувавши таким чином варіант кодування цієї ознаки за іншою змінною (ClaimAmount) із набору даних.

c_train = customer.sample(frac=0.75)
c_test = customer.drop(c_train.index)

c_train['AverageClaim'] = (c_train
                           .groupby('Coverage')['ClaimAmount']
                           .transform('mean'))

c_test = c_test.merge(
    c_train[['Coverage', 'AverageClaim']].drop_duplicates(),
    on='Coverage',
    how='left')

c_test[['Coverage', 'AverageClaim']].head(10)

# Зазначимо, що наведений перелік прийомів feature engineering не є вичерпним. 
# Наприклад, у темі "Лінійна регресія" ми стикалися з географічними ознаками, але далі не будемо розглядати.

# Крім того, поза межами курсу залишається розгляд методів вилучення ознак (feature engineering) за допомогою нейронних мереж 
# із неструктурованих даних, таких як текстові або фотоматеріали, які асоціюються з об’єктами в наборі даних. 
# Отримані за допомогою таких методів вектори додаткових ознак можна ефективно використовувати для навчання традиційних ML-алгоритмів.


# %%

x = np.linspace(0, 2, 50)
y = np.sin(2 * np.pi * 0.25 * x)

sns.regplot(x=x, y=y)

 # %%

mis = mutual_info_regression(x.reshape(-1, 1), y)[0]
cor = np.corrcoef(x, y)[0, 1]

print(f'MI score: {mis:.2f} | Cor index: {cor:.2f}')

# %%
# Відбір ознак. Взаємна інформація між ознаками й цільовою змінною
# Після отримання набору даних і його попереднього огляду (включно з перевіркою на наявність пропусків, визначенням типів змінних, 
    # оглядом розподілів та асиметрії числових змінних, визначенням кількості унікальних значень для категоріальних змінних тощо), 
    # варто спробувати оцінити "корисність" ознак у наборі для подальшого моделювання на підставі їх зв'язку з цільовою змінною. 
    # Відбір найбільш інформативних ознак полегшить проведення подальшого аналізу даних.

# Для оцінки зв'язку можна використати метрику, відому як "взаємна інформація" (MI — mutual information). 
# Вона подібна до кореляції, про яку ми згадували в темі “Дослідницький аналіз даних (EDA)”, але має перевагу в тому, 
# що може виявити будь-який тип зв'язку між змінними, а не лише лінійний.
# Ця метрика проста у використанні й інтерпретації, а також ефективна в обчисленні і має теоретичне обґрунтування. 
# MI вимірює зв'язок між величинами в термінах невизначеності, 
# тобто показує, наскільки знання однієї величини зменшує невизначеність (ентропію) щодо іншої.

# Для прикладу розглянемо взаємозв'язок між якістю зовнішнього оздоблення будинку й ціною на нього, взятий із набору даних Ames Housing. 
# Кожна точка на графіку представляє певний будинок. 
# Із графіка стає очевидним, що кожна категорія ExterQual концентрує ціну продажу в певному діапазоні.
# Значення взаємної інформації (MI), яке оцінює зв'язок між ExterQual і SalePrice, можна розрахувати як середньозважене зменшення 
# невизначеності в SalePrice, пов’язане з кожною з чотирьох категорій ExterQual. 
# Загалом це нагадує підхід до визначення важливості ознак у моделі, який ми розглядали в темі “Дерева рішень.Визначення ознак у моделі”.

# Інтерпретація показників MI: мінімальне значення = 0 свідчить про незалежність між величинами. 
# Хоча теоретично верхня межа MI не обмежена, на практиці значення, вищі за 2, зустрічаються рідко.

# Крім того, MI оцінює зв'язок ознаки з цільовою змінною окремо від інших ознак, тоді як деякі ознаки можуть бути більш
# інформативними для прогнозування, коли вони взаємодіють одна з одною. Тому важливо усвідомити, що практична "корисність" ознаки 
# залежить від здатності моделі ефективно опрацьовувати інформаційні сигнали від неї. 
# Посилити такі сигнали можна за допомогою розглянутих вище прийомів feature engineering.

# %%

# Виконаємо розрахунок МІ на прикладі набору даних Autos. (використаємо метод mutual_info_regression() пакета sklearn, 
# оскільки в ньому цільова змінна (price) є неперервною числовою величиною.)
# Особливістю реалізації методу є необхідність явного визначення "дискретних" ознак у наборі даних.
# "дискретні" тут значить більш точні "категоріальні"  Наприклад, інтенсивність пікселів на зображенні є дискретною (але не категоріальною). 
# Аналогічно, у наборі даних Autos числові ознаки, такі як num_of_doors і num_of_cylinders, дискретні (автомобіль не може мати 4.5 двері).

# Щодо інших явних категоріальних ознак, спочатку ми їх перетворюємо на числові шляхом надання унікальним категоріям довільних 
# числових міток за допомогою методу factorize(), а потім зазначаємо в переліку дискретних для подальшого використання в алгоритмі.

X = autos.copy()
y = X.pop('price')

cat_features = X.select_dtypes('object').columns

for colname in cat_features:
    X[colname], _ = X[colname].factorize()

# %%
# Розраховані показники MI перетворюємо на об’єкт Series і сортуємо:

    mi_scores = mutual_info_regression(
    X, y,
    discrete_features=X.columns.isin(
        cat_features.to_list() +
        ['num_of_doors',
         'num_of_cylinders']),
    random_state=42)

mi_scores = (pd.Series(
    mi_scores,
    name='MI Scores',
    index=X.columns)
    .sort_values())

mi_scores.sample(5)

# %%
# Тепер візуалізуємо отримані результати: 
plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')

plt.show()

# %%
# Тепер, коли в нас є "рейтинг корисності" вхідних ознак за метрикою MI, ми можемо розглянути деякі з них більш докладно.
# Наприклад, curb_weight (споряджена маса автомобіля) є лідером цього рейтингу. Це означає, що через спостереження за цією ознакою 
# можна отримати (відносно) “багато інформації” про цільову змінну, тому що між цими двома величинами існує значна взаємна залежність.

# Саме таку чітку залежність (у цьому випадку нелінійну) ми й бачимо на графіку цих двох змінних:

    sns.regplot(data=autos, x='curb_weight', y='price', order=2)

plt.show()

# %%

# З іншого боку, ознака fuel_type має досить низький показник MI, проте, згідно з наведеним нижче рисунком, 
# вона чітко розділяє дві групи з різними ціновими трендами за показником horsepower.

# Як відомо, дизельне паливо має більшу щільність за бензин (в кожному літрі більше енергії). Тому за однакової потужності 
# дизельні двигуни більш вигідні. Цим можна пояснити динаміку цін - дизельні автомобілі мають тенденцію коштувати дорожче.

sns.lmplot(data=autos,
           x='horsepower',
           y='price',
           hue='fuel_type',
           facet_kws={'legend_out': False})

plt.show()

# Це ілюструє, що ознака може залишатися корисною в контексті взаємодії з іншими ознаками, незважаючи на низький показник 
# взаємної інформації (із цільовою змінною). Побачимо в рамках домашнього завдання. 
# Часто наявність знань у відповідній предметній галузі може допомогти виявити такі приклади ефективної взаємодії між ознаками.