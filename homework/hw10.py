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
import category_encoders as ce 
from sklearn.preprocessing import PowerTransformer, StandardScaler

# %%
# 1. Завантаження набору даних Autos

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

data = datasets['autos']
data.info()


# %%
# пеевірка на на наявність пропусків
data.isna().sum()

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
print("                 Асиметрія:")
for col in data_num.columns:
    skew_value = skew(data_num[col], nan_policy='omit')  # обробка пропущених значень
    print(f"{col:<18} {skew_value:>5.2f}")
    
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

# scaler = StandardScaler().set_output(transform='pandas')
scaler = PowerTransformer().set_output(transform='pandas')
X = scaler.fit_transform(X)


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
# Тепер візуалізуємо отримані результати: 
plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')

plt.show()

# %%
# 4. Побудуйте регресійну модель / ансамбль (наприклад, за допомогою обєкта RandomForestRegressorабоGradientBoostingRegressorз пакетаsklearn`) для ефективної оцінки важливості вхідних ознак, подібно до того, як ми це робили у темі «Дерева рішень. Важливість ознак в моделі» в розділі «Важливість ознак у моделі».
# Для побудови моделі потрібно коректно виконати кодування категоріальних / дискретних ознак в наборі даних.

X = data.copy()
y = X.pop('price')

data_num = X.select_dtypes(include=np.number)
data_cat = X.select_dtypes(include='object')
data_cat['num_of_doors'] = data_num.pop('num_of_doors').astype(str)
data_cat['num_of_cylinders'] = data_num.pop('num_of_cylinders').astype(str)
    
# encoder = ce.OneHotEncoder()
encoder = ce.TargetEncoder()

# scaler = StandardScaler().set_output(transform='pandas')
scaler = PowerTransformer().set_output(transform='pandas')
data_num = scaler.fit_transform(data_num)

data_cat = encoder.fit_transform(data_cat, y)
# Об'єднання підмножини з числовими і категоріальними ознаками (після кодування) в одну:
X = pd.concat([data_num, data_cat], axis=1)


#%%
# Ініціалізація та навчання моделі RandomForestRegressor
rfr_model = RandomForestRegressor(random_state=42)
rfr_model.fit(X, y)

# Отримання важливості ознак з GradientBoostingRegressor
rfr_feature_importances = pd.Series(
    rfr_model.feature_importances_,
    name='RFR Feature Importances',
    index=X.columns
).sort_values(ascending=False)

# Вибірка 5 випадкових результатів для демонстрації
rfr_feature_importances.sample(5)


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
# 5. Масштабуйте / уніфікуйте різні за своєю природою показники взаємної інформації та важливості ознак у моделі за допомогою методу .rank(pct=True) об’єкта DataFrame з пакета pandas.

mi_ranks = mi_scores.rank(pct=True)
rf_ranks = rfr_feature_importances.rank(pct=True)
gb_ranks = gbr_feature_importances.rank(pct=True)

# Об'єднайте результати в один DataFrame для порівняння
results = pd.DataFrame({
    #'MI Scores': mi_scores,
    #'RF Importances': rfr_feature_importances,
    #'GB Importances': gbr_feature_importances,
    'MI Ranks': mi_ranks,
    'RF Ranks': rf_ranks,
    'GB Ranks': gb_ranks
})

results

#%% 

# %%

# Довжина даних для візуалізації
indices = np.arange(len(mi_ranks))

# Візуалізація MI Scores з Random Forest
plt.figure(figsize=(5, 8))
plt.barh(indices + 0.2, mi_ranks, height=0.4, label='MI Scores')
plt.barh(indices - 0.2, rf_ranks[mi_ranks.index], height=0.4, color='orange', label='RF Scores')
plt.yticks(indices, mi_ranks.index)
plt.title('Comparison of Mutual Information Scores and Random Forest Feature Importances')
plt.legend()
plt.show()
#%% 

# Візуалізація MI Scores з Boosting Feature
plt.figure(figsize=(5, 8))
plt.barh(indices + 0.2, mi_ranks, height=0.4, label='MI Scores')
plt.barh(indices - 0.2, gb_ranks[mi_ranks.index], height=0.4, label='GB Scores')
plt.yticks(indices, mi_ranks.index)
plt.title('Comparison of Mutual Information Scores and Gradient Boosting Feature Importances')
plt.legend()
plt.show()

#%% 

# Візуалізація результатів MI Scores з Random Forest та Boosting Feature
plt.figure(figsize=(5, 8))
plt.barh(indices + 0.3, mi_ranks, height=0.3, label='Mutual Information Scores')
plt.barh(indices, rf_ranks[mi_ranks.index], height=0.3, color='orange', label='Random Forest Rank')
plt.barh(indices - 0.3, gb_ranks[mi_ranks.index], height=0.3, label='Gradient Boosting Rank')
plt.yticks(indices, mi_ranks.index)
plt.title('Comparison of Mutual Information Scores and Gradient Boosting Feature Importances')
plt.legend()
plt.show()

#%%
# 6. Побудуйте візуалізацію типу grouped barsplots для порівняння обох наборів за допомогою методу catplot() з пакета seaborn.

# Сортування за MI Ranks
results = results.sort_values(by="MI Ranks", ascending=False)

# Перетворення в довгий формат
results_melted = results.reset_index().melt(id_vars='index', var_name='Rank Type', value_name='Rank')
results_melted = results_melted.rename(columns={'index': 'Feature'})

# Візуалізація (горизонтальні бари)
g = sns.catplot(
    data=results_melted, kind="bar",
    y="Feature", x="Rank", hue="Rank Type",
    errorbar="sd", palette="dark", alpha=.6, height=8
)
g.despine(left=False)
g.set_axis_labels("Rank", "")
g.legend.set_title("")

plt.title('Feature Importance Ranks Sorted by MI Ranks')
plt.show()


# %%
#7. Проаналізуйте візуалізацію та зробіть висновки.

# Важливість ознак, що стосуються характеристик двигуна:
# Найвищий вплив на цільову змінну за всіма методами показали ознаки engine_size і curb_weight, які стабільно посідають найвищі місця. Наприклад, за Gradient Boosting, engine_size має ранг 1.0, а curb_weight — 0.91.

# Fuel System та Highway MPG:
# fuel_system отримав високу оцінку у Gradient Boosting (0.65), що може вказувати на важливий внесок у результат моделі. highway_mpg також має високі показники в усіх методах, що підкреслює його значення у всіх підходах.

# Body Style та Make:
# Ці ознаки мають різні оцінки в різних підходах. Наприклад, у випадкових лісах ознака make отримала один з найвищих рангів (0.91), у той час як для MI вона займає середні позиції (0.70). Це свідчить про те, що різні методи по-різному оцінюють внесок цієї ознаки.

# Engine Location та Fuel Type:
# Ці ознаки мають низькі ранги у всіх методах. Наприклад, engine_location та fuel_type отримали значення близько 0.1 за всіма підходами, що свідчить про їхній низький вплив на прогнозовану змінну.

# Загальні висновки:

# Загалом, між методами є узгодженість щодо важливих ознак, таких як engine_size, curb_weight, та horsepower, які мають високі ранги у всіх підходах. Це може вказувати на їхню стабільну важливість для прогнозування.
# Проте деякі ознаки, такі як body_style і make, мають більшу варіативність у важливості залежно від методу, що вказує на їх потенційну залежність від специфіки методу моделювання.
# Цей аналіз показує, що для більш точного моделювання важливо враховувати кілька методів оцінки важливості ознак і можливі варіації у їхніх значеннях.



results['Average'] = results.mean(axis=1)
# Сортуємо дані по колонці 'Average'
sorted_results = results.sort_values(by='Average')

# Тепер візуалізуємо результати по колонці 'Average'
plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(sorted_results)), sorted_results['Average'])
plt.yticks(np.arange(len(sorted_results)), sorted_results.index)
plt.title('Average Feature Importance Scores')
print(sorted_results['Average'])
# Показуємо графік
plt.show()

