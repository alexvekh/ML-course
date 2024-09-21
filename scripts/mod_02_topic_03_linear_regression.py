import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# %%
# застосування лінійної регресії до набору даних, відомого як "Житловий фонд Каліфорнії". 
# Цей набір даних містить інформацію про різні округи Каліфорнії, включаючи демографічні показники, місцезнаходження та характеристики будинків.

# завантажимо дані та розглянемо їх опис, щоб зрозуміти, які ознаки доступні для використання в моделі.
california_housing = fetch_california_housing(as_frame=True)

data = california_housing['frame']
data.head()

# опис ознак у наборі даних:
#   MedInc: cередній дохід населення відповідного кварталу (блоку будинків) у місті.
#   HouseAge: середній вік будинку.
#   AveRooms: середня кількість кімнат у будинку.
#   AveBedrms: середня кількість спалень у будинку.
#   Population: кількість населення кварталу.
#   AveOccup: середня кількість зайнятих членів домогосподарства.
#   Latitude: географічна широта (центральної точки) кварталу.
#   Longitude: географічна довгота (центральної точки) кварталу.
# %%
# Наша мета — прогнозування вартості будинку, отже, це — задача регресії. 
# Відповідно, наша цільова мітка — неперервна (числова) величина.
target = data.pop('MedHouseVal')
target.head()

# %%
# Перевіримо датасет на наявність пропущених значень і познайомимось із типами даних.
data.info()
# Пропущених значень нема, усі ознаки — числові.

# %%
# Побудуємо гістограми ознак, щоб зрозуміти їх розподіл та ідентифікувати можливі викиди.
sns.set_theme()

melted = pd.concat([data, target], axis=1).melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# З аналізу розподілу ознак видно, що розподіл доходу має довгий “хвіст”, вказуючи на наявність осіб із високими доходами. 
# Водночас розподіл середнього віку будинку є більш-менш рівномірним.
# Однак розподіл цільової змінної (MedHouseVal) має довгий “хвіст”: всі будинки з ціною вище або рівною 5 отримують однакове значення 5 
# (тобто відсутні ціни, вищі за “5”: у наборі даних є ціна 2.5, 3.7, 4.1 тощо, але нема більшої за 5.0).
# Деякі ознаки, такі як середня кількість кімнат, середня кількість спалень, населення та кількість мешканців, мають широкий діапазон 
# з високими значеннями, які потенційно можуть бути викидами (аномаліями).
# %%
# Переглянемо детальніше описові статистики цих показників:
features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
data[features_of_interest].describe()
# Можливо, після навчання “базової” моделі, повернемося до цих ознак, щоб провести додатковий аналіз і очистити від викидів. 
# Поки залишаємо як є.
# %%
# Вирішимо, чи будуть географічні ознаки корисними для прогнозування вартості будинку. 
# Для цього побудуємо графік із географічними координатами наших спостережень по осях х та у і кольором позначимо вартість будинків.
fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=data,
    x='Longitude',
    y='Latitude',
    size=target,
    hue=target,
    palette='viridis',
    alpha=0.5,
    ax=ax)

plt.legend(
    title='MedHouseVal',
    bbox_to_anchor=(1.05, 0.95),
    loc='upper left')

plt.title('Median house value depending of\n their spatial location')

plt.show()

# Зіставляючи нашу візуалізацію з географічною картою штату Каліфорнія, бачимо, що будинки з високою вартістю переважно 
# розташовані на узбережжі. Отже, географічні координати будуть суттєвою ознакою в нашій моделі.

# %%

# Будуємо матрицю кореляції вхідних змінних (крім географічних координат) між собою та з цільовою змінною.

columns_drop = ['Longitude', 'Latitude']
subset = pd.concat([data, target], axis=1).drop(columns=columns_drop)

corr_mtx = subset.corr()

mask_mtx = np.zeros_like(corr_mtx)
np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(subset.corr(),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)

plt.show()

# %%

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42)

# %%

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

X_train_scaled.describe()

# %%

model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)
y_pred.head()

# %%

r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# %%

pd.Series(model.coef_, index=X_train_scaled.columns)

# %%

# [a, b] -> [1, a, b, a^2, ab, b^2]
poly = PolynomialFeatures(2).set_output(transform='pandas')

Xtr = poly.fit_transform(X_train_scaled)
Xts = poly.transform(X_test_scaled)

model_upd = LinearRegression().fit(Xtr, y_train)
y_pred_upd = model_upd.predict(Xts)
y_pred_upd = pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)

r_sq_upd = model_upd.score(Xtr, y_train)
mae_upd = mean_absolute_error(y_test, y_pred_upd)
mape_upd = mean_absolute_percentage_error(y_test, y_pred_upd)

print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

# %%

pct_error = (y_pred_upd / y_test - 1).clip(-1, 1)

sns.scatterplot(
    x=y_test,
    y=pct_error,
    hue=pct_error.gt(0),
    alpha=0.5,
    s=10,
    legend=False)

plt.show()
