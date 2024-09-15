import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
import statsmodels.api as sm
from scipy.stats import zscore
from prophet import Prophet

# %%
# EDA датасету Peyton Manning

df = pd.read_csv('../datasets/mod_02_topic_04_ts_data.csv')
df.head()

# %%
#приведення даних до типу числового ряду
df['ds'] = pd.to_datetime(df['ds'])
df = df.set_index('ds').squeeze()

# %%

df.describe()
# Як бачимо, наші спостереження (цільова змінна) є асиметричними й вимірюються в діапазоні від 628 до 319190.
# %%
# У такому випадку звичайною практикою є логарифмування вхідних даних з метою наближення їх розподілу до нормального.
# (ефективно застосувати статистичні методи для видалення аномалій)
df = np.log(df)
df.head()

# %%

df_hist = df.iloc[:-365]    # Вибирає рядки з початку df до останніх 365
df_test = df.iloc[-365:]    # Вибирає останні 365 рядків з df

# %%
# Перевіряємо тренувальну вибірку на наявність пропущених значень.
df_hist.isna().sum()

# %%
# Візуалізуємо наш набір даних.
sns.set_theme()

fig, ax = plt.subplots(figsize=(30, 7))

ax.vlines(
    x=df_hist.index,
    ymin=0,
    ymax=df_hist,
    linewidth=0.5,
    color='grey')

plt.show()

# %%
# Відновлення відсутніх значень (пропущені значення, прогалини в часових проміжках)
df_hist = df_hist.asfreq('D').interpolate()
df_hist.isna().sum()

# %%
# Декомпозиція: видалення тренду за допомогою лінійної регресії

# Побудуємо просту лінійну регресію, де вхідною ознакою буде 
# порядковий номер спостереження в наборі даних, 
# а цільовою змінною — значення часового ряду.
model = LinearRegression().fit(np.arange(len(df_hist)).reshape(-1, 1), df_hist)
trend = model.predict(np.arange(len(df_hist)).reshape(-1, 1))

ax = plt.subplots(figsize=(10, 3))
sns.scatterplot(df_hist)
sns.lineplot(y=trend, x=df_hist.index, c='black')

plt.show()

# %%

# Цей тип графіку дасть нам уявлення про характер розподілу даних (середні значення та розмах) у кожному місяці.
df_mod = df_hist - trend + trend.mean()

sns.catplot(
    y=df_hist,
    x=df_hist.index.month,
    kind='box',
    showfliers=False)

plt.show()
# Таким чином, ми спостерігаємо, що протягом перших місяців року сторінка відвідується більшою кількістю користувачів, а потім ця кількість поступово зменшується до середини року і знову зростає у 4-му кварталі.
# %%
# Декомпозиція: використання пакета statsmodels
# Декомпозицію часового ряду, тобто виділення з ряду основних компонент 
# (тренду, сезонності й залишків), можна ефективно здійснити 
# за допомогою методу seasonal_decompose пакета statsmodels:

decomp = sm.tsa.seasonal_decompose(df_hist)
decomp_plot = decomp.plot()
# В результаті отримуємо об'єкт, атрибутами якого (.trend, .seasonal, .resid) 
# і будуть компоненти нашого часового ряду. 
# Фактично decomp.resid є нашим вхідним рядом, приведеним до стаціонарного вигляду. 
# Тепер ми можемо продовжити його аналіз та підготовку для подальшого прогнозування.

# %%
# Очистка від викидів 
# (значень, що суттєво відрізняються від інших - за правилом “трьох сигм”)
df_zscore = zscore(decomp.resid, nan_policy='omit')


# %%
# або методом ковзних вікон (методом rolling() об'єктів Series і DataFrame.)
# відхилення не від середнього від всіх, а від середнього за певний період.
def zscore_adv(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z
df_zscore_adv = zscore_adv(decomp.resid, window=7)

# %%
# порівння z-критеріїв по всьому набору (верхній графік) і ковзного вікна 
# (нижній графік). Використання ковзного вікна тут краще. (за межами -3,3 = викиди)


fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(10, 7))

for i, d in enumerate([df_zscore, df_zscore_adv]):
    ax = axes[i]
    sns.lineplot(d, ax=ax)
    ax.fill_between(d.index.values, -3, 3, alpha=0.15)

plt.show()

# однак зростання кількості відвідувань може бути пов'язане з виступами 
# цього футболіста на важливих іграх.
# %%
# календар подій (ігор за участю футболіста) для історичного періоду та для 
# тестового періоду (який використовуватимемо для прогнозування майбутнього).
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2013-01-12',
                        '2014-01-12',
                          '2014-01-19',
                          '2014-02-02',
                          '2015-01-11',
                          '2016-01-17']),
    'lower_window': 0,
    'upper_window': 1})

superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2014-02-02']),
    'lower_window': 0,
    'upper_window': 1})

holidays = pd.concat((playoffs, superbowls)).reset_index(drop=True)

holidays

# %%
# Отже, тепер ми визнаємо "викидами" лише ті z-критеріїб що поза межами [-3, 3]
# і не випадають на дати важливих ігор
# відобразимо ці викиди на графіку
outliers = np.where(~df_zscore_adv.between(-3, 3) * df_zscore_adv.notna())[0]

outliers = list(set(df_hist.index[outliers]).difference(holidays['ds']))

fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(df_hist, ax=ax)
sns.scatterplot(
    x=outliers,
    y=df_hist[outliers],
    color='red',
    ax=ax)

plt.show()

# %%
# видалимо визначені викиди з набору даних і відновимо їх значення 
# за допомогою інтерполяції, як це робили раніше із пропусками.
df_hist.loc[outliers] = np.nan
df_hist = df_hist.interpolate()

# %%  
# аналіз і прогнозування часового ряду. Навчання і оцінка моделі

# перезапуск індексації
df_hist = df_hist.reset_index()

# %%
# Процес навчання моделі починається зі створення нового екземпляра 
# класу Prophet. Як додатковий параметр при створенні нового об’єкта 
# використовуємо датафрейм із календарем подій, які, на нашу думку, 
# впливали на динаміку ряду в минулому, і, очевидно, будуть мати місце 
# (заплановані) в майбутньому.
# Додатково є можливість вручну налаштувати параметри, які визначатимуть 
# сезонність, зробивши цю компоненту більш згладженою.
# Далі викликаємо метод fit, передаючи йому датафрейм з історичними даними.

mp = Prophet(holidays=holidays)
mp.add_seasonality(name='yearly', period=365, fourier_order=2)
mp.fit(df_hist)

# %%
# Прогнозування формується на основі таблиці з майбутніми датами, 
# для яких потрібно отримати значення часового ряду. 
# Зручно побудувати таку таблицю за допомогою методу Prophet.make_future_dataframe.
# За замовчуванням ця таблиця також включатиме історичні дати. 
# Метод predict присвоює кожному рядку в майбутньому прогнозоване значення, 
# яке позначається як yhat.

future = mp.make_future_dataframe(freq='D', periods=365)
forecast = mp.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# %%
# Для візуального дослідження компонентів ряду використовуємо метод Prophet.plot_components()

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    mp.plot_components(forecast)
    mp.plot(forecast)

# На першому графіку бачимо історичні дані (чорні точки), а також те, 
# як модель під них адаптувалась, та її прогноз.
# На наступних графіках виділено окремо компоненту тренду, 
# щорічну та тижневу сезонність часового ряду. 
# Оскільки ми включили календар, ми також побачимо компоненту, 
# що відповідає за корегування значень ряду в дати з календаря.
# %%
# Оцінювання точності моделі
# Отримавши прогнозні значення для цільової змінної на 365 днів вперед, 
# ми можемо візуалізувати їх на графіку разом із тестовими значеннями 
# (які не використовувалися під час навчання моделі). 
# Це дозволяє візуально оцінити точність моделі.

pred = forecast.iloc[-365:][['ds', 'yhat']]

fig, ax = plt.subplots(figsize=(20, 5))

ax.vlines(
    x=df_test.index,
    ymin=5,
    ymax=df_test,
    linewidth=0.75,
    label='fact',
    zorder=1)

ax.vlines(
    x=df_test[df_test.index.isin(holidays['ds'])].index,
    ymin=5,
    ymax=df_test[df_test.index.isin(holidays['ds'])],
    linewidth=0.75,
    color='red',
    label='special events',
    zorder=2)

sns.lineplot(data=pred, y='yhat', x='ds', c='black', label='prophet', ax=ax)

ax.margins(x=0.01)

plt.show()

# %%
# метрики точності.
approx_mape = median_absolute_error(df_test, pred['yhat'])

print(f'Accuracy: {1 - approx_mape:.1%}')
