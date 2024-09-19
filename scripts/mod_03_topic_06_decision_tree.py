import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from prosphera.projector import Projector

# %%
# особливості побудови класифікатора на основі дерева рішень
# набір даних Національного інституту діабету, захворювань органів травлення та нирок.
# Задача класифікації — прогнозування наявності чи відсутності діабету в пацієнта. 
# Усі спостереження обмежуються даними про жінок віком від 21 року з індіанського племені Піма.

# Завантажимо дані та розглянемо їх, щоб зрозуміти, які ознаки доступні для використання в моделі.
data = pd.read_csv('../datasets/mod_03_topic_06_diabets_data.csv')
data.head()

# Отже, у наборі даних маємо такі ознаки:
 # Pregnancies: кількість попередніх вагітностей.
 # Glucose: концентрація глюкози в плазмі через 2 години після перорального тесту на толерантність до глюкози.
 # BloodPressure: діастолічний артеріальний тиск у мм рт. ст.
 # SkinThickness: Товщина складки шкіри трицепса в мм.
 # Insulin: Рівень інсуліну в сироватці крові через 2 години після прийому глюкози (в мОд/мл).
 # BMI (індекс маси тіла): Визначається як вага в кілограмах, поділена на квадрат висоти в метрах.
 # DiabetesPedigreeFunction: відображає схильність до діабету в сім'ї пацієнта.
 # Age: вік пацієнта в роках.
 # Outcome: цільова змінна.

# %%
# Перевірка типів даних і відсутніх значень

data.info()

# У наборі даних маємо всі числові ознаки, колонки з пропущеними значеннями відсутні. 
# Проте можна побачити, що для окремих ознак ('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI') 
# значення дорівнюють 0, що може вказувати на некоректність даних.

# Це може бути пов’язано із різними причинами, включаючи технічні помилки або відсутність вимірювання. 
# Тому ми спочатку замінимо 0 в цих колонках на np.nan, а пізніше відновимо ці пропуски за допомогою SimpleImputer.

# %%

# Припускаємо, що рівень глюкози та індекс маси тіла можуть бути важливими ознаками наявності діабету.
# Спробуємо візуалізувати цю гіпотезу й побудувати графік, який показує, як рівень глюкози 
# та індекс маси тіла пов'язані з ознакою діабету (цільовою змінною) в нашому наборі даних.

X, y = (data.drop('Outcome', axis=1), data['Outcome'])

cols = ['Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI']

X[cols] = X[cols].replace(0, np.nan)

# %%

ax = sns.scatterplot(x=X['Glucose'], y=X['BMI'], hue=y)
ax.vlines(x=[120, 160],
          ymin=0,
          ymax=X['BMI'].max(),
          color='black',
          linewidth=0.75)

plt.show()

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42)

# %%

imputer = SimpleImputer()

X_train[cols] = imputer.fit_transform(X_train[cols])
X_test[cols] = imputer.fit_transform(X_test[cols])

# %%

clf = (tree.DecisionTreeClassifier(
    random_state=42)
    .fit(X_train, y_train))

y_pred = clf.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred)

print(f'Acc.: {acc:.1%}')

# %%

plt.figure(figsize=(80, 15), dpi=196)

tree.plot_tree(clf,
               feature_names=X.columns,
               filled=True,
               fontsize=6,
               class_names=list(map(str, y_train.unique())),
               rounded=True)

plt.savefig('../derived/mod_03_topic_06_decision_tree.png')
plt.show()

# %%

y_train.value_counts(normalize=True)

# %%

sm = SMOTE(random_state=42, k_neighbors=15)
X_res, y_res = sm.fit_resample(X_train, y_train)

y_res.value_counts(normalize=True)

# %%

clf_upd = (tree.DecisionTreeClassifier(
    max_depth=5,
    random_state=42)
    .fit(X_res, y_res))

y_pred_upd = clf_upd.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred_upd)

print(f'Acc.: {acc:.1%}')

# %%

plt.figure(figsize=(30, 8))

tree.plot_tree(clf_upd,
               feature_names=X.columns,
               filled=True,
               fontsize=8,
               class_names=list(map(str, y_res.unique())),
               rounded=True)

plt.show()

# %%

(pd.Series(
    data=clf_upd.feature_importances_,
    index=X.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()

# %%

visualizer = Projector()
visualizer.project(data=X_train, labels=y_train)
