import pandas as pd
import numpy as np
from scipy.stats import zscore
import category_encoders as ce 
    # conda install category_encoders
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Завантаження даних
data = pd.read_csv('../datasets/mod_04_hw_train_data.csv')
data.head()

# %%
valid = pd.read_csv('../datasets/mod_04_hw_valid_data.csv')
valid.head()
# 
# %%
# процент пропусків за кожною колонкою.
data.isna().mean().sort_values(ascending=False)

# %%
# Оскільки пропусків тільки до 2% - ми їх видалимо
data = data.dropna()

#%%
# ----Зміна категоріальних ознак на числові покращення результату не дало 
# # Зміна 'Role' на int
# role_mapping = {'Junior': 1,'Mid': 2,'Senior': 3}
# data['Role'] = data['Role'].map(role_mapping).astype(int)

# # Зміна 'Qualification' на int
# qualification_mapping = {'Bsc': 1, 'Msc': 2, 'PhD': 3}
# data['Qualification'] = data['Qualification'].map(qualification_mapping).astype(int)

# # Зміна 'University' на int
# u_mapping = {'Tier1': 1, 'Tier2': 2, 'Tier3': 3}
# data['University'] = data['University'].map(u_mapping).astype(int)

# %%
# ----Неправдиві дані не дозволили нам використати вік з датасету (деяким людям по 3 роки)
# Одразу отримуємо вік з Date_Of_Birth
# data['Age'] = data['Date_Of_Birth'].apply(lambda x: 2024 - int(x.split('/')[2]))
# data = data.drop('Date_Of_Birth', axis=1)

# valid['Age'] = valid['Date_Of_Birth'].apply(lambda x: 2024 - int(x.split('/')[2]))
# valid = valid.drop('Date_Of_Birth', axis=1)


# %%
# Розподіли ознак

# melted = data.melt()

# g = sns.FacetGrid(melted,
#                   col='variable',
#                   col_wrap=4,
#                   sharex=False,
#                   sharey=False,
#                   aspect=1.25)

# g.map(sns.histplot, 'value')

# g.set_titles(col_template='{col_name}')

# g.tight_layout()

# plt.show()

# %%
y_train = data.pop("Salary")

# %%
# Розбиваємо на числові та категоріальні підмножини:
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%
# ----Відновлення пропущених значень не потрібно, оскільки ми їх видалили в рядку 33
# Відновлення пропущених числових значень
# num_imputer = SimpleImputer().set_output(transform='pandas')
# data_num = num_imputer.fit_transform(data_num)

# Відновлення пропущених категоріальних значень:
# cat_imputer = SimpleImputer(
#     strategy='most_frequent').set_output(transform='pandas')
# data_cat = cat_imputer.fit_transform(data_cat)


#%%
# ----Отримання коду з номеру телефону, який міг би вказати про регіон проживання не допомогло (майже всі різні)
# data_cat['Phone_Code'] = data_cat['Phone_Number'].apply(lambda x: x.split('-')[0])
#valid_cat = valid_cat.drop('Phone_Number'], axis=1)

# %%
# Нормалізація числових ознак за допомогою об'єкта StandardScaler або PowerTransformer з пакету sklearn
# scaler = StandardScaler().set_output(transform='pandas')
scaler = PowerTransformer().set_output(transform='pandas')
X_train_num = scaler.fit_transform(data_num)

# %%
# Спроба знайти залежність між категоріальними ознаками
from scipy.stats import chi2_contingency

# Функція для обчислення Chi-Square тесту
def chi_square_test(col1, col2):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    return p  # p-значення тесту

# Обчислення p-значення для кожної пари змінних
for col1 in data_cat.columns:
    for col2 in data_cat.columns:
        if col1 != col2:
            p_value = chi_square_test(data_cat[col1], data_cat[col2])
            print(f'P-значення для {col1} і {col2}: {p_value}')
# Chi-Square тест показує статистичну значущість залежності, де низьке p-значення (наприклад, менше 0.05) свідчить про наявність залежності між змінними.

# %%
# Категоріальні ознаки. 
# Видаляємо 'Phone_Number', "Name", "Date_Of_Birth" які всі різні і як категорії не підходять
# видалення "Qualification" підбрано,- найкраше покращує результат (з Qualification та University, які по Chi-Squareу залежні)
data_cat = data_cat.drop(['Phone_Number', "Name", "Date_Of_Birth", "Qualification"], axis=1)

# %%
# Кодування категоріальних ознак (наприклад, за допомогою об’єктів OneHotEncoder / TargetEncoder з пакета category_encoders).
encoder = ce.OneHotEncoder()
# encoder = ce.TargetEncoder()

X_train_cat = encoder.fit_transform(data_cat, y_train)

# %%
# 7. Об'єднання підмножини з числовими і категоріальними ознаками (після кодування) в одну:
X_train = pd.concat([X_train_num, X_train_cat], axis=1)


# %%

# Навчаємо модель і отримуємо прогнози:
  # class_weight='balanced' - невелюємо дисбаланс між позитивним і негативним класами (Ваги обчислюються автоматично під час тренування моделі.)
  # solver='lbfgs', liblinear, newton-cg, newton-cholesky, sag, saga (майже не впливають на результат)

model = (KNeighborsRegressor(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=12, p=1, metric='minkowski', metric_params=None, n_jobs=None)
       .fit(X_train, y_train))


# %%

# Обробка VALID dataset
y_true = valid.pop("Salary")

# # Зміна типу на int
# valid['Role'] = valid['Role'].map(role_mapping).astype(int)
# valid['Qualification'] = valid['Qualification'].map(qualification_mapping).astype(int)
# valid['University'] = valid['University'].map(u_mapping).astype(int)

# 3.2.Розбиваємо датасет на підмножини
valid_num = valid.select_dtypes(include=np.number)
valid_cat = valid.select_dtypes(include='object')

# Нормалізація числових, кодування категоріальних
X_test_num = scaler.transform(valid_num)

valid_cat = valid_cat.drop(['Phone_Number', "Name", "Date_Of_Birth", "Qualification"], axis=1)
X_test_cat = encoder.transform(valid_cat)

# 7. Об'єднання підмножини
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
X_train.shape


# %%
# Результат
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f'Validation MAPE: {mape:.2%}')
# Validation MAPE: 4.54%
# %%
print("y_pred:", y_pred)
print("y_true:", y_true.values)
