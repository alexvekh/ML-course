import time
import pandas as pd
from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.metrics import f1_score

# %%

# набір даних Employee містить анонімізовані відомості про працівників компанії (рівень освіти, рік працевлаштування, деякі інші демографічні дані та ознаки).
# Наша задача — побудувати низку бінарних класифікаторів, використовуючи методи ансамблювання моделей, і оцінити їх ефективність.

data = pd.read_csv('../datasets/mod_05_topic_09_employee_data.csv')
data.head()

# %%
# Перевірка типів даних і відсутніх значень
data.info()
# У наборі даних маємо числові й категоріальні ознаки, пропусків у даних немає. 
# Цільова змінна LeaveOrNot — це бінарна ознака того, чи залишив працівник компанію.
# %%
# JoiningYear показує рік, коли працівник приєднався до компанії. 
# Щоб ефективно використати цю ознаку в моделі, ми можемо розрахувати стаж роботи працівника в компанії на момент збору даних.
# Для цього ми можемо взяти максимальний рік, що зустрічається в наборі даних (це, очевидно, і буде рік створення цього датасету), 
# і відняти рік приєднання працівника до компанії.
data['JoiningYear'] = data['JoiningYear'].max() - data['JoiningYear']

# %%
# Ознака PaymentTier показує рівень заробітної плати працівника, але вона представлена не у вигляді абсолютних цифр, 
# а визначається категоріями в певній тарифній сітці. 
# У вхідному наборі даних ця ознака є числовою, але використання її в такому вигляді не є доцільним.
# Наприклад, ми не можемо стверджувати, що рівень оплати в категорії 1 втричі вищий, ніж у категорії 3, 
# хоча саме так будуть розуміти цю ознаку алгоритми машинного навчання, якщо ми залишимо її числовою.
# Перетворюємо ознаку (для подальшого кодування) на категоріальну методом .astype(str):
data['PaymentTier'] = data['PaymentTier'].astype(str)

# %%
# Практика застосування ансамблів. Підготовка й обробка даних
# Розбиття на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('LeaveOrNot', axis=1),
        data['LeaveOrNot'],
        test_size=0.33,
        random_state=42))

# %%
# Кодування категоріальних змінних (Використаємо метод TargetEncoder)
encoder = ce.TargetEncoder()

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %%
# Нормалізація змінних (Тепер, коли всі ознаки в наборі даних є числовими, виконаємо їх нормалізацію.)
scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Балансування класів 
# Перевіряємо кількість об’єктів у кожному з класів.
y_train.value_counts(normalize=True)
    # 0 - 0.657363, 1 - 0.342637

# %%
# Далі ми розглядатимемо широкий спектр базових моделей і методів їх об'єднання в ансамблі. 
# Ці моделі та ансамблі різняться за чутливістю до дисбалансу класів у даних і відсутністю/наявністю 
# відповідних гіперпараметрів для балансування класів за допомогою ваг безпосередньо під час навчання. 
# Тому, щоб забезпечити однакові умови для всіх алгоритмів, ми попередньо збалансуємо класи на етапі підготовки даних.

# Балансуємо тренувальну вибірку, додавши до неї згенеровані методом SMOTE об’єкти менш представленого класу:
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# %%

f1_scores = {}


def measure_f1_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        predictions = func(*args, **kwargs)
        end_time = time.time()
        f1 = f1_score(args[-1], predictions)
        model_name = args[0].__class__.__name__
        execution_time = end_time - start_time
        f1_scores[model_name] = [f1, execution_time]
        print(f'{model_name} F1 Metric: {f1:.4f}')
        print(f'{model_name} Inference: {execution_time:.4f} s')
        return predictions
    return wrapper


@measure_f1_time_decorator
def predict_with_measure(model, Xt, yt):
    return model.predict(Xt)


# %%

mod_log_reg = (LogisticRegression(
    # n_jobs=-1
).fit(X_res, y_res))

prd_log_reg = predict_with_measure(mod_log_reg, X_test, y_test)

# %%

mod_rnd_frs = (RandomForestClassifier(
    random_state=42,
    # n_jobs=-1
)
    .fit(X_res, y_res))

prd_rnd_frs = predict_with_measure(mod_rnd_frs, X_test, y_test)

# %%

mod_bag_knn = (BaggingClassifier(
    KNeighborsClassifier(),
    max_samples=0.75,
    max_features=0.75,
    # n_jobs=-1,
    random_state=42)
    .fit(X_res, y_res))

prd_bag_knn = predict_with_measure(mod_bag_knn, X_test, y_test)

# %%

mod_ada_bst = (AdaBoostClassifier(
    algorithm='SAMME',
    random_state=42)
    .fit(X_res, y_res))

prd_ada_bst = predict_with_measure(mod_ada_bst, X_test, y_test)

# %%

mod_grd_bst = (GradientBoostingClassifier(
    learning_rate=0.3,
    subsample=0.75,
    max_features='sqrt',
    random_state=42)
    .fit(X_res, y_res))

prd_grd_bst = predict_with_measure(mod_grd_bst, X_test, y_test)

# %%

clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = GaussianNB()

estimators = [('lnr', clf1),
              ('knn', clf2),
              ('gnb', clf3)]

mod_vot_clf = VotingClassifier(
    estimators=estimators,
    voting='soft').fit(X_res, y_res)

prd_vot_clf = predict_with_measure(mod_vot_clf, X_test, y_test)

# %%

final_estimator = GradientBoostingClassifier(
    subsample=0.75,
    max_features='sqrt',
    random_state=42)

mod_stk_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator).fit(X_res, y_res)

prd_stk_clf = predict_with_measure(mod_stk_clf, X_test, y_test)

# %%

scores = pd.DataFrame.from_dict(
    f1_scores,
    orient='index',
    columns=['f1', 'time'])

scores.sort_values('f1', ascending=False)
