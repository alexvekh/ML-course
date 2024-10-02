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
# Практика застосування ансамблів. Навчання й оцінка ансамблів моделей
    # Створимо допоміжну функцію, яка автоматично розраховуватиме для нас час надання прогнозів на тестовій вибірці та F1. 
    # Результати будемо виводити (print) і зберігати у словнику f1_scores для подальшого порівняння ансамблів.
    # (у випадках, де важливими є швидкість реакції чи обробка великих обсягів даних, час отримання прогнозів може стати ключовим фактором.)
    # Наша допоміжна функція реалізована як декоратор, який приймає іншу функцію як аргумент і розширює її “функціональність”, не змінюючи код.
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
# Побудова базової моделі (baseline) точність якої потім порівняємо з точністю ансамблів (виберемо логістичну регресію):
mod_log_reg = (LogisticRegression(
    # n_jobs=-1
).fit(X_res, y_res))

prd_log_reg = predict_with_measure(mod_log_reg, X_test, y_test)

# %%
# Побудова RandomForestClassifier. Будуємо й оцінюємо ансамбль “глибоких” дерев рішень Random Forest:
mod_rnd_frs = (RandomForestClassifier(
    random_state=42,
    # n_jobs=-1
)
    .fit(X_res, y_res))

prd_rnd_frs = predict_with_measure(mod_rnd_frs, X_test, y_test)

# %%
# Побудова Bagging Classifier
# Будуємо й оцінюємо ансамбль моделей за принципом bagging, використовуючи алгоритм kNN. У нашому ансамблі кожна модель 
# навчатиметься на випадкових підмножинах, що складаються з 75% об'єктів та ознак тренувальної вибірки.
mod_bag_knn = (BaggingClassifier(
    KNeighborsClassifier(),
    max_samples=0.75,
    max_features=0.75,
    # n_jobs=-1,
    random_state=42)
    .fit(X_res, y_res))

prd_bag_knn = predict_with_measure(mod_bag_knn, X_test, y_test)

# %%

# Побудова AdaBoostClassifier. Будуємо й оцінюємо ансамбль моделей для класифікації за методом Ada Boost:
mod_ada_bst = (AdaBoostClassifier(
    algorithm='SAMME',
    random_state=42)
    .fit(X_res, y_res))

prd_ada_bst = predict_with_measure(mod_ada_bst, X_test, y_test)

# %%
# Побудова GradientBoostingClassifier. Створюємо та оцінюємо ансамбль моделей за методом Gradient Boost.
# задаємо гіперпараметри subsample=0.75 і max_features='sqrt':
    #- subsample: частка спостережень, на якій будуть навчатися базові моделі. 
        # Якщо частка менше 1.0, то алгоритм починає працювати за методом Stohastic Gradient Boost; 
    #- max_features: кількість ознак, які слід враховувати для пошуку найкращого розбиття при навчанні дерева рішень (базової моделі).
# Визначення параметрів 0.0 < subsample < 1.0 і max_features < n_features приводить до зменшення дисперсії прогнозів (low variance), 
# але потенційно може збільшити похибку моделі (high bias):

mod_grd_bst = (GradientBoostingClassifier(
    learning_rate=0.3,
    subsample=0.75,
    max_features='sqrt',
    random_state=42)
    .fit(X_res, y_res))

prd_grd_bst = predict_with_measure(mod_grd_bst, X_test, y_test)

# %%

# Побудова VotingClassifier (за принципом soft voting, використовуючи три моделі різних типів 
# з їхніми базовими налаштуваннями в пакеті sklearn (LogisticRegression, KNeighborsClassifier, GaussianNB)):
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

# Побудова StackingClassifier (за принципом stacking). Для цього ми використовуємо той самий набір базових моделей, 
# але додатково створюємо над ними метамодель типу GradientBoostingClassifier. (метамодель у нашому ансамблі сама є ансамблем моделей!)
final_estimator = GradientBoostingClassifier(
    subsample=0.75,
    max_features='sqrt',
    random_state=42)

mod_stk_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator).fit(X_res, y_res)

prd_stk_clf = predict_with_measure(mod_stk_clf, X_test, y_test)

# %%
# Порівняння ефективності ансамблів
# Перетворюємо словник f1_scores на таблицю, сортуємо й розглядаємо результати.

scores = pd.DataFrame.from_dict(  
    f1_scores,
    orient='index',
    columns=['f1', 'time'])

scores.sort_values('f1', ascending=False)

# З аналізу таблиці можна зробити висновок, що метод Stohastic Gradient Boost показав найкращий результат за метрикою F1. 
# Це свідчить про його високу ефективність у прогнозуванні на цьому наборі даних. Також слід зазначити, що всі ансамблі моделей 
# перевищили базову модель за ефективністю, що підкреслює доцільність використання ансамблів для покращення результатів прогнозування.

# Щодо часу отримання прогнозів, важливо враховувати його залежність від конкретних умов виконання скрипта, параметрів середовища, 
# і розміру тестової вибірки. Тому треба аналізувати не абсолютні значення часу, а порівнювати їх між собою.
# Наприклад, BaggingClassifier, який у нашому випадку використовує kNN-моделі, продемонстрував найбільший час видачі прогнозів, 
# оскільки обчислення відстаней між об’єктами в kNN відбуваються саме на етапі прогнозування.
# А GradientBoostingClassifier працює швидше за Random Forest, оскільки, серед іншого, використовує менш глибокі дерева рішень.