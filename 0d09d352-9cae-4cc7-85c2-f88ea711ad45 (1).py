#!/usr/bin/env python
# coding: utf-8

# # Промышленность

# Чтобы оптимизировать производственные расходы, металлургический комбинат «Стальная птица» решил уменьшить потребление электроэнергии на этапе обработки стали. Для этого комбинату нужно контролировать температуру сплава. Из этого следуют цель исследования и ее ход.

# **Цель исследования:**
#  -   Построить модель, которая будет предсказывать температуру сплава для уменьшения потребления электроэнергии на этапе обработки стали.
#         
#       **Ход исследования:**
#        
#       Считаю 7 CSV-файлов с данными об электродах, подаче сыпучих материалов (объём), подаче сыпучих материалов (время), продувке сплава газом, результатах  измерения температуры, проволочных материалах (объём) и проволочных материалах (время). Так как о данных ничего не известно, придется изучить общую информацию о данных.
#       
#       Далее я приступлю к исследовательскому анализу, при необходимости сделаю предобработку данных, построию и сделаю выводы о всех 7-ми датасетах. Объединю все датасеты в один по ключу и выполню исследовательский анализ данных общего датафрейма, при необходимости сделаю предобработку. Затем подготовлю общий датафрейм к обучению, то есть разделю его на две выборки, масштабирую и закодирую, и потом обучу как минимум две модели. В конце выберу лучшую модель и проверю ее качество на тестовой выборке, а также напишу общий вывод и рекомендации заказчику.
#       
#       Таким образом, моё исследование пройдет в восемь этапов:
#       
#       - Загрузка данных
#       - Анализ и предобработка данных
#       - Объединение данных
#       - Исследовательский анализ и предобработка данных объединённого датафрейма
#       - Подготовка данных
#       - Обучение моделей машинного обучения
#       - Выбор лучшей модели
#       - Общий вывод и рекомендации заказчику

# # 1. Загрузка данных

# In[1]:


get_ipython().system('pip install lightgbm -q')


# In[2]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder, 
    StandardScaler, 
    MinMaxScaler,
    RobustScaler
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as ss
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error
import math as m
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from base64 import b64encode
from IPython.display import display, HTML
import io
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
#from phik import phik
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from numpy.random import RandomState
import scipy.stats as st 
import time
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import socket
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt
from matplotlib import rcParams
from sklearn.dummy import DummyRegressor


# In[3]:


RANDOM_STATE = int('060524')


# In[4]:


try:
    arc = pd.read_csv('/datasets/data_arc_new.csv')
    bulk = pd.read_csv('/datasets/data_bulk_new.csv')
    bulk_time = pd.read_csv('/datasets/data_bulk_time_new.csv')
    gas = pd.read_csv('/datasets/data_gas_new.csv')
    temp = pd.read_csv('/datasets/data_temp_new.csv')
    wire = pd.read_csv('/datasets/data_wire_new.csv')
    wire_time = pd.read_csv('/datasets/data_wire_time_new.csv')
except FileNotFoundError:
    arc = pd.read_csv('data_arc_new.csv')
    bulk = pd.read_csv('data_bulk_new.csv')
    bulk_time = pd.read_csv('data_bulk_time_new.csv')
    gas = pd.read_csv('data_gas_new.csv')
    temp = pd.read_csv('data_temp_new.csv')
    wire = pd.read_csv('data_wire_new.csv')
    wire_time = pd.read_csv('data_wire_time_new.csv')


# In[5]:


arc.sample(3)


# In[6]:


arc.info()


# In[7]:


bulk.sample(3)


# In[8]:


bulk.info()


# In[9]:


bulk_time.sample(3)


# In[10]:


bulk_time.info()


# In[11]:


gas.sample(3)


# In[12]:


gas.info()


# In[13]:


temp.sample(3)


# In[14]:


temp.info()


# In[15]:


wire.sample(3)


# In[16]:


wire.info()


# In[17]:


wire_time.sample(3)


# In[18]:


wire_time.info()


# Увидели в датасетах `temp`, `bulk`, `bulk_time`, `wire`, `wire_time` пропущенные значения. Во всех датасетах нужно поменять названия столбцов и некоторые типы данных столбцов.

# # 2. Анализ и предобработка данныхазчику

# In[19]:


def plot_hist_box(df, column, hist_title=None, box_title=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.histplot(data=df[column], ax=axes[0])
    if hist_title is not None:
        axes[0].set_title(hist_title)

    sns.boxplot(data=df[column], ax=axes[1])
    if box_title is not None:
        axes[1].set_title(box_title)
        
    plt.show()


# Написали функцию для визуализации количественных признаков при помощи гистограммы и боксплота.

# # --------------------------------------------------------------------------------------

# `data_arc_new.csv`

# In[20]:


arc.head(2)


# In[21]:


arc.columns = ['key', 'begin_arc', 'end_arc', 'active_power', 'reactive_power']


# In[22]:


arc['begin_arc'] = pd.to_datetime(arc['begin_arc'])
arc['end_arc'] = pd.to_datetime(arc['end_arc'])


# In[23]:


arc['diff_arc'] = arc['end_arc'] - arc['begin_arc']


# In[24]:


arc['minutes'] = round(arc['diff_arc'].dt.total_seconds() / 60.0, 1)
arc.head(2)


# In[25]:


arc_imp = arc.loc[:, ['key', 'active_power', 'reactive_power','minutes']]


# In[26]:


arc_imp = arc_imp.query('reactive_power > -50')


# In[27]:


arc_imp['full_power'] = (arc_imp['active_power']**2 + arc_imp['reactive_power'])**0.5


# In[28]:


arc_pivot = arc_imp.pivot_table(index = 'key', values = ['active_power',	'reactive_power',	'minutes', 'full_power'], aggfunc = 'sum')
arc_pivot_imp = arc_pivot.reset_index()
arc_pivot_imp.head(2)


# In[29]:


arc_pivot_imp.info()


# В датасете `arc` поменяли тип данных дат на корректный тип, добавили признак длительности нагрева дугой в минутах `minutes`, изходя из дат начала и конца нагрева и добавили еще один признак - полная мощность ( сумма активной и реактивной мощностей ) `full_power`

# ![image.png](attachment:3f781510-03ec-438b-b2d6-dd9e9d86f816.png)

# В конце свели таблицу к сумме данных по ключу `key`.

# In[30]:


plot_hist_box(arc_pivot_imp, 'full_power', '''Распределение значений
активной мощности''', '''Распределение значений 
активной мощности (ящик с усами)''')


# In[31]:


plot_hist_box(arc_pivot_imp, 'active_power', '''Распределение значений
активной мощности''', '''Распределение значений 
активной мощности (ящик с усами)''')


# In[32]:


plot_hist_box(arc_pivot_imp, 'reactive_power', '''Распределение значений
реактивной мощности''', '''Распределение значений 
реактивной мощности (ящик с усами)''')


# In[33]:


arc_pivot_imp[arc_pivot_imp['reactive_power'] < -50]


# In[34]:


#arc_pivot_imp = arc_pivot_imp.query('reactive_power > -50')


# Сразу же удаляем аномальный выброс в столбце `reactive_power`, равный -715

# In[35]:


plot_hist_box(arc_pivot_imp, 'reactive_power', '''Распределение значений
реактивной мощности''', '''Распределение значений 
реактивной мощности (ящик с усами)''')


# In[36]:


plot_hist_box(arc_pivot_imp, 'minutes', '''Распределение времени нагрева''', '''Распределение времени нагрева
(ящик с усами)''')


# Можем заметить, что во всех признаках имеются статистические выбросы, аномальных выбросов нет, один мы удалили из столбца `reactive_power`.

# # --------------------------------------------------------------------------------------

# `data_bulk_new.csv`

# In[37]:


bulk.info()


# In[38]:


bulk = bulk.fillna(0)
bulk.head(2)


# В 12, 14 и 15-ой подачах материалов больше половины данных, поэтому можем взять эти признаки, заполнить пропущенные значения нулями, так как удалить их не можем из-за большой потери данных, и продолжить работать с ними.

# In[39]:


bulk_imp = bulk[['key', 'Bulk 12', 'Bulk 14', 'Bulk 15']]
bulk_imp.columns = ['key', 'bulk_12', 'bulk_14', 'bulk_15']


# In[40]:


bulk_imp['bulk_12'].plot(kind = 'box')
plt.title('Распределение суммы объемов подаваемых материаллов bulk_12')
plt.show()


# In[41]:


bulk_imp['bulk_14'].plot(kind = 'box')
plt.title('Распределение суммы объемов подаваемых материаллов bulk_12')
plt.show()


# In[42]:


bulk_imp['bulk_15'].plot(kind = 'box')
plt.title('Распределение суммы объемов подаваемых материаллов bulk_12')
plt.show()


# In[43]:


bulk_imp = bulk_imp[bulk_imp['bulk_12'] < 1000]


# Удалили одну аномалию в столбце `bulk_12`.

# In[44]:


bulk_imp.info()


# # --------------------------------------------------------------------------------------

# `data_bulk_time_new.csv`

# In[45]:


bulk_time.info()


# In[46]:


bulk_time = bulk_time.fillna(0)
bulk_time.head(2)


# Из этого датасета найдем разницу между самым большим и самым маленьким временем подачи сыпучих материалов в каждой партии.

# In[47]:


for col in bulk_time.columns[1:]:
    bulk_time[col] = pd.to_datetime(bulk_time[col], errors='coerce')

bulk_time.head(2)


# In[48]:


df_filtered = bulk_time.loc[:, bulk_time.columns[1:]].apply(lambda x: x[x >= pd.Timestamp('2019-05-03')])
df_filtered.head(2)


# In[49]:


max_date = df_filtered.max(axis=1)
min_date = df_filtered.min(axis=1)


# In[50]:


bulk_time['date_difference'] = max_date - min_date


# In[51]:


bulk_time['bulk_time_min'] = round(bulk_time['date_difference'].dt.total_seconds() / 60.0, 1)
bulk_time.head(2)


# In[52]:


bulk_time_imp = bulk_time.loc[:, ['key', 'bulk_time_min']]
bulk_time_imp.head(2)


# In[53]:


bulk_time_imp.info()


# Нашли признак длительности подачи сыпучих материалов `bulk_time_min`.

# # --------------------------------------------------------------------------------------

# `data_gas_new.csv`

# In[54]:


gas.head(3)


# In[55]:


gas.columns = ['key', 'gas']


# In[56]:


plot_hist_box(gas, 'gas', '''Распределение объема 
подаваемого газа''', '''Распределение объема подаваемого газа
(ящик с усами)''')


# Здесь никаких действий не требуется.

# # --------------------------------------------------------------------------------------

# `data_temp_new.csv`

# In[57]:


temp.head(3)


# In[58]:


temp.info()


# В датасете с целевым признаком `temp` сначала переведем время замера в тип даты, затем сгруппируем датасет по ключу `key` так, чтобы мы получили первое время замера температуры и последнее в каждой партии, а также первую и последнюю температуру.

# In[59]:


temp.columns = ['key', 'meas_time', 'temp']


# In[60]:


temp['meas_time'] = pd.to_datetime(temp['meas_time'])


# In[61]:


temp_imp = temp.dropna().reset_index(drop=True)


# In[62]:


plot_hist_box(temp_imp, 'temp', '''Распределение температур''', '''Распределение температур
(ящик с усами)''')


# Увидили на графике аномальное значение целевого признака, удаляем его.

# In[63]:


temp_imp = temp_imp[temp_imp['temp'] > 1500]


# In[64]:


final_temp_imp = temp_imp.groupby(by = 'key').agg(['first', 'last']).reset_index()
final_temp_imp.columns = ['key', 'first_time', 'finish_time', 'first_temp', 'finish_temp']


# In[65]:


final_temp_imp.head(2)


# In[66]:


final_temp_imp['time_diff'] = final_temp_imp['finish_time'] - final_temp_imp['first_time']
final_temp_imp['time_diff'] = round(final_temp_imp['time_diff'].dt.total_seconds() / 60.0, 1)
final_temp_imp = final_temp_imp.query('key < 2500')
final_temp_imp.tail(2)


# In[67]:


final_temp_imp = final_temp_imp.drop(['first_time',	'finish_time'], axis = 1)


# In[68]:


final_temp_imp.info()


# Удалили все значения температур, где в партиях были замеры всего один раз, а именно такие замеры были после 2500-ой партии. Получили датасет из времени начального и конечного замера, температуры начального и конечного замера, и длительности всех замеров по партиям.

# # --------------------------------------------------------------------------------------

# `data_wire_new.csv`

# In[69]:


wire.head(3)


# In[70]:


wire = wire.fillna(0)


# In[71]:


wire.head(5)


# In[72]:


wire_imp = wire[['key', 'Wire 1']]
wire_imp.columns = ['key', 'wire_1']


# In[73]:


wire_imp['wire_1'].plot(kind = 'box')
plt.title('Распределение сумм объемов подаваемых проволочных материаллов wire_1')
plt.show()


# In[74]:


wire_imp[wire_imp['wire_1'] > 270]


# In[75]:


wire_imp = wire_imp[wire_imp['wire_1'] < 270]


# In[76]:


wire_imp.head(2)


# In[77]:


wire_imp.info()


# Сделали то же самое, что и для датасета `bulk`. Заполнили все пропуски нулями, увидили один столбец `wire_1`, в котором данных больше половины, и удалили из него 2 аномалии. Сохранили этот признак вместе с ключом. 

# # --------------------------------------------------------------------------------------

# `data_wire_time_new.csv`

# In[78]:


wire_time.head(3)


# In[79]:


wire_time = wire_time.fillna(0)


# In[80]:


for col in wire_time.columns[1:]:
    wire_time[col] = pd.to_datetime(wire_time[col], errors='coerce')
df_filtered = wire_time.loc[:, wire_time.columns[1:]].apply(lambda x: x[x >= pd.Timestamp('2019-05-03')])
max_date = df_filtered.max(axis=1)
min_date = df_filtered.min(axis=1)
wire_time['date_difference'] = max_date - min_date
wire_time['wire_time_min'] = round(wire_time['date_difference'].dt.total_seconds() / 60.0, 1)
wire_time.sample(5)


# In[81]:


wire_time_imp = wire_time.loc[:, ['key', 'wire_time_min']]
wire_time_imp.sample(5)


# In[82]:


wire_time_imp['wire_time_min'].plot(kind = 'box')
plt.title('Распределение')


# In[83]:


wire_time_imp = wire_time_imp[wire_time_imp['wire_time_min'] < 70]


# Сделали то же самое, что и для датасета `bulk_time`. Добавили признак разницы самого большого времени подачи проволочных материалов и самого маленького. Также удалили 3 аномалии.

# # --------------------------------------------------------------------------------------

# Добавили признаки `full_power`, `minutes`, `bulk_time_min`, `wire_time_min`, `first_temp`, `finish_temp`, `time_diff`. Из датасетов также отобрали признаки, которые увеличат корреляцию с целевым признаком: `bulk_12`, `bulk_14`, `bulk_15`, `wire_1`. Удалили аномальные выбросы и подготовили датасеты к объединению в один.

# # 3. Объединение данных

# In[84]:


merged_data_1 = pd.merge(final_temp_imp, arc_pivot_imp, on = 'key', how = 'inner')


# In[85]:


merged_data_2 = pd.merge(merged_data_1, bulk_imp, on = 'key', how = 'inner')


# In[86]:


merged_data_3 = pd.merge(merged_data_2, bulk_time_imp, on = 'key', how = 'inner')


# In[87]:


merged_data_4 = pd.merge(merged_data_3, gas, on = 'key', how = 'inner')


# In[88]:


merged_data_5 = pd.merge(merged_data_4, wire_imp, on = 'key', how = 'inner')


# In[89]:


merged_data = pd.merge(merged_data_5, wire_time_imp, on = 'key', how = 'inner')
merged_data.head(2)


# In[90]:


merged_data.info()


# Объединили все обработанные датасеты. Заметим, что в датафрейме 2477 строк, в некоторых столбцах есть пропуски. Нужно обработать датафрейм и провести анализ.

# # 4. Исследовательский анализ и предобработка данных объединённого датафрейма

# In[91]:


merged_data.info()


# In[92]:


merged_data.describe()


# In[93]:


merged_data = merged_data.dropna().reset_index(drop=True)


# In[94]:


merged_data.hist(figsize = (15,15));


# In[95]:


merged_data.corr()


# In[96]:


quantitative_features = merged_data.drop(['finish_temp', 'key'], axis = 1).columns
for feature in quantitative_features:
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=feature, y='finish_temp', data = merged_data)
    plt.title(f"Диаграмма рассеяния между {feature} и температурой")
    plt.xlabel(feature)
    plt.ylabel("Температура")
    plt.show()


# Удалили из датасета пропуски. В данных не обнуражили коллениарности. Целевой признак `finish_temp` не сильно коррелирует с остальными признаками. Из обучающей выборке нужно будет удалить статистические выбросы.

# # 5. Подготовка данных

# In[97]:


X = merged_data.drop(['finish_temp', 'key'], axis = 1)
y = merged_data['finish_temp']


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.25, 
    random_state = RANDOM_STATE
)


# In[99]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# Разделили данные на обучающую и тестовую выборки в соотношении 1/3. В обучающей выборке 1744 строк, а в тестовой 582.

# In[100]:


X_train = X_train[X_train['gas'] < 60]
X_train = X_train[X_train['bulk_time_min'] < 100]
X_train = X_train[X_train['wire_time_min'] < 40]


# Удалили статистические выбросы из обучающей выборки.

# In[101]:


X_train.duplicated().sum()


# In[102]:


X_test.duplicated().sum()


# In[103]:


y_train = y_train.loc[X_train.index]


# In[104]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# Создадим пайплайн для дальнейшего обучения моделей с подобранными гиперпараметрами.

# In[105]:


num_columns  = ['first_temp',
                'time_diff',
                'active_power',
                'full_power',
                'minutes',
                'reactive_power',
                'bulk_12',	
                'bulk_14',
                'bulk_15',
                'bulk_time_min',
                'gas',
                'wire_1',
                'wire_time_min']


# In[106]:


num_pipe = Pipeline([ ('minmax', MinMaxScaler()), ('standard', StandardScaler()) ])

data_preprocessor = ColumnTransformer(
    [
        ('num', num_pipe, num_columns)
    ], 
    remainder='passthrough'
)
pipeline = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeRegressor())
    ]
)


# Разделили датафрейм на обучающую и тестовую выборку, удалили выбросы и создали пайплайн с использованием регрессора дерева решений.

# # 6. Обучение моделей машинного обучения

# На данном этапе обучим 3 модели регрессии и подберем гиперпараметры с помощью RandomizedSearchCV.

# `LinearRegression`

# In[107]:


param_grid_1 = [
    {
        'models': [LinearRegression()],
        'models__fit_intercept': [True, False],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    } 
]


# In[108]:


rs_1 = RandomizedSearchCV(
    pipeline, 
    param_grid_1, 
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state = RANDOM_STATE
)


# In[109]:


start_time_1 = time.time()
rs_1.fit(X_train, y_train)
end_time_1 = time.time()
gs_time_1 = end_time_1 - start_time_1


# In[110]:


mae_1 = abs(rs_1.best_score_)


# In[111]:


cross_mae_1 = cross_val_score(rs_1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cross_mae_1 = -cross_mae_1


# # --------------------------------------------------------------------------------------

# `GradientBoostingRegressor`

# In[112]:


param_grid_2 = [
    {
        'models': [GradientBoostingRegressor()],
        'models__max_depth': [3, 5, 7]
    }
]


# In[113]:


rs_2 = RandomizedSearchCV(
    pipeline, 
    param_grid_2, 
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state = RANDOM_STATE
)


# In[114]:


start_time_2 = time.time()
rs_2.fit(X_train, y_train)
end_time_2 = time.time()
gs_time_2 = end_time_2 - start_time_2


# In[115]:


mae_2 = abs(rs_2.best_score_)


# In[116]:


cross_mae_2 = cross_val_score(rs_2, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cross_mae_2 = -cross_mae_2


# # --------------------------------------------------------------------------------------

# `RandomForestRegressor`

# In[117]:


param_grid_3 = [
    {
        'models': [RandomForestRegressor(random_state = RANDOM_STATE)],
        'models__n_estimators': [40 ,50, 60],
        'models__max_depth': [10, 20, 30]
    }
]


# In[118]:


rs_3 = RandomizedSearchCV(
    pipeline, 
    param_grid_3, 
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state = RANDOM_STATE
)


# In[119]:


start_time_3 = time.time()
rs_3.fit(X_train, y_train)
end_time_3 = time.time()
gs_time_3 = end_time_3 - start_time_3


# In[120]:


mae_3 = abs(rs_3.best_score_)


# In[121]:


cross_mae_3 = cross_val_score(rs_3, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cross_mae_3 = -cross_mae_3


# # --------------------------------------------------------------------------------------

# Обучили 3 модели и сохранили для каждой модели время обучения в секундах, лучшую метрику MAE на обучающей выборке и метрику MAE на кросс-валидации.

# # 7. Выбор лучшей модели

# Все результаты занесем в таблицу `result`.

# In[122]:


results = pd.DataFrame({'LinearRegression': {'fit_time': gs_time_1,
                                             'mae': mae_1,
                                             'cross_mae': min(cross_mae_1)},
                         'GradientBoostingRegressor': {'fit_time': gs_time_2,
                                                   'mae': mae_2,
                                                   'cross_mae': min(cross_mae_2)},
                         'RandomForestRegressor': {'fit_time': gs_time_3,
                                            'mae': mae_3,
                                            'cross_mae': min(cross_mae_3)}})
results


# Получили, что модель градиентного бустинга `GradientBoostingRegressor` лучше всех справилась с задачей. На обучающей выборке метрика MAE = 6.053084, на кросс-валидации = 5.817608, а время обучения модели 15.487862 секунд, что меньше, чем у RandomForestRegressor. 

# In[123]:


y_pred = rs_2.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_mae


# In[124]:


dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_pred = dummy_regr.predict(X_test)
dummy_mae = mean_absolute_error(y_test, dummy_pred)

if test_mae < dummy_mae:
    print("Эта модель является адекватной по сравнению с DummyRegressor.")
else:
    print("Эта модель является адекватной по сравнению с DummyRegressor.")


# In[125]:


dummy_mae


# In[126]:


rcParams['figure.figsize'] = 18, 8
sns.lineplot(data=y_test, dashes=False)
sns.lineplot(x=y_test.index, y=y_pred, dashes=False)
plt.grid(True)
plt.title('Сравнение наших данных и результатов прогнозирования лучшей модели')
plt.show()


# In[127]:


get_ipython().system('pip install shap')
import shap


# In[128]:


best_model = rs_2.best_estimator_
explainer = shap.TreeExplainer(best_model.named_steps['models'])
shap_values = explainer.shap_values(X_test)


# In[129]:


plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, max_display=25, plot_type = 'bar')
plt.show()


# Из графика важности признаков можно сказать, что для предсказания конечной температуры стоит в приоритете длительность нагрева ковша дугой, затем идет разница во времени замеров начальной и конечной температур, и потом начальная температура. Почти никакой роли не играют такие признаки, как продувка сплава газом, замеры времени проволочных и сыпучих материалов, активная, реактивная и полная мощности.

# # 8. Общий вывод и рекомендации заказчику

# Модель градиентного бустинга `GradientBoostingRegressor` показала лучшие результаты в предсказании конечной температуры процесса нагрева металла. Она имеет более низкую ошибку и временные затраты на обучение. 

# Рекомендации для заказчика:
# 1. Продолжать использовать модель градиентного бустинга для прогнозирования конечной температуры в процессе нагрева металла.
# 2. Обратить внимание на признаки, которые оказывают наибольшее влияние на конечную температуру (длительность нагрева ковша дугой, разница во времени замеров начальной и конечной температур, начальная температура) и уделить им особое внимание в процессе управления производством. 
# 3. Провести дополнительный анализ и оптимизацию других факторов, таких как продувка сплава газом, замеры времени проволочных и сыпучих материалов, активная, реактивная и полная мощности, чтобы выявить их потенциальное влияние на конечную температуру и возможность их оптимизации.
