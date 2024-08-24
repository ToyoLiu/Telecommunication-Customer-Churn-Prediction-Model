#!/usr/bin/env python
# coding: utf-8

# In[15]:


# 数据处理
import joblib
import numpy as np
import pandas as pd

# 读入数据集
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('./Telco-Customer-Churn.csv')

df.head()

# 检查特征缺失值情况
null_cols = df.isnull().sum()[df.isnull().sum() > 0]
print(null_cols[null_cols / len(df) > 0.5])

# In[16]:

# 数据初步清洗 首先进行初步的数据清洗工作，包含错误值和异常值处理，并划分类别型和数值型字段类型，其中清洗部分包含：MultipleLines、OnlineSecurity、OnlineBackup、DeviceProtection
# 、TechSupport、StreamingTV、StreamingMovies：错误值处理 TotalCharges：异常值处理 tenure：自定义分箱 错误值处理
# 数值特征异常点检查

cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['float', 'int']).columns

for col in num_cols:
    fig = plt.figure(figsize=(8, 4))
    plt.boxplot(df[col])
    plt.grid(True)
    plt.title(col)
    plt.show()
repl_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in repl_columns:
    df[i] = df[i].replace({'No internet service': 'No'})
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

# 替换值SeniorCitizen
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})

# 替换值TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
# TotalCharges空值：数据量小，直接删除
df = df.dropna(subset=['TotalCharges'])
df.reset_index(drop=True, inplace=True)  # 重置索引

# In[17]:


# 转换数据类型
df['TotalCharges'] = df['TotalCharges'].astype('float')


# 转换tenure
def transform_tenure(x):
    if x <= 12:
        return ('Tenure_1')
    elif x <= 24:
        return ('Tenure_2')
    elif x <= 36:
        return ('Tenure_3')
    elif x <= 48:
        return ('Tenure_4')
    elif x <= 60:
        return ('Tenure_5')
    else:
        return ('Tenure_over_5')


df['tenure_group'] = df.tenure.apply(transform_tenure)

# 数值型和类别型字段
Id_col = ['customerID']

target_col = ['Churn']

cat_cols = df.nunique()[df.nunique() < 10].index.tolist()

num_cols = [i for i in df.columns if i not in cat_cols + Id_col]

print('类别型字段：n', cat_cols)

print('-' * 30)

print('数值型字段：n', num_cols)

# In[20]:


# 探索性分析
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

# 目标变量Churn分布
df['Churn'].value_counts()

trace0 = go.Pie(labels=['未流失客户', '流失客户'],
                #                 labels=df['Churn'].value_counts().index,
                values=df['Churn'].value_counts().values,
                hole=.5,
                rotation=90,
                marker=dict(colors=['rgb(154,203,228)', 'rgb(191,76,81)'],
                            line=dict(color='white', width=1.3))
                )
data = [trace0]
layout = go.Layout(title='目标变量Churn分布', font=dict(size=26))

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='整体流失情况分布.html', auto_open=False)


# In[27]:


def plot_bar(input_col: str, target_col: str, title_name: str):
    cross_table = round(pd.crosstab(df[input_col], df[target_col], normalize='index') * 100, 2)

    # 索引
    index_0 = cross_table.columns.tolist()[0]
    index_1 = cross_table.columns.tolist()[1]

    # 绘图轨迹
    trace0 = go.Bar(x=cross_table.index.tolist(),
                    y=cross_table[index_0].values.tolist(),
                    #                     name=index_0,
                    marker=dict(color='rgb(154,203,228)'),
                    name='未流失客户'
                    )
    trace1 = go.Bar(x=cross_table.index.tolist(),
                    y=cross_table[index_1].values.tolist(),
                    #                     name=index_1,
                    marker=dict(color='rgb(191,76,81)'),
                    name='流失客户'
                    )

    data = [trace0, trace1]
    # 布局
    layout = go.Layout(title=title_name, bargap=0.4, barmode='stack', font=dict(size=26))

    # 画布
    fig = go.Figure(data=data, layout=layout)
    # 绘图
    py.offline.plot(fig, filename=f'./html/{title_name}.html', auto_open=False)


# 性别与是否流失的关系
chars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
         'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn', 'tenure_group']

plot_bar(input_col='tenure_group', target_col='Churn', title_name='在网时长与是否流失的关系')


# plot_bar(input_col='gender', target_col='Churn', title_name='性别与是否流失的关系')
# plot_bar(input_col='SeniorCitizen', target_col='Churn', title_name='是否为老年人与是否流失的关系')
# plot_bar(input_col='Dependents', target_col='Churn', title_name='是否独立与是否流失的关系')
# plot_bar(input_col='Partner', target_col='Churn', title_name='是否有配偶与是否流失的关系')
# plot_bar(input_col='PhoneService', target_col='Churn', title_name='是否开通电话服务业务与是否流失的关系')
# plot_bar(input_col='MultipleLines', target_col='Churn', title_name='是否开通多线业务与是否流失的关系')
# plot_bar(input_col='OnlineSecurity', target_col='Churn', title_name='是否开通网络安全服务与是否流失的关系')
# plot_bar(input_col='InternetService', target_col='Churn', title_name='是否开通互联网服务与是否流失的关系')
# ...

# In[30]:


def plot_histogram(input_col: str, title_name: str):
    churn_num = df[df['Churn'] == 'Yes'][input_col]
    not_churn_num = df[df['Churn'] == 'No'][input_col]

    # 图形轨迹
    trace0 = go.Histogram(x=churn_num,
                          bingroup=25,
                          histnorm='percent',
                          name='流失客户',
                          marker=dict(color='rgb(191,76,81)')
                          )
    trace1 = go.Histogram(x=not_churn_num,
                          bingroup=25,
                          histnorm='percent',
                          name='未流失客户',
                          marker=dict(color='rgb(154,203,228)')
                          )

    data = [trace0, trace1]
    layout = go.Layout(title=title_name, font=dict(size=26))

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename=f'./html/{title_name}.html', auto_open=False)


plot_histogram(input_col='MonthlyCharges', title_name='月费用与是否流失的关系')
# plot_histogram(input_col='TotalCharges', title_name='总费用与是否流失的关系')

# In[42]:


# 探索数值型变量相关性
# 中文显示问题
import matplotlib

matplotlib.rc("font", family='SimHei')

# 仅选择数值型列计算相关性
num_cols = df.select_dtypes(include=[np.number]).columns
df_num = df[num_cols]

plt.figure(figsize=(8, 6))
corr = df_num.corr()
sns.heatmap(corr, linewidths=0.1, cmap='tab20c_r', annot=True)
plt.title('数值型属性的相关性', fontdict={'fontsize': 'xx-large', 'fontweight': 'heavy'})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# In[43]:


# 对于二分类变量，编码为0和1;
# 对于多分类变量，进行one_hot编码；
# 对于数值型变量，部分模型如KNN、神经网络、Logistic需要进行标准化处理。
# 建模数据
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df_model = df
Id_col = ['customerID']
target_col = ['Churn']
# 分类型
cat_cols = df_model.nunique()[df_model.nunique() < 10].index.tolist()
# 二分类属性
binary_cols = df_model.nunique()[df_model.nunique() == 2].index.tolist()
# 多分类属性
multi_cols = [i for i in cat_cols if i not in binary_cols]
# 数值型
num_cols = [i for i in df_model.columns if i not in cat_cols + Id_col]
# 标准化处理
scaler = StandardScaler()
df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
# 二分类-标签编码
le = LabelEncoder()
for i in binary_cols:
    df_model[i] = le.fit_transform(df_model[i])
# 多分类-哑变量转换
df_model = pd.get_dummies(data=df_model, columns=multi_cols)
df_model.head()

# In[5]:


# 使用统计检定方式进行特征筛选。
# from sklearn import feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

X = df_model.copy().drop(['customerID', 'Churn'], axis=1)
y = df_model[target_col]
fs = SelectKBest(score_func=f_classif, k=20)
X_train_fs = fs.fit_transform(X, y)
X_train_fs.shape


def SelectName(feature_data, model):
    scores = model.scores_
    indices = np.argsort(scores)[::-1]
    return list(feature_data.columns.values[indices[0:model.k]])


# 输出选择变量名称
print(SelectName(X, fs))
fea_name = [i for i in X.columns if i in SelectName(X, fs)]
X_train = pd.DataFrame(X_train_fs, columns=fea_name)
X_train.head()

# In[10]:

# 建模
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, train_test_split

# 实例模型
logit = LogisticRegression()

knn = KNeighborsClassifier(n_neighbors=5)

svc_lin = SVC(kernel='linear', random_state=0, probability=True)

svc_rbf = SVC(kernel='rbf', random_state=0, probability=True)

mlp_model = MLPClassifier(hidden_layer_sizes=(8,), alpha=0.05, max_iter=50000,
                          activation='logistic', random_state=0)

gnb = GaussianNB()

decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)

rfc = RandomForestClassifier(n_estimators=100, random_state=0)

lgbm_c = LGBMClassifier(boosting_type='gbdt', n_estimators=100, random_state=0)

xgc = XGBClassifier(n_estimators=100, eta=0.02, max_depth=15, random_state=0, learning_rate=0.001)

# 模型列表
models = [logit, knn, svc_lin, svc_rbf, mlp_model, gnb,
          decision_tree, rfc, lgbm_c, xgc]

model_names = ["Logistic Regression", "KNN Classifier", "SVM Classifier Linear",
               "SVM Classifier RBF", "MLP Classifier", "Naive Bayes",
               "Decision Tree", "Random Forest Classifier", "LGBM Classifier",
               "XGBoost Classifier"]

scoring = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
# 多折交叉验证
results = pd.DataFrame(columns=['Model'] + list(scoring))
from imblearn.over_sampling import SMOTE

smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)
for i, model in enumerate(models):

    name = model_names[i]

    cv_results = cross_validate(model, X_smo, y_smo, cv=10,
                                scoring=scoring, return_train_score=False)
    mean_scores = np.zeros(len(scoring))
    for i, score in enumerate(scoring):
        mean_scores[i] = np.mean(cv_results[f'test_{score}'])

    temp_df = pd.DataFrame([{
        'Model': name,
        **dict(zip(scoring, mean_scores))
    }], columns=results.columns)

    results = pd.concat([results, temp_df], ignore_index=True)

table = ff.create_table(np.round(results, 4))
py.offline.iplot(table)

# In[10]:

# 导入评估指标计算函数
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier, log_evaluation, early_stopping

# 评估指标列表
scoring = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']

# 数据处理
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 模型训练
callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]
gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)

# 模型保存
joblib.dump(gbm, 'model.pkl')

# 预测和评估
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

result = {}
for metric in scoring:
    if metric == 'accuracy':
        result[metric] = accuracy_score(y_test, y_pred)
    elif metric == 'recall':
        result[metric] = recall_score(y_test, y_pred)
    elif metric == 'precision':
        result[metric] = precision_score(y_test, y_pred)
    elif metric == 'f1':
        result[metric] = f1_score(y_test, y_pred)
    elif metric == 'roc_auc':
        result[metric] = roc_auc_score(y_test, y_pred)
# 生成结果表
results = pd.DataFrame([result], columns=scoring)
print(results)

# 我们也可以对模型进行进一步优化，比如对决策树参数进行调优。
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

parameters = {'splitter': ('best', 'random'),
              'criterion': ("gini", "entropy"),
              "max_depth": [*range(3, 20)],
              }

clf = DecisionTreeClassifier(random_state=25)

GS = GridSearchCV(clf, parameters, scoring='f1', cv=10)
GS.fit(X_train, y_train)
print(GS.best_params_)
print(GS.best_score_)
clf = GS.best_estimator_
test_pred = clf.predict(X_test)
print('测试集：n', classification_report(y_test, test_pred))

# In[12]:


# 将这棵树画出来
import graphviz

from pydotplus.graphviz import graph_from_dot_data
from sklearn.tree import export_graphviz

part_DT = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
part_DT.fit(X_train, y_train)

dot_data = tree.export_graphviz(decision_tree=part_DT, max_depth=3,
                                out_file=None,
                                #                                  feature_names=X_train.columns,
                                feature_names=X_train.columns,
                                class_names=['not_churn', 'churn'],
                                filled=True,
                                rounded=True
                                )
graph = graphviz.Source(dot_data)
graph = graph_from_dot_data(dot_data)  # Create graph from dot data

graph.write_png('./决策树.png')  # Write graph to PNG image

# In[13]:


# 输出决策树属性重要性排序
import plotly.figure_factory as ff

import plotly as py

imp = pd.DataFrame(zip(X_train.columns, clf.feature_importances_))

imp.columns = ['feature', 'importances']

imp = imp.sort_values('importances', ascending=False)

imp = imp[imp['importances'] != 0]

table = ff.create_table(np.round(imp, 4))

py.offline.iplot(table)

# In[14]:


# 绘制ROC曲线
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

fprs = []
tprs = []
aucs = []

# 数据处理
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_smo, y_smo)


def roc_img(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(roc_auc)

    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)


models = [logit, knn, svc_lin, svc_rbf, mlp_model, gnb, decision_tree, rfc, lgbm_c, xgc]
names = ["Logistic Regression", "KNN Classifier", "SVM Clissifier Linear",
         "SVM Classifier RBF", "MLP Classifier", "Naive Bayes", "Decision Tree",
         "Random Forest Classifier", "LGBM Classifier", "XGBoost Classifier"]

for i in range(10):
    roc_img(models[i], X_train, X_test, y_train, y_test, names[i])
    plt.plot(fprs[i], tprs[i], lw=1.5, label="%s, AUC=%.3f" % (names[i], aucs[i]))

plt.xlabel("FPR", fontsize=15)
plt.ylabel("TPR", fontsize=15)

plt.title("ROC")
plt.legend(loc="lower right")
plt.show()
