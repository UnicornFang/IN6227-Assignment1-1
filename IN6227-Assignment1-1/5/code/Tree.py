import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , roc_curve , auc , classification_report ,accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
start_time = time.time()
def cal_information_entropy(data):
    '''
    Calculate information entropy
    :parameter data :data
    :return: information entropy
    '''
    data_label = data.iloc[:,-1]
    label_class = data_label.value_counts()
    Ent = 0
    for k in label_class.keys():
        p_k = label_class[k]/len(data_label)
        Ent += -p_k*np.log2(p_k)
    return Ent

def cal_information_gain(data,features):
    '''
    Calculate information gain
    :param data: data set
    :param features: features
    :return: the information gain of this feature
    '''
    Ent = cal_information_entropy(data)
    feature_class = data[features].value_counts()
    gain = 0
    for item in feature_class.keys():
        weight = feature_class[item] / data.shape[0]
        Ent_item = cal_information_entropy(data.loc[data[features] == item])
        gain += weight*Ent_item
    return Ent - gain

def get_most_label(data):
    '''
    Get the category with the most tags
    :param data: data set
    :return: The category with the most tags
    '''
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

def get_best_feature(data):
    '''
        Get the features with the greatest information gain
        :param data: data set
        :return: Features with the largest information gain
        '''
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = cal_information_gain(data, a)
        res[a] = temp
    res = sorted(res.items(),key=lambda x:x[1],reverse=True)
    return res[0][0]

def drop_exist_feature(data,best_feature):
    '''
    Update the data set to facilitate calculations
    :param data: metadata set
    :param best_feature:feature
    :return: updated data set
    '''
    attr = pd.unique(data[best_feature])
    # print(attr)
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    # print(new_data)
    return new_data

def create_tree(data):
    '''
    创建决策树
    :param data: 数据集
    :return: 返回字典形式的决策树
    '''
    data_label = data.iloc[:,-1]
    # 统计每个特征的取值情况作为全局变量
    features_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
    # 只有一类的情况
    if len(data_label.value_counts()) == 1:
        return data_label.values[0]

    # 所有数据特征值相同的情况
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns):
        return get_most_label(data)

    # 最优划分
    best_feature = get_best_feature(data)

    tree = {best_feature:{}}

    exist_vals = pd.unique(data[best_feature])
    if len(exist_vals) != len(features_count[best_feature]):
        no_exist_attr = set(features_count[best_feature]) - set(exist_vals)
        for no_feat in no_exist_attr:
            tree[best_feature][no_feat] = get_most_label(data)
    # 递归树
    for item in drop_exist_feature(data,best_feature):
        tree[best_feature][item[0]] = create_tree(item[1])
    return tree

def predict(tree,test_data):

    try:
        first_feature = list(tree.keys())[0]
        # print(first_feature)
        second_dict = tree[first_feature]
        # print(second_dict)
        input_first = test_data.get(first_feature)
        # print(input_first)
        try:
            input_value = second_dict[input_first]
            if isinstance(input_value, dict):
                class_label = predict(input_value, test_data)
            else:
                class_label = input_value

            if class_label == None:
                return 1
            else:
                return class_label
        except:
            if str(input_first)[2] == '<' and str(input_first)[3] == '-':
                new_input_first = str(input_first)[:2] + '>=' + str(input_first)[3:]
                input_value = second_dict[new_input_first]
            elif str(input_first)[2] == '>' and str(input_first)[3] == '=':
                new_input_first = str(input_first)[:2] + '<' + str(input_first)[4:]
                input_value = second_dict[new_input_first]

                if isinstance(input_value,dict):
                    class_label = predict(input_value, test_data)
                else:
                    class_label = input_value

                if class_label == None:
                    return 1
                else:
                    return class_label
    except:
        return 1

def Accuracy(label_1,label_2):
    '''
    Calculate prediction accuracy
    :param label_1: original label
    :param label_2: predicted label
    :return: prediction accuracy
    '''
    # 假设 0 是正类  1 是负类
    # 被检索为正样本，实际也是正样本
    TP = 0
    # 被检索为正样本，实际是负样本
    FP = 0
    # 未被检索为正样本，实际是正样本
    FN = 0
    # 未被检索为正样本，实际也是负样本
    TN = 0
    for i in range(len(label_1)):
        if label_1[i] == label_2[i] and label_2[i] == 0:
            TP += 1
        elif label_1[i] == label_2[i] and label_2[i] == 1:
            TN += 1
        elif label_1[i] != label_2[i] and label_1[i] == 0 and label_2[i] == 1:
            FN += 1
        elif label_1[i] != label_2[i] and label_1[i] == 1 and label_2[i] == 0:
            FP += 1
    value = (TP+TN)/(TP+TN+FN+FP)
    return value



# 读取数据
columns_list = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
train_data = pd.read_csv("../data/adult.data",header=None,names=columns_list)
test_data = pd.read_csv("../data/adult.test",header=None,names=columns_list)
print(train_data.shape)
print(test_data.shape)

# 查看缺失值
train_data = train_data.replace(' ?', np.nan)
test_data = test_data.replace(' ?', np.nan)
missing_rows_train  = train_data.isnull().sum(axis=1)
missing_rows_test  = test_data.isnull().sum(axis=1)
print("Number of rows with missing values in the training set：",len(missing_rows_train[missing_rows_train > 0]))
print("Number of rows with missing values in test set：",len(missing_rows_test[missing_rows_test > 0]))

# 删除测试集具有缺失值的行
test_data = test_data.dropna().reset_index(drop=True)
missing_rows_test  = test_data.isnull().sum(axis=1)
print("Number of rows with missing values in the test set after deleting missing values：",len(missing_rows_test[missing_rows_test > 0]))
print(train_data.shape)
print(test_data.shape)


# 标签化
train_data['label'] = train_data['label'].apply(lambda x:1 if x==" <=50K" else 0)
test_data['label'] = test_data['label'].apply(lambda x:1 if x==" <=50K." else 0)

# 将训练集的数据类型和测试集的数据类型进行统一
train_data[["fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]] = train_data[["fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]].astype(np.float64)
test_data["age"] = test_data["age"].astype(np.int64)

# 对age数据进行划分 18岁以下岁为 青少年 ；18-45岁为 青年 ；46-69岁为中年；69岁以上为 老年
def split_age(x):
    if x < 18:
        return "juvenile"
    elif 18<= x < 45:
        return "youth"
    elif 45 <= x < 69:
        return "midlife"
    else:
        return "old age"

train_data["age"] = train_data["age"].apply(lambda x:split_age(x))
test_data["age"] = test_data["age"].apply(lambda x:split_age(x))



# 将连续型数据分组化
for item in ["fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]:
    train_data[item] = pd.cut(train_data[item], 5)
    test_data[item] = pd.cut(test_data[item],5)

# 标签编码
encoder = LabelEncoder()
features = [item for item in list(train_data.columns) if item not in ["fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]]
for items in features:
    if items == "label":
        continue
    train_data[items] = encoder.fit_transform(train_data[items])
    test_data[items] = encoder.fit_transform(test_data[items])

features_list = list(train_data.columns)

tree = create_tree(train_data)
y_test = list(test_data.loc[:,'label'])

y_predict = []
for item in test_data.loc[:,features_list[:-1]].values:
    y_predict.append(predict(tree,dict(zip(features_list[:-1],item))))
print(f'决策树的准确率为：{Accuracy(y_test,y_predict)}')



# 计算混淆矩阵
cm = confusion_matrix(y_test, y_predict)

# 创建一个热力图
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# 设置坐标轴标签和标题
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 显示图形
plt.show()

# AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict)

# 计算AUC值
auc_value = auc(fpr, tpr)

# 绘制AUC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curve')
plt.legend(loc="lower right")
plt.show()

# Report
report = classification_report(y_test, y_predict)

# 打印分类报告
print(report)
end_time = time.time()
print("Time elapsed: %.2f seconds" % (end_time - start_time))