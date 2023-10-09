import math
import pandas as pd
import numpy as np
import time

from sklearn.metrics import confusion_matrix , roc_curve , auc , classification_report ,accuracy_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
start_time = time.time()
# 忽略警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def divide_on_feature(X, feature_i, threshold):
    """
    依据切分变量和切分点，将数据集分为两个子区域
    """
    split_func = None
    if isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    return X_1 , X_2

def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_gini(y):
    unique_labels = np.unique(y)
    var = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        var += p ** 2
    return 1 - var


class DecisionNode():
    def __init__(self, feature_i=None, threshold=None,
        value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i # 当前结点测试的特征的索引
        self.threshold = threshold # 当前结点测试的特征的阈值
        self.value = value # 结点值（如果结点为叶子结点）
        self.true_branch = true_branch # 左子树（满足阈值， 将特征值大于等于切分点值的数据划分为左子树）
        self.false_branch = false_branch # 右子树（未满足阈值， 将特征值小于切分点值的数据划分为右子树）


class DecisionTree(object):
    def __init__(self, min_samples_split=100, min_impurity=1e-7,max_depth=float("inf"), loss=None):
        self.root = None # 根结点
        self.min_samples_split = min_samples_split # 满足切分的最少样本数
        self.min_impurity = min_impurity # 满足切分的最小纯度
        self.max_depth = max_depth # 树的最大深度
        self._impurity_calculation = None # 计算纯度的函数，如对于分类树采用信息增益
        self._leaf_value_calculation = None # 计算 y 在叶子结点值的函数
        self.one_dim = None # y 是否为 one-hot 编码

    def fit(self, X, y):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """
        递归方法建立决策树
        """
        largest_impurity = 0
        best_criteria = None # 当前最优分类的特征索引和阈值
        best_sets = None # 数据子集
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 对每个特征计算纯度
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # 遍历特征 i 所有的可能值找到最优纯度
                for threshold in unique_values:
                    # 基于 X 在特征 i 处是否满足阈值来划分 X 和 y， Xy1 为满足阈值的子集
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # 取出 Xy 中 y 的集合
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]     
                        # 计算纯度
                        impurity = self._impurity_calculation(y, y1, y2)
                        # 如果纯度更高，则更新
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                            "leftX": Xy1[:, :n_features], # X 的左子树
                            "lefty": Xy1[:, n_features:], # y 的左子树
                            "rightX": Xy2[:, :n_features], # X 的右子树
                            "righty": Xy2[:, n_features:] # y 的右子树
                            }
        if largest_impurity > self.min_impurity:
            # 建立左子树和右子树
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"], true_branch=true_branch, false_branch=false_branch)
        # 如果是叶结点则计算值
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)
    
    def predict_value(self, x, tree=None):
        """
        预测样本，沿着树递归搜索
        """
        # 根结点
        if tree is None:
            tree = self.root
        # 递归出口
        if tree.value is not None:
            return tree.value
        # 选择当前结点的特征
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        return self.predict_value(x, branch)


    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
    
    def print_tree(self, tree=None, indent=" "):
        """
        输出树
        """
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("feature|threshold -> %s | %s" % (tree.feature_i, tree.threshold))
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class ClassificationTree(DecisionTree):
    """
    分类树，在决策树节点选择计算信息增益/基尼指数，在叶子节点选择多数表决。
    """
    def _calculate_gini_index(self, y, y1, y2):
        """
        计算基尼指数
        """
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_index = gini - p * calculate_gini(y1) - (1 - p) * calculate_gini(y2)
        return gini_index
    
    def _calculate_information_gain(self, y, y1, y2):
        """
        计算信息增益
        """
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)
        return info_gain
    
    def _majority_vote(self, y):
        """
        多数表决
        """
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    def fit(self, X, y):
        self._impurity_calculation = self._calculate_gini_index
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)




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

print("Training set label distribution:\n",train_data['label'].value_counts())
print("Test set label distribution:\n",test_data['label'].value_counts())

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
features = [item for item in list(train_data.columns) if item != "label"]
for items in features:
    train_data[items] = encoder.fit_transform(train_data[items])
    test_data[items] = encoder.fit_transform(test_data[items])

# 特征标签分离
X_train, y_train = np.array(train_data)[:,:-1], np.array(train_data)[:,-1] 
X_test, y_test = np.array(test_data)[:,:-1], np.array(test_data)[:,-1]

model = ClassificationTree(max_depth=5, min_impurity=0.001)
model.fit(X_train, y_train)

print('waiting······')

# 测试数据
print(model.score(X_test, y_test))

# 预测值
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 创建一个热力图
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# 设置坐标轴标签和标题
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 显示图形
plt.show()

# AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 计算AUC值
auc_value = auc(fpr, tpr)

# 绘制AUC曲线
plt.figure(figsize=(12,8))
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
report = classification_report(y_test, y_pred)

# 打印分类报告
print(report)
end_time = time.time()
print("Time elapsed: %.2f seconds" % (end_time - start_time))