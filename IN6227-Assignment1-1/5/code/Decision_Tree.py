import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix , roc_curve , auc , classification_report ,accuracy_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 忽略警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()
class DecisionTree:
    def __init__(self, max_depth=None, min_impurity=0):
        self.max_depth = max_depth  # 树深度的最大值
        self.min_impurity = min_impurity  # 树分裂时的最小不纯度
        self.feature_index = None  # 分裂特征的索引
        self.threshold = None  # 分裂特征的阈值
        self.left = None  # 左子树，小于等于阈值
        self.right = None  # 右子树，大于阈值
        self.leaf_value = None  # 叶子节点的预测值
        
    def _calc_gini(self, y):
        _, counts = np.unique(y, return_counts=True)  # 计算类别及其出现次数
        probabilities = counts / len(y)  # 计算每个类别在样本中的比例
        gini = 1 - np.sum(probabilities**2)  # 计算基尼不纯度
        return gini
    
    def _split_dataset(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold  # 根据分裂阈值划分左右子集
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def _find_best_split(self, X, y):
        best_gini = np.inf  # 最小基尼不纯度
        best_feature_index = None  # 最佳分裂特征的索引
        best_threshold = None  # 最佳分裂特征的阈值
        
        for feature_index in range(X.shape[1]):  # 遍历每个特征
            unique_values = np.unique(X[:, feature_index])  # 获取该特征的所有取值
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # 在该特征的相邻取值之间计算中点作为阈值集合

            for threshold in thresholds:  # 遍历该特征的阈值集合
                X_1, y_1, X_2, y_2 = self._split_dataset(X, y, feature_index, threshold)
                # 计算分裂后左右子集的基尼不纯度
                gini_1 = self._calc_gini(y_1)
                gini_2 = self._calc_gini(y_2)
                gini = len(y_1) / len(y) * gini_1 + len(y_2) / len(y) * gini_2
                # 如果基尼不纯度更小，则更新最佳分裂特征和阈值
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold
    
    def _build_tree(self, X, y, depth):
        # 如果深度超过了最大深度，或者数据集中的样本全部属于同一类别，则返回叶子节点的预测值
        if depth == self.max_depth or len(np.unique(y)) == 1:
            self.leaf_value = np.bincount(y).argmax()
            return

        # 如果当前数据集的基尼不纯度小于指定的最小不纯度，则返回叶子节点的预测值
        gini = self._calc_gini(y)
        if gini < self.min_impurity:
            self.leaf_value = np.bincount(y).argmax()
            return

        # 找到最佳分裂特征和阈值
        self.feature_index, self.threshold = self._find_best_split(X, y)

        # 如果找不到可用的最佳分裂特征和阈值，则返回叶子节点的预测值
        if self.feature_index is None or self.threshold is None:
            self.leaf_value = np.bincount(y).argmax()
            return

        # 分裂数据集，构建左右子树
        X_1, y_1, X_2, y_2 = self._split_dataset(X, y, self.feature_index, self.threshold)
        self.left = DecisionTree(max_depth=self.max_depth, min_impurity=self.min_impurity)
        self.left._build_tree(X_1, y_1, depth + 1)
        self.right = DecisionTree(max_depth=self.max_depth, min_impurity=self.min_impurity)
        self.right._build_tree(X_2, y_2, depth + 1)

    
    def fit(self, X, y):
        self._build_tree(X, y, depth=0)  # 构建决策树
    
    def predict_sample(self, x):
        if self.left is None and self.right is None:  # 如果是叶子节点，则返回该节点的预测值
            return self.leaf_value
        
        if x[self.feature_index] <= self.threshold:  # 否则根据分裂特征和阈值决定往左还是往右走
            return self.left.predict_sample(x)
        else:
            return self.right.predict_sample(x)
    
    def predict(self, X):
        return np.array([self.predict_sample(x) for x in X])  # 预测整个样本集```
    

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

# 创建模型
model = DecisionTree(max_depth=20, min_impurity=0.01)

print('Waiting······\n')
model.fit(X_test, y_test)
print('Finishing!!!\n')

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