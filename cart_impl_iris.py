import numpy as np
import copy as copy


class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.items = []  # a node in decision tree contains many data samples
        self.feature = None  # indicate split dataset into two parts based on which feature
        self.feature_value = None  # the feature's value

    @property
    # the prediction of a node depends on its majority's value
    def predict(self):
        maxCount = 0
        for i in np.unique(self.items[1]):
            if self.items[1].count(i) > maxCount:
                maxCount = self.items[1].count(i)
                maxPredict = i
        return maxPredict

    def __str__(self):
        if self.left == None and self.right == None:
            return "size:%d predict:%s" % (len(self.items), str(self.predict))
        else:
            return "feature:%s feature_value:%s" % (self.feature, self.feature_value)

    # calculate Gini index of a node
    def get_leafEntropy(self):
        g = 1
        n = len(self.items[1])
        p = {}
        for item in self.items[1]:
            p.setdefault(item, 0)
            p[item] += 1
        for v in p.values():
            g -= (v / n) ** 2
        return g

    # calculate the number of children node
    def get_leaf_num(self):
        if self.left is not None and self.right is not None:
            return self.right.get_leaf_num() + self.left.get_leaf_num()
        else:
            return 1


class Dtree(object):
    def __init__(self):
        self.root = Node()

    def __str__(self):
        queue = [(self.root, -1)]
        level = 0
        res = []
        while queue:
            node, prelevel = queue.pop(0)
            res.append("%d -> %d: %s" % (prelevel, prelevel + 1, str(node)))
            if node.left:
                queue.append((node.left, prelevel + 1))
            if node.right:
                queue.append((node.right, prelevel + 1))

            level += 1
        return "\n".join(res)

    # calculate Gini index of a node based on a specified feature
    def get_nodeEntropy(self, node):
        ll = len(node.left.items[0])
        lr = len(node.right.items[0])
        return (ll * node.left.get_leafEntropy() + lr * node.right.get_leafEntropy()) / (ll + lr)

    def split(self, feature, feature_value, idx, X):
        div = [[], []]  # 对于不同类型的特征选用不同的划分方法：对于离散的，根据是否相等来划分；对于连续的，根据大于还是小于进行划分
        # for i in idx:
        #     if X[i][feature] == feature_value:
        #         div[0].append(i)
        #     else:
        #         div[1].append(i)
        # return div
        for i in idx:
            if X[i][feature] <= feature_value:
                div[0].append(i)
            else:
                div[1].append(i)
        return div

    def get_G(self, idx, X, y):
        g = 1
        n = len(idx)
        p = {}
        for i in idx:
            p.setdefault(y[i], 0)
            p[y[i]] += 1
        for v in p.values():
            g -= (v / n) ** 2
        return g

    # choose a specified feature's value which has minimum Gini value
    def get_bestFeatureValue_forAFeature(self, X, y, idx, feature, best_feature, best_feature_value, minG):
        feature_vs = np.unique([X[i][feature] for i in idx])
        for feature_v in feature_vs:
            div = self.split(feature, feature_v, idx, X)
            ll = len(div[0])
            lr = len(div[1])
            curG = (ll * self.get_G(div[0], X, y) + lr * self.get_G(div[1], X, y)) / (ll + lr)
            if curG < minG:
                minG = curG
                best_feature = feature
                best_feature_value = feature_v
        return best_feature, best_feature_value, minG

    # choose the best feature and its value such that has minimum Gini index
    def get_bestFeatureAndValue(self, X, y, idx):
        best_feature = 0
        best_feature_value = X[0][best_feature]
        minG = 1
        for feature in range(len(X[0])):
            best_feature, best_feature_value, minG = self.get_bestFeatureValue_forAFeature(X, y, idx, feature,
                                                                                           best_feature,
                                                                                           best_feature_value, minG)
        return best_feature, best_feature_value

    def create_Dtree(self, X, y):
        queue = [(self.root, range(len(X)))]
        while queue:
            node, idx = queue.pop(0)
            if len(np.unique([y[i] for i in idx])) == 1:
                node.items = [copy.deepcopy(idx), [y[i] for i in idx]]
                continue
            best_feature, best_feature_value = self.get_bestFeatureAndValue(X, y, idx)

            print("bestFeature: %s, bestFeatureValue: %s" % (str(best_feature), str(best_feature_value)))

            node.feature = best_feature
            node.feature_value = best_feature_value
            node.items = [copy.deepcopy(idx), [y[i] for i in idx]]  # 为便于剪枝时比较单节点树形式和子树形式的基尼系数，子树的标记也需要保存到根节点

            div = self.split(best_feature, best_feature_value, idx, X)
            if div[0] != []:
                node.left = Node()
                node.left.parent = node
                queue.append((node.left, div[0]))
            if div[1] != []:
                node.right = Node()
                node.right.parent = node
                queue.append((node.right, div[1]))

    def predict(self, xi):
        node = self.root
        while node.left or node.right:
            if xi[node.feature] <= node.feature_value:
                node = node.left
            else:
                node = node.right
        return node.predict

    def get_min_gt(self):
        minGt = 0
        targetNode = None
        queue = [(self.root)]
        while queue:
            node = queue.pop(0)
            Ct = node.get_leafEntropy()  # 寻找最小的g(t)，见统计学习方法p86
            CTt = self.get_nodeEntropy(node)
            leafnum = node.get_leaf_num()
            curGt = (Ct - CTt) / (leafnum - 1)
            if minGt == 0 or curGt < minGt:
                minGt = curGt
                targetNode = node
            if node.left.left and node.left.right:
                queue.append((node.left))
            if node.right.left and node.right.right:
                queue.append((node.right))
        return targetNode, minGt

    def merge_subTree(self, node):
        node.left = None
        node.right = None


from sklearn import datasets
from numpy.random import choice

iris = datasets.load_iris()

X = iris.data
y = iris.target
n = X.shape[0]

train_size = int(0.6 * n)  # the size of training sets: 60% of the whole dataset
cv_size = int(0.2 * n)  # the size of cross validation sets: 20% of the whole dataset
test_size = int(0.2 * n)  # the size of testing sets: 20% of the whole dataset

# choose training sets randomly
train_rows = choice(range(n), size=train_size, replace=False)
X_train = [X[i] for i in train_rows]
y_train = [y[i] for i in train_rows]

# choose cross validation sets randomly
remains = [i for i in range(n) if i not in train_rows]
cv_rows = choice(remains, size=cv_size, replace=False)
X_cv = [X[i] for i in cv_rows]
y_cv = [y[i] for i in cv_rows]

# remaining is testing sets
X_test = [X[i] for i in remains if i not in cv_rows]
y_test = [y[i] for i in remains if i not in cv_rows]

t1 = Dtree()
t1.create_Dtree(X_train, y_train)

predictTrue = 0
queue = [t1]  # candidate queue of decision tree
bestTree = t1
maxAcc = 0
alpha = []

while queue:  # prune and cross validate
    curTree = queue.pop(0)
    predictTrue = 0
    for i in range(len(X_cv)):
        curPredict = curTree.predict(X_cv[i])
        if curPredict == y_cv[i]:
            predictTrue += 1
    curAcc = predictTrue / len(y_cv)

    print("the accuracy of cross validation", curAcc)

    # choose the tree which has best performance on cross validation tree
    if curAcc > maxAcc:
        bestTree = curTree
        maxAcc = curAcc
    if curTree.root.left and curTree.root.right:
        nextTree = copy.deepcopy(curTree)
        # the decision tree with the highest accuracy on the cross validation set is selected as the optimal one
        bestNode, ai = nextTree.get_min_gt()

        print("alpha: ", ai)

        alpha.append(ai)
        nextTree.merge_subTree(bestNode)
        queue.append(nextTree)

predictTrue = 0
for i in range(len(X_train)):
    curPredict = bestTree.predict(X_train[i])
    if curPredict == y_train[i]:
        predictTrue += 1
acc = predictTrue / len(y_train)
print("the accuracy of decision tree on training data", acc)

# apply the selected decision tree to testing sets
predictTrue = 0
for i in range(len(X_test)):
    curPredict = bestTree.predict(X_test[i])
    if curPredict == y_test[i]:
        predictTrue += 1
acc = predictTrue / len(y_test)
print("the accuracy of decision tree on testing data", acc)
