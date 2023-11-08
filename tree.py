from sklearn.metrics import silhouette_score
import numpy as np
from collections import deque
from tqdm import tqdm
import loss
class Node():
    def __init__(self, id_):
        #ルール・クラスタリングに関する情報
        self.id_ = id_
        self.feature = None
        self.threshold = None
        self.cluster = None
        self.is_leaf = False
        
        #親子関係
        self.parent = None
        self.smaller_equall = None
        self.larger = None
        
    def set_data(self, real_data, feature_data, square):
        self.real_data = np.copy(real_data)
        self.feature_data = np.copy(feature_data)
        self.square = np.copy(square)
    # ルール(特徴 feature に関して　閾値 threshold で分岐する)を登録する
    def set_rule(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        
    # 親子関係を登録する
    def set_parent_children(self, se, l):
        self.smaller_equall = se
        self.larger = l
        se.parent = self
        l.parent = self
        
    # クラスターを設定する
    def set_cluster(self, cluster_id):
        self.is_leaf = True
        self.cluster = cluster_id
        
class DecisionTree():
    def fit(self, real_data, feature_data, greedy_score_thre=0.5, stop_amount=0, num_kouho=1):
        self.num_kouho = num_kouho
        self.stop_amount = stop_amount
        square = np.sum(real_data**2, axis=1)
        self.n_dim = feature_data.shape[1]
        self.root = Node(0)
        self.root.set_data(real_data, feature_data, square)
        # 幅優先探索
        current_score = None
        node_i = 1
        cluster_i = 0
        queue = deque()
        queue.append(self.root)
        while queue:
            node = queue.popleft()
            if len(node.feature_data) < stop_amount:
                node.set_cluster(cluster_i)
                cluster_i += 1
                continue
            feature, threshold, score = self.search_split_point(node.real_data, node.feature_data, node.square)
            node_i, smaller_equall_node, larger_node = self.create_node(node_i)
            print(max(score))
            if np.array([i is None for i in threshold]).all():
                node_i -= 2
                node.set_cluster(cluster_i)
                cluster_i += 1
                continue
            new_score_list = np.array([self.calc_new_score(node, smaller_equall_node, larger_node, i, j) for i,j in zip(feature, threshold) if i is not None])
            max_score_index = np.argmax(new_score_list)
            max_score = new_score_list[max_score_index]
            greedy_score = score[max_score_index]
            print(f"current_score:{max_score}")
            if current_score is None or greedy_score > greedy_score_thre:
                self.set_node_info(node, smaller_equall_node, larger_node, feature[max_score_index], threshold[max_score_index])
                if current_score is None or max_score > current_score:
                    current_score = max_score
                queue.append(smaller_equall_node)
                queue.append(larger_node)
            else:
                node_i -= 2
                node.set_cluster(cluster_i)
                cluster_i += 1
        return
    
    def predict(self, x):
        node = self.root
        while True:
            if not node.is_leaf:
                feature, threshold = node.feature, node.threshold
                sm_eq_flg = x[feature] <= threshold
                if sm_eq_flg:
                    node = node.smaller_equall
                else:
                    node = node.larger
            else:
                return node.cluster
    
    def search_split_point(self, real_data, feature_data, square_data):
        num_kouho = min(len(real_data), self.num_kouho)
        best_score, best_feature, best_threshold = np.array([-np.inf]*num_kouho), [None]*num_kouho, [None]*num_kouho
        for feature in tqdm(range(self.n_dim)):
            for x in feature_data:
                threshold = x[feature]
                cluster_labels = np.array(list(map(int, feature_data[:, feature] <= threshold)))
                n_label = np.unique(cluster_labels).size
                if n_label == 1:
                    continue
                score= loss.loss(real_data, square_data, cluster_labels, self.stop_amount)
                min_index = np.argmin(best_score)
                if best_score[min_index] < score:
                    best_score[min_index] = score
                    best_feature[min_index], best_threshold[min_index] = feature, threshold
        return best_feature, best_threshold, best_score
    
    def calc_new_score(self, node, smaller_equall_node, larger_node, feature, threshold):
        self.set_node_info(node, smaller_equall_node, larger_node, feature, threshold)
        score = self.calc_score()
        node.smaller_equall, node.larger = None, None
        return score
    
    def calc_score(self):
        queue = deque([self.root])
        X, cluster_labels = [], []
        cluster_i = 0
        while queue:
            node = queue.popleft()
            if node.smaller_equall is not None:
                queue.append(node.smaller_equall)
                queue.append(node.larger)
            else:
                X += list(node.real_data)
                cluster_labels += [cluster_i] * len(node.real_data)
                cluster_i += 1
        X = np.array(X)
        cluster_labels = np.array(cluster_labels)
        return silhouette_score(X, cluster_labels.ravel())
    
    
    def create_node(self, node_i):
        smaller_equall_node = Node(node_i)
        node_i += 1
        larger_node = Node(node_i)
        node_i += 1
        return node_i, smaller_equall_node, larger_node
        
    def set_node_info(self, parent, smaller_equall, larger, feature, threshold):
        feature_data = parent.feature_data
        real_data = parent.real_data
        X_sq = parent.square
        is_smaller = (feature_data[:, feature] <= threshold)
        smaller_equall.set_data(real_data[is_smaller], feature_data[is_smaller], X_sq[is_smaller])
        larger.set_data(real_data[~is_smaller], feature_data[~is_smaller], X_sq[~is_smaller])
        parent.set_parent_children(smaller_equall, larger)
        parent.set_rule(feature, threshold)
        
    def check_structure(self, node):
        print(f"node id: {node.id_}")
        if node.is_leaf:
            print(f"cluster id: {node.cluster}")
        else:
            print(f"rule: feature{node.feature} <= {node.threshold}")
            if node.smaller_equall is not None:
                self.check_structure(node.smaller_equall)
            if node.larger is not None:
                self.check_structure(node.larger)
        
    # 葉ノードを入れる(クラスターidを示すノード)と，親ノードを探りルールを取得する
    def get_rule(self, leaf_node):
        rules = []
        node = leaf_node.parent
        while node is not None:
            rules.append([node.feature, node.threshold])
            node = node.parent
        print(rules)
        
    def plot(self, feature_names=None, filename=None):
        """make a dot file to visualize the tree. you can run the dot file with graphviz online (https://dreampuf.github.io/GraphvizOnline/).

        Args:
            feature_names (list, optional): list of feature names. Defaults to None.
            filename (str, optional): input any file name to save dot file. Defaults to None.
        """
        dot_str = ["digraph ClusteringTree {\n"]
        queue = [self.root]
        nodes = []
        edges = []
        id = 0
        while queue: #BFS
            curr = queue.pop(0)
            num_data = len(curr.feature_data)
            if curr.is_leaf:
                label = str(curr.cluster)
            else:
                feature_name = f'x_{curr.feature}' if feature_names is None else feature_names[curr.feature]
                label = "%s <= %.3f" % (feature_name, curr.threshold)
                queue.append(curr.smaller_equall)
                queue.append(curr.larger)
                edges.append((id, id + len(queue) - 1))
                edges.append((id, id + len(queue)))
            nodes.append({"id": id,
                        "label": label,
                        "node": curr,
                        "amount": num_data})
            id += 1
        for node in nodes:
            dot_str.append("n_%d [label=\"%s,amount=%s\"];\n" % (node["id"], node["label"], node["amount"]))
            
        for i, edge in enumerate(edges):
            label = True if i%2==0 else False
            dot_str.append("n_%d -> n_%d [label=\"%s\"];\n" % (edge[0], edge[1], label))
        dot_str.append("}")
        dot_str = "".join(dot_str)

        if filename is not None:
            with open(f'{filename}.dot', 'w') as o:
                o.write(dot_str)

        else:
            print(r"Run the following code at https://dreampuf.github.io/GraphvizOnline/")
            print('----------------------CODE----------------------')
            print(dot_str)

if __name__ == '__main__':
    #動かし方
    tree1 = DecisionTree()
    real_data = np.random.rand(100,10) #amount_item * time_point
    feature_data = np.random.rand(100,2) #amount_item * amount_feature
    tree1.fit(real_data, #時系列特徴量
            feature_data, #分枝特徴量
            greedy_score_thre=0, #分割を止める閾値
            stop_amount = 10 #クラスタ内に少なくとも含まれるデータ数
            # それ以外にも引数が載ってると思うけど動き的に信頼性がないから気にしないで
            )
    tree1.plot()