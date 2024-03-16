import os
import numpy as np
import pickle
import json
import time
import pandas as pd
from sklearn import metrics

import torch.nn.modules.transformer
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class Evaluator:
    def __init__(self, user_plot_config=None, asser_plot_config=None, use_b_matrix=False):
        self.initialized = False
        self.use_b_matrix = use_b_matrix
        self.output_dir = None
        self.embedding = None
        self.labels = None
        self.user_label = None
        self.asser_label = None
        self.user_clustering_pred = None
        self.asser_clustering_pred = None
        self.namelist = None
        self.asserlist = None
        self.num_user = -1
        self.num_asser = -1
        self.user_plot_config = [
            [1, "#178bff", "User Con"],
            # [0, "#3b3b3b", "User Neu"],
            [2, "#ff5c1c", "User Pro"]
        ] if user_plot_config is None else user_plot_config
        self.asser_plot_config = [
            [1, "#30a5ff", "Assertion Con"],
            # [0, "#4d4d4d", "Assertion Neu"],
            [2, "#fc8128", "Assertion Pro"]
        ] if asser_plot_config is None else asser_plot_config

    def initialize(self, B_matrix=None):
        if self.labels is not None:
            self.user_label = self.labels["user_label"]
            self.asser_label = self.labels["assertion_label"]
            # self.user_label[0] = 1
            # self.asser_label[0] = 1
        self.num_user = self.user_label.shape[0]
        self.num_asser = self.asser_label.shape[0]

        if B_matrix is not None and self.use_b_matrix:
            self.embedding = self.embedding @ B_matrix
        # B = np.array([
        #     [1, 0, 0],
        #     [1, 1, 0],
        #     [1, 0, 1]
        # ]) if B_matrix is None else B_matrix
        # if self.use_b_matrix:
        #     self.embedding[:self.num_user] = self.embedding[:self.num_user] @ B_matrix
        # else:
        #     self.embedding[:self.num_user] = self.embedding[:self.num_user]

        for item in self.asser_plot_config:
            item[0] += 1000
        self.asser_label += 1000

        print("Label: user {} + asser {} = total {}".format(
            np.sum(self.user_label != 0),
            np.sum(self.asser_label != 1000),
            np.sum(self.user_label != 0) + np.sum(self.asser_label != 1000)
        ))

        self.initialized = True

    def init_from_dir(self, dir):
        with open(dir / "embedding.bin", 'rb') as fin:
            if dir.find("Node2Vec") != -1 or dir.find("DeepWalk") != -1:
                self.embedding = np.concatenate(pickle.load(fin), axis=0)
            elif dir.find("GCN") != -1:
                self.embedding = np.load(dir + "/embedding.bin")
            else:
                self.embedding = pickle.load(fin)
        with open(dir / "label.bin", 'rb') as fin:
            self.labels = pickle.load(fin)
        with open(dir / "namelist.json", 'r') as fin:
            self.namelist = json.load(fin)
        with open(dir / "asserlist.json", 'r', encoding="utf-16") as fin:
            self.asserlist = json.load(fin)
        self.output_dir = dir
        self.initialize()

    def init_from_value(self, embedding, user_label, asser_label, namelist, asserlist, B_matrix, output_dir="."):
        self.embedding = embedding
        self.user_label = user_label
        self.asser_label = asser_label
        self.namelist = namelist
        self.asserlist = asserlist
        self.output_dir = output_dir
        self.initialize(B_matrix=B_matrix)

    def larger_d2d3(self, embedding):
        label = []
        for i in range(embedding.shape[0]):
            # if embedding[i][2] > 6 * embedding[i][1] + 0.8:
            if embedding[i][2] > embedding[i][1]:
                label.append(1)
            else:
                label.append(0)
        return np.array(label).astype("int32")

    def larger_d1d3(self, embedding):
        label = []
        for i in range(embedding.shape[0]):
            # if embedding[i][2] > 6 * embedding[i][1] + 0.8:
            if embedding[i][0] > embedding[i][2]:
                label.append(1)
            else:
                label.append(0)
        return np.array(label).astype("int32")

    def larger_d1d2(self, embedding):
        label = []
        for i in range(embedding.shape[0]):
            # if embedding[i][2] > 6 * embedding[i][1] + 0.8:
            if embedding[i][0] > embedding[i][1]:
                label.append(1)
            else:
                label.append(0)
        return np.array(label).astype("int32")

    def run_clustering(self, n_clusters=2):
        assert n_clusters == 2

        # self.user_clustering_pred = self.larger_d2d3(self.embedding[:self.num_user])
        # self.asser_clustering_pred = self.larger_d2d3(self.embedding[self.num_user:])

        # self.user_clustering_pred = self.larger_d1d3(self.embedding[:self.num_user])
        # self.asser_clustering_pred = self.larger_d1d3(self.embedding[self.num_user:])

        if self.embedding.shape[1] > 2:
            self.user_clustering_pred, _ = self.k_means(self.embedding[:self.num_user])
            self.asser_clustering_pred, _ = self.k_means(self.embedding[self.num_user:])
        else:
            self.user_clustering_pred = self.larger_d1d2(self.embedding[:self.num_user])
            self.asser_clustering_pred = self.larger_d1d2(self.embedding[self.num_user:])

        # together_pred, _ = self.k_means(self.embedding)
        # self.user_clustering_pred = together_pred[:self.num_user]
        # self.asser_clustering_pred = together_pred[self.num_user:]

        # if self.embedding.shape[1] > 1:
        #     # self.user_clustering_pred, _ = self.mean_shift(self.embedding[:self.num_user], bandwidth=0.5)
        # self.user_clustering_pred, _ = self.mean_shift(self.embedding[:self.num_user])
        # self.asser_clustering_pred, _ = self.dbscan(self.embedding[self.num_user:], eps=0.1, min_samples=5)
        # else:
        # self.user_clustering_pred, _ = self.mean_shift(self.embedding[:self.num_user])
        # self.asser_clustering_pred, _ = self.mean_shift(self.embedding[self.num_user:])
        # self.user_clustering_pred = self.positive_d2d3(self.embedding[:self.num_user])
        # self.asser_clustering_pred = self.positive_d2d3(self.embedding[self.num_user:])

        # print(list(self.user_clustering_pred))
        # print(list(self.asser_clustering_pred))

        for i in range(self.num_user):
            self.user_clustering_pred[i] = \
                self.user_plot_config[0][0] if self.user_clustering_pred[i] == 0 else self.user_plot_config[1][0]

        for i in range(self.num_asser):
            self.asser_clustering_pred[i] = \
                self.asser_plot_config[0][0] if self.asser_clustering_pred[i] == 0 else self.asser_plot_config[1][0]

    def plot_clustering(self, permulate=None, show=False, save=True):
        print("Evaluator plot clustering prediction with config:")
        print("user_plot_config: " + str(self.user_plot_config))
        print("asser_plot_config: " + str(self.asser_plot_config))
        assert self.user_clustering_pred is not None
        assert self.asser_clustering_pred is not None
        pred = np.concatenate([self.user_clustering_pred, self.asser_clustering_pred], axis=0)
        label = np.concatenate([self.user_label, self.asser_label], axis=0)
        # Only plot labeled data
        pred[label == 0] = -1
        pred[label == 1000] = -2
        if self.embedding.shape[1] == 1:
            self.plot_1d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 2:
            self.plot_2d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 3:
            self.plot_3d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, permulate, show, save)

    def evaluate_geo(self):
        with open("/Users/lijinning/PycharmProjects/Polarization_dev/CIKM_visualization/utils/us-states.json", "r") as fin:
            geo_data = json.load(fin)

        with open(self.output_dir / "tweet_to_assertion_id_map.json", "r") as fin:
            tweet_to_assertion_id = json.load(fin)

        counter = {}
        state_names = []
        colormap = {"View A": "#29b0ff", "View B": "#ff4929", "Neutral": "#b5b5b5", "Unknown": "#dedede"}
        for feature in geo_data["features"]:
            state_name = feature["properties"]["name"]
            state_names.append(state_name)
            counter[state_name] = {"name": state_name,
                                   "pro": 0,
                                   "anti": 0,
                                   "neutral": 0,
                                   "color": colormap["Unknown"],
                                   "caption": state_name}

        with open("/Users/lijinning/PycharmProjects/Polarization_dev/dataset/election/tweets.json", "r") as fin:
            for line in fin:
                js = json.loads(line.strip())
                location = js["user"]["location"]
                tweet_id = str(js["id"])
                if tweet_id in tweet_to_assertion_id.keys():
                    for state_name in state_names:
                        if location != "" and location.find(state_name) != -1:
                            pred = self.asser_clustering_pred[tweet_to_assertion_id[tweet_id]]
                            if pred == 1001:
                                counter[state_name]["anti"] += 1
                            elif pred == 1002:
                                counter[state_name]["pro"] += 1
                            else:
                                raise NotImplementedError()
                            print(location, pred)

        for item in counter.values():
            max_count = max(item["pro"], item["anti"], item["neutral"])
            if max_count == 0:
                item["color"] = colormap["Unknown"]
                item["caption"] = " [Unknown]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])
                continue

            if item["pro"] > 2.625 * item["anti"]:
                item["color"] = colormap["View A"]
                item["caption"] = " [View A]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])
            else:
                item["color"] = colormap["View B"]
                item["caption"] = " [View B]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])

            if item["neutral"] == max_count:
                item["color"] = colormap["Neutral"]
                item["caption"] = " [Neutral]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])

        result = {"data": counter, "colormap": colormap}
        with open(self.output_dir / "geo_data.json", "w") as fout:
            json.dump(result, fout)

    def evaluate_geo_eurovision(self):
        with open("/Users/lijinning/PycharmProjects/Polarization_dev/CIKM_visualization/utils/world.json", "r") as fin:
            geo_data = json.load(fin)

        with open(self.output_dir / "tweet_to_assertion_id_map.json", "r") as fin:
            tweet_to_assertion_id = json.load(fin)

        counter = {}
        state_names = []
        name_to_abbr = {}
        colormap = {"View A": "#29b0ff", "View B": "#ff4929", "Neutral": "#b5b5b5", "Unknown": "#dedede"}
        for feature in geo_data["features"]:
            state_name = feature["properties"]["name"]
            state_names.append(state_name)
            name_to_abbr[state_name] = feature["id"]
            counter[state_name] = {"name": state_name,
                                   "pro": 0,
                                   "anti": 0,
                                   "neutral": 0,
                                   "color": colormap["Unknown"],
                                   "caption": state_name}

        with open("/Users/lijinning/PycharmProjects/Polarization_dev/dataset/eurovision/tweets.json", "r") as fin:
            for line in fin:
                js = json.loads(line.strip())
                location = js["user"]["location"]
                tweet_id = str(js["id"])
                if tweet_id in tweet_to_assertion_id.keys():
                    for state_name in state_names:
                        if location != "" and (location.find(state_name) != -1 or location.find(name_to_abbr[state_name]) != -1):
                            pred = self.asser_clustering_pred[tweet_to_assertion_id[tweet_id]]
                            if pred == 1001:
                                counter[state_name]["anti"] += 1
                            elif pred == 1002:
                                counter[state_name]["pro"] += 1
                            else:
                                raise NotImplementedError()
                            # print(location, pred, js["text"].replace("\n", " "))

        for item in counter.values():
            max_count = max(item["pro"], item["anti"], item["neutral"])
            if max_count == 0:
                item["color"] = colormap["Unknown"]
                item["caption"] = " [Unknown]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])
                continue

            if item["pro"] > 1.25 * item["anti"]:
            # if item["pro"] > item["anti"]:
                item["color"] = colormap["View A"]
                item["caption"] = " [View A]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])
            else:
                item["color"] = colormap["View B"]
                item["caption"] = " [View B]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])

            if item["neutral"] == max_count:
                item["color"] = colormap["Neutral"]
                item["caption"] = " [Neutral]" + " " + item["name"] + " " + \
                                  "(View A: {}, View B: {}, Neutral: {}) ".format(item["pro"], item["anti"],
                                                                                  item["neutral"])

        result = {"data": counter, "colormap": colormap}
        with open(self.output_dir / "geo_data_world.json", "w") as fout:
            json.dump(result, fout)


    def dump_interface_data(self, init_csv_path, tweet2asserid_path, output_path):
        # TODO current pred only include pro and anti
        data = pd.read_csv(init_csv_path, sep="\t", dtype={'id': str, 'tweet_id': str, 'user_id': str})
        with open(tweet2asserid_path, "r") as fin:
            tweet2asserid = json.load(fin)
        # print(tweet2asserid)
        for i in range(len(data.index)):
            tweet_id = str(data.loc[i]["id"])
            if tweet_id not in tweet2asserid.keys():
                continue
            asser_id = tweet2asserid[tweet_id]
            label = self.asser_clustering_pred[asser_id]
            prob = abs(self.embedding[self.num_user + asser_id][0])
            if label == 0:
                data.loc[i, "label"] = "anti"
                data.loc[i, "anti_prob"] = prob
            elif label == 1:
                data.loc[i, "label"] = "pro"
                data.loc[i, "pro_prob"] = prob
            else:
                data.loc[i, "label"] = "unknown"
        data.to_csv(output_path, sep=",", encoding="utf-8")
        with open(output_path.replace(".csv", ".pkl"), "wb") as fout:
            pickle.dump(data, fout)
        print("Dump interface data success {}".format(output_path))

    def dump_text_result(self):
        assert self.user_clustering_pred is not None
        assert self.asser_clustering_pred is not None
        save_path = self.output_dir / "text_result.txt"
        with open(save_path, "w") as fout:
            for i in range(self.num_asser):
                if self.asser_label[i] == 1000:
                    continue
                fout.write("label:{} pred: {} | assertion: {}\n".format(
                    self.asser_label[i], self.asser_clustering_pred[i], self.asserlist[i]
                ))
            for i in range(self.num_user):
                if self.user_label[i] == 0:
                    continue
                fout.write("label:{} pred: {} | user: {}\n".format(
                    self.user_label[i], self.user_clustering_pred[i], self.namelist[i]
                ))
        print("Dump text result to {}".format(save_path))

    def dump_topk_json(self, K="all"):
        save_path = self.output_dir / "top_{}_claims.json".format(K)
        res = {"group1": [], "group2": []}

        # Tweet ranking with dim1
        collection = [(
            [self.asserlist[i], str(self.asser_label[i])],
            self.embedding[self.num_user + i][0]
        ) for i in range(self.num_asser) if self.embedding[self.num_user + i][0] > self.embedding[self.num_user + i][1]]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection if K == "all" else collection[:K]:
            res["group1"].append(item[0])

        # Tweet ranking with dim2
        collection = [(
            [self.asserlist[i], str(self.asser_label[i])],
            self.embedding[self.num_user + i][1]
        ) for i in range(self.num_asser) if self.embedding[self.num_user + i][1] > self.embedding[self.num_user + i][0]]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection if K == "all" else collection[:K]:
            res["group2"].append(item[0])

        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(res, fout, indent=2, ensure_ascii=False)

    def dump_topk_json_user(self, K=50):
        save_path = self.output_dir / "top{}_user.json".format(K)
        res = {"reprA": [], "reprB": [], "reprC": [], "Euclidean": []}

        # Tweet ranking with dim1
        collection = [(
            self.namelist[i],
            self.embedding[i][0]
        ) for i in range(self.num_user)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprA"].append(item[0])

        # Tweet ranking with dim2
        collection = [(
            self.namelist[i],
            self.embedding[i][1]
        ) for i in range(self.num_user)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprB"].append(item[0])

        # Tweet ranking with dim3
        collection = [(
            self.namelist[i],
            self.embedding[i][2]
        ) for i in range(self.num_user)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprC"].append(item[0])

        # Tweet ranking with E distance to neutral
        collection = [(
            self.namelist[i],
            self.embedding[i][2]
        ) for i in range(self.num_user)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["Euclidean"].append(item[0])

        with open(save_path, "w") as fout:
            json.dump(res, fout, indent=2)

    # def dump_topk_text_result(self, K=20):
    #     assert self.user_clustering_pred is not None
    #     assert self.asser_clustering_pred is not None
    #     save_path = self.output_dir / "top{}_text_result.txt".format(K)
    #     with open(save_path, "w") as fout:
    #         for dim in range(self.embedding.shape[1]):
    #             fout.write("\nAssertion Top{} in Dim{}\n"
    #                        "------------------------------------------------------------------\n".format(K, dim + 1))
    #             collection = [(self.asser_label[i], self.asser_clustering_pred[i],
    #                            self.asserlist[i], self.embedding[self.num_user + i][dim])
    #                           for i in range(self.num_asser)]
    #             collection = sorted(collection, key=lambda x: x[3], reverse=True)
    #             for item in collection[:K]:
    #                 fout.write("label:{} pred: {} dim{}: {:.4f} | assertion: {}\n".format(item[0] - 1000, item[1] - 1000,
    #                                                                                   dim + 1, item[3],
    #                                                                                   item[2]))
    #     print("Dump text topk to {}".format(save_path))

    def dump_topk_text_result(self, K=20):
        assert self.user_clustering_pred is not None
        assert self.asser_clustering_pred is not None
        save_path = self.output_dir / "top{}_text_result.txt".format(K)
        with open(save_path, "w") as fout:
            fout.write("\nAssertion Top{} of Neutral\n"
                       "------------------------------------------------------------------\n".format(K))
            neutral_collection = [(
                self.asser_label[i] - 1000,
                self.asserlist[i],
                self.embedding[self.num_user + i][0] - np.abs(self.embedding[self.num_user + i][1] - self.embedding[self.num_user + i][2])
            ) for i in range(self.num_asser)]
            neutral_collection = sorted(neutral_collection, key=lambda x: x[2], reverse=True)
            for item in neutral_collection[:K]:
                fout.write("label: {}, Dim1-|Dim2-Dim3|={:.4f} | assertion: {}\n".format(
                    item[0] if item[0] != 0 else "?", item[2], item[1]))

            fout.write("\nAssertion Top{} of One Polarity\n"
                       "------------------------------------------------------------------\n".format(K))
            p1_collection = [(
                self.asser_label[i] - 1000,
                self.asserlist[i],
                self.embedding[self.num_user + i][1] - (self.embedding[self.num_user + i][0] + self.embedding[self.num_user + i][2])
            ) for i in range(self.num_asser)]
            p1_collection = sorted(p1_collection, key=lambda x: x[2], reverse=True)
            for item in p1_collection[:K]:
                fout.write("label: {}, Dim2-(Dim1+Dim3)={:.4f} | assertion: {}\n".format(
                                    item[0] if item[0] != 0 else "?", item[2], item[1]))

            fout.write("\nAssertion Top{} of Another Polarity\n"
                       "------------------------------------------------------------------\n".format(K))
            p2_collection = [(
                self.asser_label[i] - 1000,
                self.asserlist[i],
                self.embedding[self.num_user + i][2] - (self.embedding[self.num_user + i][0] + self.embedding[self.num_user + i][1])
            ) for i in range(self.num_asser)]
            p2_collection = sorted(p2_collection, key=lambda x: x[2], reverse=True)
            for item in p2_collection[:K]:
                fout.write("label: {}, Dim3-(Dim1+Dim2)={:.4f} | assertion: {}\n".format(
                                    item[0] if item[0] != 0 else "?", item[2], item[1]))

        print("Dump text topk to {}".format(save_path))

    def plot(self, permulate=None, show=False, save=True, note=""):
        print("Evaluator plot label with config:")
        print("user_plot_config: " + str(self.user_plot_config))
        print("asser_plot_config: " + str(self.asser_plot_config))
        label = np.concatenate([self.user_label, self.asser_label], axis=0)
        if self.embedding.shape[1] == 1:
            self.plot_1d(self.embedding, label, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 2:
            self.plot_2d(self.embedding, label, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 3:
            # label[
            #     (self.embedding[:, 1] < 0.1) & (self.embedding[:, 2] > 1.5) & (self.embedding[:, 0] < 0.5) & (label[:] == 1002)
            #     ] = 1001
            # label[
            #     (self.embedding[:, 1] < 0.1) & (self.embedding[:, 2] > 1.5) & (self.embedding[:, 0] < 0.5) & (label[:] == 2)
            #     ] = 1
            self.plot_3d(self.embedding, label, self.user_plot_config, self.asser_plot_config, permulate, show, save, note)

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    # def purity_score(self, y_true, y_pred):
    #     """Purity score
    #         Args:
    #             y_true(np.ndarray): n*1 matrix Ground truth labels
    #             y_pred(np.ndarray): n*1 matrix Predicted clusters
    #
    #         Returns:
    #             float: Purity score
    #     """
    #     # matrix which will hold the majority-voted labels
    #     y_voted_labels = np.zeros(y_true.shape)
    #     # Ordering labels
    #     ## Labels might be missing e.g with set like 0,2 where 1 is missing
    #     ## First find the unique labels, then map the labels to an ordered set
    #     ## 0,2 should become 0,1
    #     labels = np.unique(y_true)
    #     ordered_labels = np.arange(labels.shape[0])
    #     for k in range(labels.shape[0]):
    #         y_true[y_true==labels[k]] = ordered_labels[k]
    #     # Update unique labels
    #     labels = np.unique(y_true)
    #     # We set the number of bins to be n_classes+2 so that
    #     # we count the actual occurence of classes between two consecutive bins
    #     # the bigger being excluded [bin_i, bin_i+1[
    #     bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    #
    #     for cluster in np.unique(y_pred):
    #         hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
    #         # Find the most present label in the cluster
    #         winner = np.argmax(hist)
    #         y_voted_labels[y_pred==cluster] = winner
    #
    #     return accuracy_score(y_true, y_voted_labels)

    def overall_accuracy(self, label, pred):
        return np.sum(np.array(label) == np.array(pred)) / np.array(label).shape[0]

    def precision_purity(self, label, pred, which):
        l = label[pred == which]
        return np.sum(l == which) / l.shape[0]

    # purity = precision
    def numerical_evaluate(self):
        asser_pred = self.asser_clustering_pred[self.asser_label != 999] - 1
        asser_pred_reverse = asser_pred.copy()  # 1000 --> 1001; 1001 --> 1000
        asser_pred_reverse = 1 - (asser_pred_reverse - 1000) + 1000
        asser_label = self.asser_label[self.asser_label != 999]

        # not reversed
        acc = self.overall_accuracy(asser_label, asser_pred)
        f1 = f1_score(asser_label, asser_pred, average="macro")
        purity_score = [self.precision_purity(asser_label, asser_pred, which=1000), self.precision_purity(asser_label, asser_pred, which=1001)]

        # reversed
        acc_reversed = self.overall_accuracy(asser_label, asser_pred_reverse)
        f1_reversed = f1_score(asser_label, asser_pred_reverse, average="macro")
        purity_score_reversed = [self.precision_purity(asser_label, asser_pred_reverse, which=1000), self.precision_purity(asser_label, asser_pred_reverse, which=1001)]

        if acc >= acc_reversed:
            print("\nAccuracy: {:.4f}, Macro F1: {:.4f}, Avg Purity: {:.4f}\n".format(acc, f1, np.average(purity_score)))
            return acc, f1, np.average(purity_score)
        else:
            print("\nAccuracy: {:.4f}, Macro F1: {:.4f}, Avg Purity: {:.4f}\n".format(acc_reversed, f1_reversed, np.average(purity_score_reversed)))
            return acc_reversed, f1_reversed, np.average(purity_score_reversed)

    # -------------------------------- Function Utils --------------------------------

    def positive(self, embedding):
        if embedding.shape[1] == 1:
            print(embedding[:][0] > 0).astype("int32")
        if embedding.shape[1] == 2:
            pred = []
            for i in range(embedding.shape[0]):
                if embedding[i][0] > 0 and embedding[i][1] > 0:
                    pred.append(1)
                else:
                    pred.append(0)
            return np.array(pred)
        if embedding.shape[1] == 3:
            pred = []
            for i in range(embedding.shape[0]):
                if embedding[i][0] > 0 and embedding[i][1] > 0 and embedding[i][2] > 0:
                    pred.append(1)
                else:
                    pred.append(0)
            return np.array(pred)

    def positive_d2(self, embedding):
        assert embedding.shape[1] == 3
        pred = []
        for i in range(embedding.shape[0]):
            if embedding[i][1] > 0:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)

    def positive_d2d3(self, embedding):
        assert embedding.shape[1] == 3
        pred = []
        for i in range(embedding.shape[0]):
            if embedding[i][1] < 0 and embedding[i][2] > 0:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)

    def dbscan(self, embedding, cosine_norm=False, eps=0.5, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length
        clustering = db.fit(embedding)
        pred = clustering.labels_
        pred[pred < 0] = 0
        return pred, clustering

    def mean_shift(self, embedding, cosine_norm=False, bandwidth=None):
        ms = MeanShift(bandwidth=bandwidth)
        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length
        clustering = ms.fit(embedding)
        return clustering.labels_, clustering

    def k_means(self, embedding, cosine_norm=False, n_clusters=2, n_init=10):
        km = KMeans(
            n_clusters=n_clusters, n_init=n_init
        )

        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length

        km_result = km.fit_predict(embedding)
        return km_result, km

    def time_tag(self):
        return time.strftime("%Y%m%d%H%M%S_", time.localtime()) + str(time.time()).split(".")[1]

    def plot_1d(self, embedding, label, user_plot_config, asser_plot_config, show=False, save=True):
        assert embedding.shape[1] == 1
        assert embedding.shape[0] == label.shape[0]

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1),
                        np.zeros(emb[:, 0].reshape(-1).shape) + 0.15 * np.random.random(
                            size=emb[:, 0].reshape(-1).shape),
                        marker="o", color=c, label=t, s=10)

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1),
                        np.ones(emb[:, 0].reshape(-1).shape) + 0.15 * np.random.random(
                            size=emb[:, 0].reshape(-1).shape),
                        marker="^", color=c, label=t, s=10)

        plt.tick_params(labelsize=14)
        plt.legend(loc='best', prop={'size': 14})
        if save:
            plt.savefig(self.output_dir / "1d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
        if show:
            plt.show()

    def plot_2d(self, embedding, label, user_plot_config, asser_plot_config, show=False, save=True):
        assert embedding.shape[1] == 2
        assert embedding.shape[0] == label.shape[0]

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            # emb += np.random.random(size=emb.shape) * 0.15
            plt.scatter(emb[:, 0].reshape(-1), emb[:, 1].reshape(-1), marker="o", color=c, label=t, s=10)

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            # emb += np.random.random(size=emb.shape) * 0.15
            plt.scatter(emb[:, 0].reshape(-1), emb[:, 1].reshape(-1), marker="^", color=c, label=t, s=10)

        plt.tick_params(labelsize=16)
        plt.legend(loc='best', prop={'size': 14})
        if save:
            plt.savefig(self.output_dir / "2d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
        if show:
            plt.show()

    def plot_3d_init(self, embedding, label, user_plot_config, asser_plot_config, permulate=None, show=False, save=True, note=""):
        if permulate is None:
            permulate = [0, 1, 2]
        assert embedding.shape[1] == 3
        assert embedding.shape[0] == label.shape[0]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # with open(self.output_dir / "3dplot_log_{}.txt".format(self.time_tag()), "w") as fout:
        #     k = 0

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            # emb += np.random.random(size=emb.shape) * 0.05
            ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
                       marker="o", color=c,
                       # alpha=0.4,
                       # label=t)
                       s=4,
                       # edgecolors='none',
                       label="Actors")

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            # emb += np.random.random(size=emb.shape) * 0.05
            ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
                       marker="^", color=c,
                       # alpha=0.4,
                       # label=t)
                       s=4,
                       # edgecolors='none',
                       label="Messages")

        # plt.legend(loc='upper right', prop={'size': 14})
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        # ax.view_init(20, 30)
        # ax.view_init(20, 120)

        # ax.view_init(20, 150)
        # ax.view_init(30, 210)
        ax.view_init(20, 30)
        # ax.view_init(60, 210)
        # ax.view_init(80, 180)
        # ax.view_init(20, 300)

        plt.legend()

        if save:
            # plt.savefig(self.output_dir / "3d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
            plt.savefig(self.output_dir / "3d_evaluation_cluster_{}_{}.pdf".format(note, self.time_tag()), bbox_inches='tight')
        if show:
            plt.show()

    def plot_3d(self, embedding, label, user_plot_config, asser_plot_config, permulate=None, show=False, save=True, note=""):
        if permulate is None:
            permulate = [0, 1, 2]
        assert embedding.shape[1] == 3
        assert embedding.shape[0] == label.shape[0]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # with open(self.output_dir / "3dplot_log_{}.txt".format(self.time_tag()), "w") as fout:
        #     k = 0

        # for l, c, t in user_plot_config:
        #     emb = embedding[label == l]
        #     # emb += np.random.random(size=emb.shape) * 0.05
        #     ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
        #                marker=".", color=c,
        #                # alpha=0.4,
        #                # label=t)
        #                s=2,
        #                # edgecolors='none',
        #                label="Actors")

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            # emb += np.random.random(size=emb.shape) * 0.05
            ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
                       marker=".", color=c,
                       # alpha=0.4,
                       # label=t)
                       s=2,
                       # edgecolors='none',
                       label="Messages")

        # plt.legend(loc='upper right', prop={'size': 14})
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        # ax.view_init(20, 30)
        # ax.view_init(20, 120)

        # ax.view_init(20, 150)
        # ax.view_init(20, 210)
        ax.view_init(20, 30)
        # ax.view_init(60, 210)
        # ax.view_init(80, 180)
        # ax.view_init(20, 300)

        plt.legend()

        if save:
            # plt.savefig(self.output_dir / "3d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
            plt.savefig(self.output_dir / "3d_evaluation_cluster_{}_{}.pdf".format(note, self.time_tag()), bbox_inches='tight')
        if show:
            plt.show()

if __name__ == "__main__":
    evaluator = Evaluator(use_b_matrix=False,
    user_plot_config = [
        [1, "#178bff", "User Con"],
        # [0, "#3b3b3b", "User Neu"],
        # [2, "#ff5c1c", "User Pro"]
    ],
    asser_plot_config = [
        # [1, "#30a5ff", "Assertion Con"],
        [1, "#30a5ff", "Assertion Con"],
        # [0, "#4d4d4d", "Assertion Neu"],
        # [2, "#fc8128", "Assertion Pro"]
    ])
    evaluator.init_from_dir(
        "/Users/lijinning/PycharmProjects/incas_interface/output/HVGAE_ukraine_3D_20220426011555"
    )
    evaluator.plot(show=True)
    exit()
    evaluator.run_clustering()
    evaluator.plot_clustering(show=False)
    evaluator.dump_text_result()
    evaluator.numerical_evaluate()
    evaluator.dump_topk_json()
    evaluator.dump_topk_json_user()

