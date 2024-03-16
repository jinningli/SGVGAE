import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import json
import torch
import pickle
import itertools
import random
import pathlib
from feature_builder import DiagFeatureBuilder, TfidfEmbeddingVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class ApolloDataset:
    def __init__(self, pickle_path, args=None):
        self.args = args
        self.name = "ApolloDataset"
        self.pickle_path = pickle_path
        self.processed_data = None
        self.user_label = None
        self.asser_label = None
        self.asserlist = None
        # self.tweet_id_list = None
        self.num_user = None
        self.num_assertion = None
        self.num_nodes = None
        self.timelist = None
        self.name_list = None
        self.tweetid2asserid = None
        self.tweeting_matrix = None

        self.freeze_dict = None  # tweet_id (merged) to embedding
        self.freeze_mask = None
        self.freeze_tensor = None
        self.semi_indexs = [[], []]

        print("Preprocess and dump dataset...")
        self.preprocessing()

    def get_feature_similarity_matrix(self):
        normed_t2k = self.feature_builder.tweets2keywords / self.feature_builder.tweets2keywords.sum(axis=1).reshape(-1, 1)
        # normed_t2k = self.feature_builder.tweets2keywords
        return normed_t2k @ normed_t2k.transpose()

    def preprocessing(self):
        # Preprocessing
        if self.pickle_path.find(".csv") != -1:
            self.data = pd.read_csv(self.pickle_path)
        else:
            self.data = pd.read_parquet(self.pickle_path)

        if self.args.prev_artifact_dir is not None:
            prev_artifact_path = self.args.prev_artifact_dir / f"infovgae_artifact_{self.args.hidden2_dim}.pkl"
            if prev_artifact_path.exists():
                with open(prev_artifact_path, "rb") as fin:
                    self.freeze_dict = pickle.load(fin)
                print("USING freeze")

        #self.data = pd.read_csv(self.csv_path, sep='\t')
        print("Data Read Done")
        print(self.data.iloc[0])
        self.processed_data = self.data.rename(columns={'index_text': 'postTweet', 'message_id': 'tweet_id', 'actor_id': 'name', 'text': 'rawTweet'})
        # self.processed_data.to_csv("tmp.csv", sep="\t", index=False)
        print(self.processed_data.head)
        print("Text Process Done")
        print(self.processed_data)
        self.name_list = self.processed_data.name.unique().tolist()
        # print(self.name_list)
        # Feature builder for index mapping
        self.feature_builder = DiagFeatureBuilder(processed_data=self.processed_data)
        print("Feature_builder Init Done")
        self.feature_builder.build_index_mapping_only()
        print("Feature Builder Done")
        self.num_user = len(self.processed_data["name"].unique())
        self.num_assertion = len(self.processed_data["postTweet"].unique())
        self.num_nodes = self.num_user + self.num_assertion
        self.build_labels()
        print("Label Build Done")

    def build_labels(self):
        assert self.processed_data is not None
        num_assertion = len(self.feature_builder.tweet2index)

        # calculate tweet assertion label
        self.asser_label = np.zeros(num_assertion).astype("int32")
        self.asserlist = [None for _ in range(num_assertion)]
        self.tweetlist = [None for _ in range(num_assertion)]
        # self.tweet_id_list = [None for _ in range(num_assertion)]
        self.timelist = [None for _ in range(num_assertion)]
        label_mapping = {v: i for i, v in enumerate(self.args.label_types.split(","))}
        for i, item in self.processed_data.iterrows():
            label = label_mapping[item["manual_label"]] if ("manual_label" in self.processed_data.columns and item["manual_label"] in label_mapping) else -1
            postTweet = item["postTweet"]
            tweet_id = self.feature_builder.tweet2index[postTweet]
            # if self.tweet_id_list[tweet_id] is None:
            #     self.tweet_id_list[tweet_id] = item["tweet_id"]

            # """"""
            # # TODO Remove this (only for incas)
            # time_published = item["time_published"]
            # """"""
            # if self.timelist[tweet_id] is None:
            #     self.timelist[tweet_id] = time_published
            # else:
            #     self.timelist[tweet_id] = min(self.timelist[tweet_id], time_published)

            if self.asserlist[tweet_id] is None:
                self.asserlist[tweet_id] = item["postTweet"]
            if self.tweetlist[tweet_id] is None:
                self.tweetlist[tweet_id] = item["rawTweet"]
            if self.asser_label[tweet_id] == 0:
                self.asser_label[tweet_id] = label

        # get freeze mask
        freeze_tensor_list = []
        if self.freeze_dict is not None:
            self.freeze_mask = np.zeros(self.num_nodes)
            for i in range(self.num_assertion):
                if self.asserlist[i] in self.freeze_dict:
                    print(self.asserlist[i])
                    self.freeze_mask[i + self.num_user] = 1
                    freeze_tensor_list.append(self.freeze_dict[self.asserlist[i]].reshape(1, -1))
            if freeze_tensor_list:
                self.freeze_tensor = np.concatenate(freeze_tensor_list, axis=0)
                self.freeze_tensor = torch.from_numpy(self.freeze_tensor)
            else:
                print("No freeze overlap")
                self.freeze_tensor = None
            self.freeze_mask = self.freeze_mask.astype("bool")
            print("Freeze Overlap: {} / {}".format(np.sum(self.freeze_mask), self.num_assertion))

        # calculate tweet label
        num_user = len(self.feature_builder.user2index)
        self.user_label = np.zeros(num_user).astype("int32")
        user_label_candidate = [[] for _ in range(num_user)]
        for i, item in self.processed_data.iterrows():
            label = 1 #item["label"]
            user_name = item["name"]
            user_index = self.feature_builder.user2index[user_name]
            if label != 0:
                user_label_candidate[user_index].append(label)
        for i in range(num_user):
            if not user_label_candidate[i]:
                self.user_label[i] = 0
            else:
                self.user_label[i] = Counter(user_label_candidate[i]).most_common(1)[0][0]

        # build tweet to assertion id mapping
        self.tweetid2asserid = {}
        self.original_tweetid2asserid = {}
        for i, item in self.processed_data.iterrows():
            postTweet = item["postTweet"]
            try:
                tweet_id = str(item["id"])
            except KeyError:
                tweet_id = str(item["tweet_id"])
            assertion_id = self.feature_builder.tweet2index[postTweet]
            self.tweetid2asserid[tweet_id] = assertion_id

            original_tweet_id = str(item["tweet_id"])
            self.original_tweetid2asserid[original_tweet_id] = assertion_id

    def compute_tweet_tweet_matrix(self, label_types, label_sampling):
        """
        label_types: ["pro", "anti"]
        label_sampling: [0.3, 0.7]
        """
        indexs = []
        tt_matrix = np.zeros((self.num_assertion, self.num_assertion))
        self.processed_data["semi_label"] = self.processed_data.apply(lambda x: x["gpt_label"] if x["is_gt"] == 0 else np.nan, axis=1)
        print(self.processed_data)
        if "label" not in self.processed_data.columns and "gpt_label" not in self.processed_data.columns:
            raise NotImplementedError("Warning: label is needed when doing semi-supervision. Will skip supervision now")
        for i, label_type in enumerate(label_types):
            df = self.processed_data[self.processed_data["semi_label"] == label_type]
            index = list(df["postTweet"].apply(lambda x: self.feature_builder.tweet2index[x]).unique())
            if label_sampling[i] < 1:
                sampled_size = int(len(index) * label_sampling[i])  # now percentage of semi_label we have
                if sampled_size > len(index):
                    sampled_size = len(index)
                    print("Cannot do because no enough label, using full index")
                print(f"Label sampling {label_type} with ratio {label_sampling[i]} {len(index)} --> {sampled_size}")
                index = random.sample(index, k=sampled_size)
                indexs.append(index)
                print(index)
            else:
                print(f"No label sampling {label_type} {len(index)} --> {len(index)}")
                indexs.append(index)
            # print(list(itertools.combinations(index, 2)))
            for s, t in itertools.combinations(index, 2):
                tt_matrix[s][t] += 1
                tt_matrix[t][s] += 1
        return tt_matrix, indexs

    def build(self):
        print("{} Building...".format(self.name))

        self.num_user = len(self.feature_builder.user2index)
        self.num_assertion = len(self.feature_builder.tweet2index)
        self.num_nodes = self.num_user + self.num_assertion
        # Heterogeneous adjacent matrix
        self.het_matrix = sp.lil_matrix((self.num_nodes, self.num_nodes))

        # Get tweeting matrix
        self.tweeting_matrix = self.get_tweeting_matrix(self.processed_data, self.num_user, self.num_assertion)
        self.het_matrix[:self.num_user, self.num_user:self.num_user + self.num_assertion] = self.tweeting_matrix
        self.het_matrix[self.num_user:self.num_user + self.num_assertion, :self.num_user] = self.tweeting_matrix.transpose()

        if self.args.edge_guidance or self.args.axis_guidance:
            print("semi_supervised")
            tt_matrix, self.semi_indexs = \
                self.compute_tweet_tweet_matrix(self.args.label_types.split(","),
                                                [float(x) for x in self.args.label_sampling.split(",")])  # semi_indexs are not added num_user
            if self.args.edge_guidance:
                self.het_matrix[self.num_user:, self.num_user:] = tt_matrix

            if self.args.axis_guidance:
                assert self.args.hidden2_dim == 2
                index_cnts = [len(ind) for ind in self.semi_indexs]
                self.axis_guidance_indexes = [ind + self.num_user for ind in self.semi_indexs[0]] + [ind + self.num_user for ind in self.semi_indexs[1]]
                self.axis_guidance_units = torch.zeros(size=(len(self.axis_guidance_indexes), self.args.hidden2_dim))
                self.axis_guidance_units[:index_cnts[0], 0] = 1.0
                self.axis_guidance_units[index_cnts[0]:, 1] = 1.0
                self.axis_guidance_adj_matrix = torch.zeros(size=(len(self.axis_guidance_indexes), len(self.axis_guidance_indexes)))
                self.axis_guidance_adj_matrix[:index_cnts[0], :index_cnts[0]] = 1.0
                self.axis_guidance_adj_matrix[index_cnts[0]:, index_cnts[0]:] = 1.0
                self.axis_guidance_N = len(self.axis_guidance_indexes)

        # Get tweet similarity matrix
        # print("USING TWEET_TWEET GRAPH")
        # tweet_tweet_matrix = self.get_DMG_tweet_tweet_matrix(
        #             # dictionary_dir="/home/jinning4/data/text_mining",
        #             dictionary_dir="./data/text_mining",
        #             method="cosine"
        #         )
        # print(tweet_tweet_matrix[:10, :10])
        # self.het_matrix[self.num_user:, self.num_user:] = tweet_tweet_matrix

        print("{} Processing Done. num_user: {}, num_assertion: {}".format(self.name, self.num_user, self.num_assertion))
        # Return adj matrix
        return self.het_matrix

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-07)

    def get_tweet_similarity_matrix(self, processed_data):
        X = []
        for tweet in processed_data.postTweet.unique():
            X.append(tweet.split(" "))
        builder = TfidfEmbeddingVectorizer(X=X)
        X_emb = builder.transform(X)
        X_emb = np.array(X_emb)
        similarity_matrix = np.array(cosine_similarity(X_emb, dense_output=True))
        return similarity_matrix

    def get_mention_matrix(self, processed_data, num_user):
        assert self.name_list is not None
        mention_matrix = np.zeros((num_user, num_user))
        for i, item in processed_data.iterrows():
            user_name = item["name"]
            init_tweet = item["rawTweet"]
            splits = init_tweet.split(" ")
            mention_name = None
            if splits[0] == "RT":
                mention_name = splits[1].strip().replace("@", "").replace(":", "")
            if mention_name is not None and user_name in self.name_list and mention_name in self.name_list:
                mention_matrix[self.name_list.index(user_name)][self.name_list.index(mention_name)] += 1
                mention_matrix[self.name_list.index(mention_name)][self.name_list.index(user_name)] += 1
        return mention_matrix

    def get_tweeting_matrix(self, processed_data, num_user, num_assertion):
        tweeting_matrix = np.zeros((num_user, num_assertion))
        for i, item in processed_data.iterrows():
            postTweet = item["postTweet"]
            tweet_index = self.feature_builder.tweet2index[postTweet]
            user_name = item["name"]
            user_index = self.feature_builder.user2index[user_name]
            tweeting_matrix[user_index][tweet_index] += 1
        return tweeting_matrix

    def dump_label(self):
        # dump label to label.bin
        save_path = self.args.output_path / "label.bin"
        with open(save_path, "wb") as fout:
            pickle.dump({"user_label": self.user_label, "assertion_label": self.asser_label}, fout)

        # dump the representative assertion list
        save_path = self.args.output_path / "asserlist.json"
        with open(save_path, "w", encoding="utf-16") as fout:
            json.dump(self.asserlist, fout, indent=2, ensure_ascii=False)

        save_path = self.args.output_path / "tweetlist.json"
        with open(save_path, "w", encoding="utf-16") as fout:
            json.dump(self.tweetlist, fout, indent=2, ensure_ascii=False)

        save_path = self.args.output_path / "timelist.json"
        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(self.timelist, fout, indent=2)

        # dump namelist for evaluation
        with open(self.args.output_path / "namelist.json", 'w') as fout:
            json.dump(self.name_list, fout, indent=2)

        # dump tweet to assertion id mapping
        with open(self.args.output_path / "tweet_to_assertion_id_map.json", 'w') as fout:
            json.dump(self.tweetid2asserid, fout, indent=2)

        # Dump adjacency matrix
        sp.save_npz(self.args.output_path / "h_adj_matrix.npz", self.het_matrix.tocsr())

        # TODO(jinning) remove this; dump adj matrix for testing
        # with open(self.args.output_path / "tweeting_matrix.bin", 'wb') as fout:
        #     pickle.dump(self.get_tweeting_matrix(self.processed_data, self.num_user, self.num_assertion), fout)

        print("Dump Label file success {}".format(save_path))


    def dump_processed_data(self):
        self.processed_data.to_csv(self.args.output_path / "processed_data.csv",
                                   sep="\t", encoding="utf-8", index=False)
        print("Dump processed data success {}".format(self.args.output_path / "processed_data.csv"))

