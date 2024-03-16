"""
 python3 main.py --data_path ${dataset} --exp_name exp --axis_guidance --edge_guidance --hidden2_dim 2 --label_types supportive,opposing --learning_rate 0.2 --label_sampling 0.05,0.05 --device 0 --seed 0
"""
import os.path
import random
import time

import torch
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pandas as pd

from dataset import ApolloDataset
from model_trainer import ControlVGAETrainer
from evaluate import Evaluator

parser = argparse.ArgumentParser()

# General
parser.add_argument('--model', type=str, default="VGAE", help="model to use")
parser.add_argument('--epochs', type=int, default=500, help='epochs (iterations) for training')
parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate of model')
parser.add_argument('--device', type=str, default="cpu", help='cpu/gpu device')
parser.add_argument('--num_process', type=int, default=40, help='num_process for pandas parallel')

# For ICCCN
parser.add_argument('--edge_guidance', action="store_true")
parser.add_argument('--axis_guidance', action="store_true")
parser.add_argument('--label_types', type=str, default="pro,anti")
parser.add_argument('--label_sampling', type=str, default="1,1", help="label_sampling percentage ")

# Data
parser.add_argument('--exp_name', type=str, help='exp_name to use', required=True)
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--year', type=int, default=None, help='year')
parser.add_argument('--add_self_loop', type=bool, default=True, help='add self loop for adj matrix')
parser.add_argument('--directed', type=bool, default=False, help='use directed adj matrix')
parser.add_argument('--data_path', type=str, default=None, help='specify the data path', required=True)
parser.add_argument('--data_json_path', type=str, default=None)
parser.add_argument('--friend_path', type=str, default=None)
parser.add_argument('--stopword_path', type=str, default=None)
parser.add_argument('--keyword_path', type=str, default="N")
parser.add_argument('--pos_weight_lambda', type=float, default=5.0, help='Lambda for positive sample weight')

# For GAE/VGAE model
parser.add_argument('--hidden1_dim', type=int, default=32, help='graph conv1 dim')
parser.add_argument('--hidden2_dim', type=int, default=3, help='graph conv2 dim')
parser.add_argument('--use_feature', type=bool, default=True, help='Use feature')
parser.add_argument('--use_b_matrix', action="store_true", help='using B matrix for GAE and VGAE model')
parser.add_argument('--seed', type=int, default=None)

# Embeddding Freeze
parser.add_argument('--artifact_dir', type=str, default=None, help='dir to save pickle of freeze dict (tweet only)')
parser.add_argument('--prev_artifact_dir', type=str, default=None, help='dir to load pickle of freeze dict (tweet only)')

args = parser.parse_args()

if args.artifact_dir is not None:
    args.artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    setattr(args, "output_path", args.artifact_dir / "infovgae" / f"belief_embedding_output_{args.exp_name}_{args.hidden2_dim}")
else:
    setattr(args, "output_path", Path(f"./belief_embedding_output_{args.exp_name}_{args.hidden2_dim}"))

if args.prev_artifact_dir is not None:
    args.prev_artifact_dir = Path(args.prev_artifact_dir).expanduser().resolve()

args.output_path.mkdir(parents=True, exist_ok=True)

# Setting the device
if not torch.cuda.is_available():
    args.device = torch.device('cpu')
else:
    args.device = torch.device(int(args.device) if args.device.isdigit() else args.device)
print("Device: {}".format(args.device))

# Setting the random seeds
if args.seed is not None:
    print("set seed")
    random.seed(a=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.device != torch.device('cpu'):
    #     print("seed2")
    torch.cuda.manual_seed(args.seed)

# Prepare dataset (Use ApolloDataset for incas)
dataset = ApolloDataset(pickle_path=args.data_path, args=args)
adj_matrix = dataset.build()
setattr(args, "num_user", dataset.num_user)
setattr(args, "num_assertion", dataset.num_assertion)
# dump label and namelist for evaluation
dataset.dump_label()

# Start Training
try:
    feature = sp.diags([1.0], shape=(dataset.num_nodes, dataset.num_nodes))
    setattr(args, "input_dim", dataset.num_nodes)
    trainer = ControlVGAETrainer(adj_matrix, feature, args, dataset)
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    running_time = end_time - start_time
    trainer.save()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("WARNING: ran out of vram, using cpu")
        setattr(args, "device", "cpu")
        feature = sp.diags([1.0], shape=(dataset.num_nodes, dataset.num_nodes))
        setattr(args, "input_dim", dataset.num_nodes)
        trainer = ControlVGAETrainer(adj_matrix, feature, args, dataset)
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        running_time = end_time - start_time
        trainer.save()
    else:
        raise e

# Start Evaluation (Apollo dataset)
print("Running Evaluation ...")
evaluator = Evaluator(use_b_matrix=args.use_b_matrix)
evaluator.init_from_value(trainer.result_embedding, dataset.user_label, dataset.asser_label,
                          dataset.name_list, dataset.tweetlist,
                          B_matrix=
                          None,
                          output_dir=args.output_path)
evaluator.plot(show=False, save=True)

evaluator.run_clustering()
evaluator.plot_clustering(show=False)
# evaluator.dump_text_result()
acc, macro_f1, avg_purity = evaluator.numerical_evaluate()

if not os.path.exists("records.csv"):
    with open("records.csv", "a") as fout:
        fout.write("dataset,semi_cnt_0,semi_cnt_1,num_assertion,seed,edge_guidance,axis_guidance,running_time,accuracy,macro_f1,average_purity\n")
with open("records.csv", "a") as fout:
    fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
        args.data_path,
        len(dataset.semi_indexs[0]),
        len(dataset.semi_indexs[1]),
        args.num_assertion,
        args.seed,
        args.edge_guidance,
        args.axis_guidance,
        running_time,
        acc,
        macro_f1,
        avg_purity
    ))

with open(args.output_path / "records.csv", "w") as fout:
    fout.write(
        "dataset,semi_cnt_0,semi_cnt_1,num_assertion,seed,edge_guidance,axis_guidance,running_time,accuracy,macro_f1,average_purity\n")
    fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
        args.data_path,
        len(dataset.semi_indexs[0]),
        len(dataset.semi_indexs[1]),
        args.num_assertion,
        args.seed,
        args.edge_guidance,
        args.axis_guidance,
        running_time,
        acc,
        macro_f1,
        avg_purity
    ))

with open("score.txt", "w") as fout:
    fout.write("{}".format(acc))

evaluator.dump_topk_json()
# evaluator.dump_topk_json_user()

# Dump top messages
m = dataset.original_tweetid2asserid
df = dataset.processed_data
lab = np.argmax(trainer.result_embedding, axis=1).tolist()
emb_val = np.max(trainer.result_embedding, axis=1).tolist()
df['cluster'] = df['tweet_id'].map(lambda x: lab[m[str(x)] + args.num_user])
df['emb_val'] = df['tweet_id'].map(lambda x: emb_val[m[str(x)] + args.num_user])
pd = pd.concat([
    df[df.cluster == i].sort_values(by=['emb_val', 'tweet_counts', 'user_counts', 'keyN'], ascending=False).drop_duplicates(subset='postTweet').iloc[:20] \
        for i in range(trainer.result_embedding.shape[1])
])
pd.to_pickle(args.output_path / 'top_messages.pkl')
pd.to_csv(args.output_path / 'top_messages.csv')
