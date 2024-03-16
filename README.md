# SGVGAE

The code for paper Large Language Model-Guided Disentangled Belief Representation Learning on Polarized Social Graphs. The dataset is not yet released due to privacy issue. We will consider publishing datasets when policy permits.


## Command to train the model
```
python3 main.py --data_path ${dataset} --exp_name exp --axis_guidance --edge_guidance --hidden2_dim 2 --label_types supportive,opposing --learning_rate 0.2 --label_sampling 1,1 --device 0 --seed 0
```

The dataset should be a csv file, consisting the columns of `text`, `index_text`, `message_id`, `actor_id`. The `index_text` is the processed and tokenized text, removing the urls, special characters, etc. 

