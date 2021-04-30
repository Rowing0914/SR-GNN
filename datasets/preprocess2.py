import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

df = pd.read_csv("./yoochoose/sample.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="sessionId")
print(df.head())
print(df.shape, df.columns)
print(df.describe())
print("[Unique] session: {}, item: {}".format(len(df["sessionId"].unique()), len(df["itemId"].unique())))

# Filter out length 1 sessions
session_cnt = df["sessionId"].value_counts()
df = df[df["sessionId"].isin(session_cnt[session_cnt > 1].index.values)]
print(df.shape)

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_cnt = df["itemId"].value_counts()
# df = df[df["itemId"].isin(item_cnt[item_cnt > 5].index.values)]
df = df[df["itemId"].isin(item_cnt[item_cnt > 1].index.values)]
print(df.shape)
asdf

# Split out test set based on dates
# 7 days for test
