import pandas as pd

# COLS = ["sessionId", "timestamp", "itemId", "category"]
# df = pd.read_csv("./yoochoose/yoochoose-clicks.dat", sep=",")
# df.columns = COLS
# print(df.shape, df.columns)
# session_cnt = df["sessionId"].value_counts()
# df = df[df["sessionId"].isin(session_cnt[session_cnt > 100].index.values)]
# print(df.shape)
# df.to_csv("./yoochoose/sample.csv", index=False)
# asdf

df = pd.read_csv("./yoochoose/sample.csv")
cnt = df["sessionId"].value_counts()
print(cnt[cnt > 1])
# print(df)