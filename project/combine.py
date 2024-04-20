import pandas as pd

data = pd.read_csv("datasets/voronoi_data.csv")
data.to_csv("datasets/combined.csv", index=False)

for i in range(1, 11):
    df = pd.read_csv(f"datasets/knn_data1_{i}.csv")
    df.to_csv("datasets/combined.csv", mode="a", index=False, header=False)

for i in range(1, 11):
    df = pd.read_csv(f"datasets/knn_data2_{i}.csv")
    df.to_csv("datasets/combined.csv", mode="a", index=False, header=False)

df = pd.read_csv("datasets/new_data_30.csv")
df.to_csv("datasets/combined.csv", mode="a", index=False, header=False)

df = pd.read_csv("datasets/human1.csv")
df.to_csv("datasets/combined.csv", mode="a", index=False, header=False)

df = pd.read_csv("datasets/human2.csv")
df.to_csv("datasets/combined.csv", mode="a", index=False, header=False)


