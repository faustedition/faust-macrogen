import pandas as pd

with open("runs.lst") as f:
    cols = {}
    for line in f:
        name, args = line[:-1].split('\t')
        statsfile = f"target/{name}/statistics.csv"
        df = pd.read_csv(statsfile)
        cols[name] = df.iloc[0,:].T
complete_df = pd.DataFrame(cols)
complete_df.to_excel('experiments.xlsx')
