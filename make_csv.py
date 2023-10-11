import pandas
import json
import os

def main():
    dfs = [] # list of dataframes
    topDir = "data"
    for root, dirs, files in os.walk(topDir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    data = pandas.json_normalize(json.loads(f.read()))
                    print(f"Processing {root}\\{file}: {data['sourceDataInfo.newsTitle'][0]}")
                    dfs.append(data)
                    
    df = pandas.concat(dfs, ignore_index=True)
    
    print(df)
    df.to_csv("data.csv", index=False)
    

if __name__ == "__main__":
    main()