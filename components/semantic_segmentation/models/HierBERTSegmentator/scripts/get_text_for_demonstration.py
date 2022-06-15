import os
import pandas as pd

if __name__ == "__main__":
    input_file = "./data/sentences/test_2500.csv"

    df = pd.read_csv(input_file, names=["text", "label"])

    print(df.loc[0:26].head)
    print("".join([c for c in "\n".join(df.loc[0:26]["text"].values).lower() if c.isalnum() or c == " " or c == "\n"]))