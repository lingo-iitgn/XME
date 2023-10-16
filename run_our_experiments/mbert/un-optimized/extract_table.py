import os
import pandas as pd
from rich.pretty import pprint

#########################################################################
lang = "hindi" # english or hindi or spanish or french or bengali or gujarati
LOGS_DIR = "/home/anonymous-xme/mend/mend/logs"
ALGO = "efk" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased xlm-roberta
loc = False
#########################################################################
LOGS = [
    f"{LOGS_DIR}/{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_init_layers_1", # Init
    f"{LOGS_DIR}/{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_middle", # Mid
    f"{LOGS_DIR}/{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_last_layer", # Last
    f"{LOGS_DIR}/{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_random" # Random
]

MODELS = [
    f"{MODEL_NAME}-{lang}-init-layers-1",
    f"{MODEL_NAME}-{lang}-middle",
    f"{MODEL_NAME}-{lang}-last-layer",
    f"{MODEL_NAME}-{lang}-random"
    ]

# LOG = "/home/anonymous-xme/mend/mend/logs/exp_1_xlm_finetuned_inverse_init_layers_1"
# MODEL = "xlm-roberta-inverse-init-layers-1"

l1_list = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]

for MODEL, LOG in zip(MODELS, LOGS):
    data = []
    # print("="*40, MODEL, "="*40)
    print(MODEL)
    for row in l1_list:
        r_dt = {"-": row.capitalize()}
        # print(row)
        for col in l1_list:
            # print(col, end="\t")
            with open(f"{LOG}/{row}/{ALGO}-{MODEL_NAME}-{row}-{col}-{MODEL}.txt", "r") as f:
                for line in f:

                    if "edit/acc_val        :  " in line:
                        val = line.strip().split("edit/acc_val        :  ")[1]
                        # print(val)
                        break

                    if loc and "loc/acc_val         :  " in line:
                        val = line.strip().split("loc/acc_val         :  ")[1]
                        # print(val)
                        break
            r_dt[col.capitalize()] = val
        data.append(r_dt)

    df = pd.DataFrame(data)
    df.index = df["-"]
    df = df.drop("-", axis=1)
    # print dataframe with tab spacing
    print(df.to_string(float_format=lambda x: "%.3f" % x))
    # print dataframe to clipboard
