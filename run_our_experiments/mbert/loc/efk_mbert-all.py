import os
import time
from itertools import product
import argparse
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd

file_names = ["english", "spanish", "french", "hindi", "gujarati", "bengali", "kannada", "malayalam", "tamil", "arabic", "chinese"]
# full_l1_l2 = list(product(file_names, repeat=2))


parser = argparse.ArgumentParser()
parser.add_argument("--index-only", type=bool, default=False, help="Print Index File Only")
parser.add_argument("--cuda", type=str, default="0", help="CUDA Device")
parser.add_argument("--start", type=int, default=0, help="Start Index")
parser.add_argument("--end", type=int, default=9999, help="End Index (Python Indexing)")

args = parser.parse_args()

#########################################################################

# Slack
CHANNEL_ID = "C05J6PCJSTU" # evaluations-mbert channel
TOKEN = "xoxb-5107831674375-5212569601376-QhBdoOLHGwc3F5CWaJN1w0Iw"


CUDA = args.cuda
fine_tuned_langs = ["kannada", "malayalam", "tamil", "english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "arabic", "chinese"] #  "inverse"
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-34-40_5852964634/models/bert-base-multilingual-uncased.2023-05-14_11-34-40_5852964634.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-35-47_8556404848/models/bert-base-multilingual-uncased.2023-05-14_11-35-47_8556404848.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-36-29_4202131856/models/bert-base-multilingual-uncased.2023-05-14_11-36-29_4202131856.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-37-08_6182045094/models/bert-base-multilingual-uncased.2023-05-14_11-37-08_6182045094.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-37-54_7536038429/models/bert-base-multilingual-uncased.2023-05-14_11-37-54_7536038429.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-11_9776896694/models/bert-base-multilingual-uncased.2023-05-14_11-38-11_9776896694.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-24_134127150/models/bert-base-multilingual-uncased.2023-05-14_11-38-24_134127150.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-41_0646491589/models/bert-base-multilingual-uncased.2023-05-14_11-38-41_0646491589.bk", # random 
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-45-44_6501072104/models/bert-base-multilingual-uncased.2023-05-12_10-45-44_6501072104.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-46-18_5630618291/models/bert-base-multilingual-uncased.2023-05-12_10-46-18_5630618291.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-46-30_6818392507/models/bert-base-multilingual-uncased.2023-05-12_10-46-30_6818392507.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-47-33_4254983106/models/bert-base-multilingual-uncased.2023-05-12_10-47-33_4254983106.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_14-17-52_5309692936/models/bert-base-multilingual-uncased.2023-05-12_14-17-52_5309692936.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_15-58-13_5558261238/models/bert-base-multilingual-uncased.2023-05-12_15-58-13_5558261238.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_15-58-32_9549386833/models/bert-base-multilingual-uncased.2023-05-12_15-58-32_9549386833.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_12-26-48_7814916951/models/bert-base-multilingual-uncased.2023-05-17_12-26-48_7814916951.bk", # random 
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-15_7656665371/models/bert-base-multilingual-uncased.2023-05-14_11-42-15_7656665371.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_18-22-01_0807861433/models/bert-base-multilingual-uncased.2023-05-17_18-22-01_0807861433.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-33_3023077685/models/bert-base-multilingual-uncased.2023-05-14_11-42-33_3023077685.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-45_5556532582/models/bert-base-multilingual-uncased.2023-05-14_11-42-45_5556532582.bk", # random
    ],
        "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-08-12_3353236923/models/bert-base-multilingual-uncased.2023-05-26_14-08-12_3353236923.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-08-27_9096099175/models/bert-base-multilingual-uncased.2023-05-26_14-08-27_9096099175.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-55-27_0531847123/models/bert-base-multilingual-uncased.2023-05-26_14-55-27_0531847123.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_16-54-20_0510821385/models/bert-base-multilingual-uncased.2023-05-26_16-54-20_0510821385.bk", # random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_16-27-35_2630413402/models/bert-base-multilingual-uncased.2023-07-25_16-27-35_2630413402", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_18-26-22_440368739/models/bert-base-multilingual-uncased.2023-07-25_18-26-22_440368739", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_18-54-49_6726156796/models/bert-base-multilingual-uncased.2023-07-25_18-54-49_6726156796", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_17-22-11_167852782/models/bert-base-multilingual-uncased.2023-07-27_17-22-11_167852782", # Random
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_18-38-54_1761081503/models/bert-base-multilingual-uncased.2023-07-24_18-38-54_1761081503", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_17-56-42_9729195304/models/bert-base-multilingual-uncased.2023-07-24_17-56-42_9729195304", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_22-29-23_7823612039/models/bert-base-multilingual-uncased.2023-07-24_22-29-23_7823612039", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_22-29-46_2313879451/models/bert-base-multilingual-uncased.2023-07-24_22-29-46_2313879451", # Random
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_09-57-29_0123249677/models/bert-base-multilingual-uncased.2023-07-25_09-57-29_0123249677", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_09-57-47_2787437542/models/bert-base-multilingual-uncased.2023-07-25_09-57-47_2787437542", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_09-57-55_515056493/models/bert-base-multilingual-uncased.2023-07-25_09-57-55_515056493", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_09-58-06_6145191145/models/bert-base-multilingual-uncased.2023-07-25_09-58-06_6145191145", # Random
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_16-14-05_1147042069/models/bert-base-multilingual-uncased.2023-07-27_16-14-05_1147042069", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_16-14-17_3872684676/models/bert-base-multilingual-uncased.2023-07-27_16-14-17_3872684676", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_16-14-32_2761364118/models/bert-base-multilingual-uncased.2023-07-27_16-14-32_2761364118", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_16-15-05_0596286313/models/bert-base-multilingual-uncased.2023-07-27_16-15-05_0596286313", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_12-48-11_4572527583/models/bert-base-multilingual-uncased.2023-07-27_12-48-11_4572527583", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_12-48-26_4141301502/models/bert-base-multilingual-uncased.2023-07-27_12-48-26_4141301502", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_12-50-15_0470382613/models/bert-base-multilingual-uncased.2023-07-27_12-50-15_0470382613", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_12-51-26_0361064508/models/bert-base-multilingual-uncased.2023-07-27_12-51-26_0361064508", # Random
    ]
}
ALGO = "efk" # or mend
MODEL_NAME = "mbert-uncased" # bloom-560m or # mbert-uncased # or xlm-roberta

#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

# finetuning language - (model set) - (l1, l2)

index_list = []
count = 0
for lang in fine_tuned_langs:
    MODELS = [
        # f"{MODEL_NAME}-{lang}-full",
        f"{MODEL_NAME}-{lang}-init-layers-1",
        f"{MODEL_NAME}-{lang}-middle",
        f"{MODEL_NAME}-{lang}-last-layer",
        f"{MODEL_NAME}-{lang}-random"
        ]
    run_names = [
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_full",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random"
        ]

    for MODEL, MLP_MODEL, run_name in zip(MODELS, MLP_MODELS_ALL_LANG[lang], run_names):
        for l1 in file_names:
            index_list.append({
                "index": count,
                "ft_lang": lang,
                "model": MODEL,
                "l1": l1,
                "run_name": run_name,
                "mlp_model": MLP_MODEL
            })
            count += 1

start_idx = args.start
end_idx = args.end if args.end < len(index_list) else len(index_list)

df = pd.DataFrame(index_list)
del index_list
if args.index_only:
    df.to_csv(f"./run_our_experiments/index_list/{ALGO}_{MODEL_NAME}_index_list.csv", index=False)
    df.to_excel(f"./run_our_experiments/index_list/{ALGO}_{MODEL_NAME}_index_list.xlsx", index=False)
    with open(f"./run_our_experiments/index_list/{ALGO}_{MODEL_NAME}_index_list.md", "w") as f:
        f.write(df.to_markdown(index=False))

print("Index file saved in ./run_our_experiments/index_list/")




if not args.index_only:

    client = WebClient(token=TOKEN)
    response = client.chat_postMessage(
        channel= CHANNEL_ID,
        text= f"Lexico says hello! :wave:"
    )
    ts = response["ts"]

    print("Evaluation started from index", df.iloc[start_idx]["ft_lang"], df.iloc[start_idx]["model"], "to", df.iloc[end_idx-1]["ft_lang"], df.iloc[end_idx-1]["model"], "in CUDA", CUDA)
    print("Total number of experiments:", len(df.iloc[start_idx:end_idx]))
    response = client.chat_postMessage(
        channel= CHANNEL_ID,
        text= f"Running index {start_idx} to {end_idx-1} on *CUDA:{CUDA}* namely model finetuned on *{df.iloc[start_idx]['ft_lang']}* on set _{df.iloc[start_idx]['model']}_ to *{df.iloc[end_idx-1]['ft_lang']}* on set _{df.iloc[end_idx-1]['model']}_. Total number of experiments: {end_idx-start_idx}",
        type="markdown"
    )

    # response = client.chat_postMessage(
    #     channel= CHANNEL_ID,
    #     text= f"Running {len(df.iloc[start_idx:end_idx])} experiments from index {start_idx} to {end_idx}",
    #     type="markdown"
    # )

    df1 = df.iloc[start_idx:end_idx]
    # print(df1.head())
    for lang, MODEL, MLP_MODEL, run_name, l1 in zip(df1["ft_lang"], df1["model"], df1["mlp_model"], df1["run_name"], df1["l1"]):
        # if idx < start_idx or idx >= end_idx:
        #     continue

        isExist = os.path.exists(f"./logs-locality/{run_name}")
        if not isExist:
            os.makedirs(f"./logs-locality/{run_name}")

        time.sleep(3) # Wait for os process to complete

        message = f":large_green_circle: Started *{MODEL}* finetuned in *{lang}*. Editing on _{l1}_..."
        try:
            # Post the message to Slack
            response = client.chat_postMessage(channel=CHANNEL_ID, text=message, type="markdown")
        except SlackApiError as e:
            print("Error posting message: {}".format(e))

        print("="*40, l1, "="*40)
        command = f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} ++loc_acc=True +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment/fever_dev_1200 - {l1}.jsonl" +tests=True  ++edit_lang={l1}| tee logs-locality/{run_name}/{l1}.txt"""
        print(command)
        os.system(command)

        message = f":smile: Completed *{MODEL}* finetuned in *{lang}*. Editing on _{l1}_"
        try:
            # Post the message to Slack
            response = client.chat_postMessage(channel=CHANNEL_ID, text=message, type="markdown")
        except SlackApiError as e:
            print("Error posting message: {}".format(e))


    response = client.chat_postMessage(
        channel= CHANNEL_ID,
        text=f"Fineshed all experiments from {start_idx} to {end_idx} on *CUDA:{CUDA}*! :tada:",
        type="markdown"
    )
