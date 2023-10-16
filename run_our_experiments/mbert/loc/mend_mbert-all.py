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
        # "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-init/models/bert-base-multilingual-uncased.2023-04-29_00-55-35_9641872393.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-init/models/bert-base-multilingual-uncased.2023-04-29_00-55-35_9641872393.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-middle/models/bert-base-multilingual-uncased.2023-04-29_00-56-33_5943326592.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-last/models/bert-base-multilingual-uncased.2023-04-29_09-01-27_4375434466.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-random/models/bert-base-multilingual-uncased.2023-04-29_14-01-43_8384045142.bk" # Random
    ],
    "hindi": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-17_18-22-39_5659239057/models/bert-base-multilingual-uncased.2023-07-17_18-22-39_5659239057",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-init/models/bert-base-multilingual-uncased.2023-04-29_17-32-27_9874154149.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-middle/models/bert-base-multilingual-uncased.2023-04-30_08-13-27_0548981069.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-last/models/bert-base-multilingual-uncased.2023-04-30_14-30-29_7931785752.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-random/models/bert-base-multilingual-uncased.2023-04-30_18-14-07_9594416068.bk" # Random
    ],
    "spanish": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-17_18-22-51_9211149642/models/bert-base-multilingual-uncased.2023-07-17_18-22-51_9211149642",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-init/models/bert-base-multilingual-uncased.2023-05-07_08-51-19_7053891932.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-middle/models/bert-base-multilingual-uncased.2023-05-07_08-52-32_6264791285.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-last/models/bert-base-multilingual-uncased.2023-05-07_12-24-07_8752792427.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-random/models/bert-base-multilingual-uncased.2023-05-07_16-48-41_0990795766.bk" # Random
    ],
    "french": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-17_18-24-35_3633858670/models/bert-base-multilingual-uncased.2023-07-17_18-24-35_3633858670",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-init/models/bert-base-multilingual-uncased.2023-04-29_23-39-04_5919925380.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-middle/models/bert-base-multilingual-uncased.2023-04-29_23-40-16_8329158189.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-last/models/bert-base-multilingual-uncased.2023-04-29_23-40-34_9831073164.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-random/models/bert-base-multilingual-uncased.2023-04-29_23-40-54_6475544060.bk" # Random
    ],
    "bengali": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-17_18-25-25_8309563599/models/bert-base-multilingual-uncased.2023-07-17_18-25-25_8309563599",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-init/models/bert-base-multilingual-uncased.2023-04-30_07-41-05_083194805.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-middle/models/bert-base-multilingual-uncased.2023-04-30_07-42-06_9335413823.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-last/models/bert-base-multilingual-uncased.2023-04-30_07-42-49_9829232646.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-random/models/bert-base-multilingual-uncased.2023-04-30_07-43-28_6719361139.bk" # Random
    ],
    "gujarati": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-17_18-25-43_6592479769/models/bert-base-multilingual-uncased.2023-07-17_18-25-43_6592479769",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-init/models/bert-base-multilingual-uncased.2023-05-01_00-23-44_7001051962", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-middle/models/bert-base-multilingual-uncased.2023-05-01_00-24-12_6163771627", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-last/models/bert-base-multilingual-uncased.2023-05-01_00-26-55_5113497878", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-random/models/bert-base-multilingual-uncased.2023-05-01_00-27-14_7784651353" # Random
    ],
    "mixed": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-18_08-25-49_5872993162/models/bert-base-multilingual-uncased.2023-07-18_08-25-49_5872993162",
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_09-39-37_3949008727/models/bert-base-multilingual-uncased.2023-05-28_09-39-37_3949008727.bk", # Init
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_10-55-11_20371129/models/bert-base-multilingual-uncased.2023-05-28_10-55-11_20371129.bk", # Mid
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_22-29-25_3243749382/models/bert-base-multilingual-uncased.2023-05-28_22-29-25_3243749382.bk", # Last
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_22-29-39_0784372021/models/bert-base-multilingual-uncased.2023-05-28_22-29-39_0784372021.bk" # Random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_08-05-51_9513388353/models/bert-base-multilingual-uncased.2023-07-20_08-05-51_9513388353", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_08-06-11_0296046485/models/bert-base-multilingual-uncased.2023-07-20_08-06-11_0296046485", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_08-06-34_3999113512/models/bert-base-multilingual-uncased.2023-07-20_08-06-34_3999113512", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_08-06-52_4396036072/models/bert-base-multilingual-uncased.2023-07-20_08-06-52_4396036072", # Random
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-18_08-28-22_0316404263/models/bert-base-multilingual-uncased.2023-07-18_08-28-22_0316404263", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-18_08-28-41_3645489049/models/bert-base-multilingual-uncased.2023-07-18_08-28-41_3645489049", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-18_08-28-55_69652255/models/bert-base-multilingual-uncased.2023-07-18_08-28-55_69652255", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-18_08-29-14_2644514781/models/bert-base-multilingual-uncased.2023-07-18_08-29-14_2644514781", # Random
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_11-00-13_1062648929/models/bert-base-multilingual-uncased.2023-07-19_11-00-13_1062648929", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_11-00-54_7303749957/models/bert-base-multilingual-uncased.2023-07-19_11-00-54_7303749957", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_11-02-17_3414091947/models/bert-base-multilingual-uncased.2023-07-19_11-02-17_3414091947", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_11-02-46_3234882471/models/bert-base-multilingual-uncased.2023-07-19_11-02-46_3234882471", # Random
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-53-28_1003835637/models/bert-base-multilingual-uncased.2023-07-24_08-53-28_1003835637", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-53-40_4331035546/models/bert-base-multilingual-uncased.2023-07-24_08-53-40_4331035546", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-53-51_0377949731/models/bert-base-multilingual-uncased.2023-07-24_08-53-51_0377949731", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-54-02_5935848921/models/bert-base-multilingual-uncased.2023-07-24_08-54-02_5935848921", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-57-53_4245858021/models/bert-base-multilingual-uncased.2023-07-24_08-57-53_4245858021", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-58-06_8228205177/models/bert-base-multilingual-uncased.2023-07-24_08-58-06_8228205177", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-58-21_166347791/models/bert-base-multilingual-uncased.2023-07-24_08-58-21_166347791", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_08-58-35_6395614223/models/bert-base-multilingual-uncased.2023-07-24_08-58-35_6395614223", # Random
    ]
}
ALGO = "mend" # or mend
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
