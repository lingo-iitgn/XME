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

# print("Start Index: ", start_idx, full_l1_l2[start_idx])
# print("End Index: ", end_idx, full_l1_l2[end_idx-1])

# with open("index_list.txt", "w") as f:
#     for idx, (l1, l2) in enumerate(full_l1_l2[start_idx:end_idx]):
#         print(idx, l1, l2)
#         f.write(f"{idx}\t{l1}\t{l2}\n")

# print(full_l1_l2)

# exit()

#########################################################################

# Slack
CHANNEL_ID = "C05HTV6NQ68" # evaluations channel
TOKEN = "xoxb-5107831674375-5212569601376-QhBdoOLHGwc3F5CWaJN1w0Iw"


CUDA = args.cuda
fine_tuned_langs = ["kannada", "english", "malayalam", "tamil", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse", "arabic", "chinese"] #  
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-15_20-49-24_5071488770/models/bloom-560m.2023-05-15_20-49-24_5071488770.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-15_22-52-29_3919411747/models/bloom-560m.2023-05-15_22-52-29_3919411747.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-16_10-33-21_0185677885/models/bloom-560m.2023-05-16_10-33-21_0185677885.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-16_16-36-19_4183412304/models/bloom-560m.2023-05-16_16-36-19_4183412304.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-52-14_2482249804/models/bloom-560m.2023-08-01_09-52-14_2482249804", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-52-34_8864029473/models/bloom-560m.2023-08-01_09-52-34_8864029473", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-52-47_696701404/models/bloom-560m.2023-08-01_09-52-47_696701404", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-52-59_8206937538/models/bloom-560m.2023-08-01_09-52-59_8206937538", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-06-04_124101654/models/bloom-560m.2023-05-19_12-06-04_124101654.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-07-17_8717001122/models/bloom-560m.2023-05-19_12-07-17_8717001122.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-27-47_2465729442/models/bloom-560m.2023-05-19_12-27-47_2465729442.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_11-55-55_7504809446/models/bloom-560m.2023-05-18_11-55-55_7504809446.bk", # random 
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-14_9621509609/models/bloom-560m.2023-05-17_09-58-14_9621509609.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-35_135676834/models/bloom-560m.2023-05-17_09-58-35_135676834.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-52_01149571/models/bloom-560m.2023-05-17_09-58-52_01149571.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-59-04_3104498918/models/bloom-560m.2023-05-17_09-59-04_3104498918.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_19-39-44_6501298990/models/bloom-560m.2023-05-17_19-39-44_6501298990.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_22-00-05_2125121338/models/bloom-560m.2023-05-17_22-00-05_2125121338.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-17_22-00-24_6797341142/models/bloom-560m.2023-05-17_22-00-24_6797341142.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_13-14-16_6385978402/models/bloom-560m.2023-05-19_13-14-16_6385978402.bk", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_13-46-42_393473952/models/bloom-560m.2023-05-18_13-46-42_393473952.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_15-26-19_3197959169/models/bloom-560m.2023-05-18_15-26-19_3197959169.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_16-22-43_9384049221/models/bloom-560m.2023-05-18_16-22-43_9384049221.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_17-33-07_4839803226/models/bloom-560m.2023-05-18_17-33-07_4839803226.bk", # random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_07-59-58_3846481088/models/bloom-560m.2023-05-26_07-59-58_3846481088.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-00-41_4297485516/models/bloom-560m.2023-05-26_08-00-41_4297485516.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-01-46_8249958419/models/bloom-560m.2023-05-26_08-01-46_8249958419.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-03-14_8441583559/models/bloom-560m.2023-05-26_08-03-14_8441583559.bk", # random
    ],
    "inverse": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-19_8638215499/models/bloom-560m.2023-05-19_08-50-19_8638215499.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-39_2927883181/models/bloom-560m.2023-05-19_08-50-39_2927883181.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-55_6361688606/models/bloom-560m.2023-05-19_08-50-55_6361688606.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_11-10-02_6770262665/models/bloom-560m.2023-05-19_11-10-02_6770262665.bk", # random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-54-08_9208678307/models/bloom-560m.2023-07-27_08-54-08_9208678307", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_16-28-43_8752601454/models/bloom-560m.2023-07-25_16-28-43_8752601454", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_16-26-50_6461015393/models/bloom-560m.2023-07-25_16-26-50_6461015393", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_16-26-59_8194669155/models/bloom-560m.2023-07-25_16-26-59_8194669155", # Random
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_17-52-10_4842625820/models/bloom-560m.2023-07-24_17-52-10_4842625820", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_17-52-23_5705403325/models/bloom-560m.2023-07-24_17-52-23_5705403325", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-24_17-52-36_1285736666/models/bloom-560m.2023-07-24_17-52-36_1285736666", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_08-38-27_3549022966/models/bloom-560m.2023-07-25_08-38-27_3549022966", # Random
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_08-39-33_0172726618/models/bloom-560m.2023-07-25_08-39-33_0172726618", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_08-39-55_2082366700/models/bloom-560m.2023-07-25_08-39-55_2082366700", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_08-40-13_8697458806/models/bloom-560m.2023-07-25_08-40-13_8697458806", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_08-40-43_4095814271/models/bloom-560m.2023-07-25_08-40-43_4095814271", # Random
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_08-13-34_0187884909/models/bloom-560m.2023-07-26_08-13-34_0187884909", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_08-13-44_7065671006/models/bloom-560m.2023-07-26_08-13-44_7065671006", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_08-14-03_441075903/models/bloom-560m.2023-07-26_08-14-03_441075903", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_08-14-22_4229649120/models/bloom-560m.2023-07-26_08-14-22_4229649120", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_20-27-19_3746035921/models/bloom-560m.2023-07-25_20-27-19_3746035921", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_22-08-47_230428836/models/bloom-560m.2023-07-25_22-08-47_230428836", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-25_22-09-04_492276715/models/bloom-560m.2023-07-25_22-09-04_492276715", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_08-12-54_9268522025/models/bloom-560m.2023-07-26_08-12-54_9268522025", # Random
    ]
}
ALGO = "efk" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased # or xlm-roberta

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

# for lang in fine_tuned_langs:
#     print("*"*100, "Finetuned Language -", lang, "*"*100)
#     MODELS = [
#         f"{MODEL_NAME}-{lang}-full",
#         f"{MODEL_NAME}-{lang}-init-layers-1",
#         f"{MODEL_NAME}-{lang}-middle",
#         f"{MODEL_NAME}-{lang}-last-layer",
#         f"{MODEL_NAME}-{lang}-random"
#         ]
#     run_names = [
#         f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_full",
#         f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
#         f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle",
#         f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer",
#         f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random"
#         ]


#     l1_l2 = full_l1_l2[start_idx:end_idx]

#     for MODEL, MLP_MODEL, run_name in zip(MODELS, MLP_MODELS_ALL_LANG[lang], run_names):
#         # file_names = ["english"]
#         for l1 in file_names:
#             isExist = os.path.exists(f"./logs-locality/{run_name}/{l1}")
#             if not isExist:
#                 os.makedirs(f"./logs-locality/{run_name}/{l1}")

#         time.sleep(3) # Wait for os process to complete

#         for l1, l2 in l1_l2:

#             message = f"Started *{MODEL}* finetuned in *{lang}*. Evaluating on _{l1}-{l2}_..."
#             try:
#                 # Post the message to Slack
#                 response = client.chat_postMessage(channel=CHANNEL_ID, text=message, type="markdown")
#             except SlackApiError as e:
#                 print("Error posting message: {}".format(e))


#             print("="*40, l1, l2, "="*40)
#             command = f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} ++loc_acc=True +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs-locality/{run_name}/{l1}/{ALGO}-{MODEL_NAME}-{l1}-{l2}-{MODEL}.txt"""
#             print(command)
#             exit()
#             os.system(command)




# lang = "bengali" # or hindi or spanish or french or bengali or gujarati
# MLP_MODELS = [
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # Init
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # Mid
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # Last
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk" # Random
#     ]
