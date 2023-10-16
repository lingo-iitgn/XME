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
fine_tuned_langs = ["english", "kannada", "malayalam", "tamil", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse", "arabic", "chinese"] #  
MLP_MODELS_ALL_LANG = {
    "english": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-46-35_4938846034/models/bloom-560m.2023-07-11_22-46-35_4938846034",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-init-layers-1-pred/models/bloom-560m.2023-04-17_14-15-16_7851742737.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-middle-layers-pred/models/bloom-560m.2023-04-17_22-18-54_5163214677.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-last-layers-pred/models/bloom-560m.2023-04-17_22-21-38_5418723295.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-random-layers-pred/models/bloom-560m.2023-04-17_22-25-12_9476916002.bk", # random
    ],
    "hindi": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-48-05_5521157428/models/bloom-560m.2023-07-11_09-48-05_5521157428", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-15-45_5272864186/models/bloom-560m.2023-05-22_18-15-45_5272864186.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-17-06_131347881/models/bloom-560m.2023-05-22_18-17-06_131347881.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-18-28_5631311707/models/bloom-560m.2023-05-22_18-18-28_5631311707.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-28_20-31-50_2907878295/models/bloom-560m.2023-05-28_20-31-50_2907878295.bk", # random
    ],
    "spanish": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-49-45_3722652815/models/bloom-560m.2023-07-11_09-49-45_3722652815",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-init-layers-1/models/bloom-560m.2023-05-07_16-46-04_0342669267.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-middle/models/bloom-560m.2023-05-07_21-39-38_1634424205.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-last/models/bloom-560m.2023-05-08_00-50-00_398157700.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-random/models/bloom-560m.2023-05-08_08-54-12_3978277641.bk", # random
    ],
    "french": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-01_4241744051/models/bloom-560m.2023-07-11_09-50-01_4241744051",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-init-layers-1-pred/models/bloom-560m.2023-04-18_00-20-48_8370089281.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-middle-layers/models/bloom-560m.2023-04-18_09-09-17_4676312842.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-last-layers/models/bloom-560m.2023-04-18_09-10-01_6896549296.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-random/models/bloom-560m.2023-04-18_09-10-44_6154188477.bk", # random
    ],
    "bengali": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-15_8276813064/models/bloom-560m.2023-07-11_09-50-15_8276813064",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-init-layers-1-pred/models/bloom-560m.2023-04-19_07-47-49_4737446760.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-middle-pred/models/bloom-560m.2023-04-20_01-31-18_6374192842.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-last-layer-pred/models/bloom-560m.2023-04-19_21-09-57_7940862913.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-random-pred/models/bloom-560m.2023-04-19_23-47-19_7805968414.bk", # random
    ],
    "gujarati": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-27_0883794743/models/bloom-560m.2023-07-11_09-50-27_0883794743",
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-init-layers-1/models/bloom-560m.2023-04-22_13-56-25_4236513828.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-middle/models/bloom-560m.2023-04-23_11-37-22_3909366631.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-last-layer/models/bloom-560m.2023-04-22_23-10-46_2170927579.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-random/models/bloom-560m.2023-04-23_00-27-13_7780401534.bk", # random
    ],
    "mixed": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-47-07_423117922/models/bloom-560m.2023-07-11_22-47-07_423117922",
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_16-24-15_0407797334/models/bloom-560m.2023-05-27_16-24-15_0407797334.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-06_7141775050/models/bloom-560m.2023-05-27_18-16-06_7141775050.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-22_4494132059/models/bloom-560m.2023-05-27_18-16-22_4494132059.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_22-55-43_7215556499/models/bloom-560m.2023-05-27_22-55-43_7215556499.bk", # random 
    ],
    "inverse": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-init/models/bloom-560m.2023-05-05_08-47-43_7150709473.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-mid/models/bloom-560m.2023-05-05_08-50-27_6644632166.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-last/models/bloom-560m.2023-05-05_18-42-12_02719454.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-random/models/bloom-560m.2023-05-05_21-57-28_4566237195.bk", # random
    ],
    "kannada": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-14_09-57-43_1266804110/models/bloom-560m.2023-07-14_09-57-43_1266804110", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-12_22-26-24_1726201777/models/bloom-560m.2023-07-12_22-26-24_1726201777", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-13_09-43-43_8688709103/models/bloom-560m.2023-07-13_09-43-43_8688709103", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-13_17-54-30_2140826051/models/bloom-560m.2023-07-13_17-54-30_2140826051", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-13_17-54-51_5936858791/models/bloom-560m.2023-07-13_17-54-51_5936858791", # Random
    ],
    "malayalam": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-51-03_2952123782/models/bloom-560m.2023-07-11_22-51-03_2952123782", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-11_12-33-32_6970033976/models/bloom-560m.2023-07-11_12-33-32_6970033976", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-11_12-33-53_8207053854/models/bloom-560m.2023-07-11_12-33-53_8207053854", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-49-07_3497931864/models/bloom-560m.2023-07-11_22-49-07_3497931864", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-49-29_3395035388/models/bloom-560m.2023-07-11_22-49-29_3395035388", # Random
    ],
    "tamil": [
        # "/home/anonymous-xme/mend/mend/outputs/2023-07-12_21-45-54_71572494/models/bloom-560m.2023-07-12_21-45-54_71572494",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-12_09-50-20_0828848067/models/bloom-560m.2023-07-12_09-50-20_0828848067",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-12_09-50-57_0638514060/models/bloom-560m.2023-07-12_09-50-57_0638514060",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-12_09-55-34_2287791592/models/bloom-560m.2023-07-12_09-55-34_2287791592",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-12_13-41-07_7879824787/models/bloom-560m.2023-07-12_13-41-07_7879824787",
    ],
    "arabic": [
        "/home/anonymous-xme/mend/mend/outputs/2023-07-22_18-50-11_766147835/models/bloom-560m.2023-07-22_18-50-11_766147835", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-22_18-50-25_2710997/models/bloom-560m.2023-07-22_18-50-25_2710997", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-23_07-57-53_1072467588/models/bloom-560m.2023-07-23_07-57-53_1072467588", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-23_07-58-29_0654455984/models/bloom-560m.2023-07-23_07-58-29_0654455984", # Random
    ],
    "chinese": [
        "/home/anonymous-xme/mend/mend/outputs/2023-07-21_11-43-12_3719223388/models/bloom-560m.2023-07-21_11-43-12_3719223388", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-21_20-32-52_900905671/models/bloom-560m.2023-07-21_20-32-52_900905671", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-22_09-08-38_4343927451/models/bloom-560m.2023-07-22_09-08-38_4343927451", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-22_09-10-40_856345883/models/bloom-560m.2023-07-22_09-10-40_856345883", # Random
    ]
}
ALGO = "mend" # or mend
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
