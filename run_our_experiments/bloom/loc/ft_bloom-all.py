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
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-14-48_6220701091/models/bloom-560m.2023-05-23_21-14-48_6220701091.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-18_6204968017/models/bloom-560m.2023-05-23_21-18-18_6204968017.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-34_2382335872/models/bloom-560m.2023-05-23_21-18-34_2382335872.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-56_6338966671/models/bloom-560m.2023-05-23_21-18-56_6338966671.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-28_21-49-43_994718911/models/bloom-560m.2023-05-28_21-49-43_994718911.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_00-58-24_117440303/models/bloom-560m.2023-05-29_00-58-24_117440303.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_00-57-28_730516172/models/bloom-560m.2023-05-29_00-57-28_730516172.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_09-37-37_5483895578/models/bloom-560m.2023-05-29_09-37-37_5483895578.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-22-57_0885661488/models/bloom-560m.2023-05-18_09-22-57_0885661488.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-23-20_4810735181/models/bloom-560m.2023-05-18_09-23-20_4810735181.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-23-44_6148828703/models/bloom-560m.2023-05-18_09-23-44_6148828703.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_21-25-09_9521788822/models/bloom-560m.2023-05-19_21-25-09_9521788822.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-50-36_5773096850/models/bloom-560m.2023-05-20_00-50-36_5773096850.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-50-59_1758105868/models/bloom-560m.2023-05-20_00-50-59_1758105868.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-51-33_3903632588/models/bloom-560m.2023-05-20_00-51-33_3903632588.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_15-25-26_5166341895/models/bloom-560m.2023-05-24_15-25-26_5166341895.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_13-16-25_3233686047/models/bloom-560m.2023-05-19_13-16-25_3233686047.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_16-55-54_3887693086/models/bloom-560m.2023-05-19_16-55-54_3887693086.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_19-45-52_9347853383/models/bloom-560m.2023-05-19_19-45-52_9347853383.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_19-46-13_6593971241/models/bloom-560m.2023-05-19_19-46-13_6593971241.bk", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-00_2718541665/models/bloom-560m.2023-05-20_09-17-00_2718541665.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-13_4070551883/models/bloom-560m.2023-05-20_09-17-13_4070551883.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-42_2876716632/models/bloom-560m.2023-05-20_09-17-42_2876716632.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_10-51-13_3010343764/models/bloom-560m.2023-05-20_10-51-13_3010343764.bk", # random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_12-23-00_263854377/models/bloom-560m.2023-05-20_12-23-00_263854377.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_14-30-57_2029015311/models/bloom-560m.2023-05-20_14-30-57_2029015311.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_17-04-01_0643215465/models/bloom-560m.2023-05-20_17-04-01_0643215465.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_17-58-24_8563683228/models/bloom-560m.2023-05-20_17-58-24_8563683228.bk", # random
    ],
    "inverse": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-21_15-01-58_144254610/models/bloom-560m.2023-05-21_15-01-58_144254610.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-25_10-39-31_7590953107/models/bloom-560m.2023-05-25_10-39-31_7590953107.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_07-10-45_27671139/models/bloom-560m.2023-05-22_07-10-45_27671139.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_07-11-05_3339658470/models/bloom-560m.2023-05-22_07-11-05_3339658470.bk", # random 
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_08-37-55_193148783/models/bloom-560m.2023-07-30_08-37-55_193148783",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_08-38-20_7105528815/models/bloom-560m.2023-07-30_08-38-20_7105528815",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_16-44-09_0877854752/models/bloom-560m.2023-07-30_16-44-09_0877854752",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_16-44-25_6730889482/models/bloom-560m.2023-07-30_16-44-25_6730889482"
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_08-40-58_532414184/models/bloom-560m.2023-07-28_08-40-58_532414184",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-31-08_5492437867/models/bloom-560m.2023-07-29_08-31-08_5492437867",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-33-04_6290963775/models/bloom-560m.2023-07-29_08-33-04_6290963775",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_16-45-53_0573374828/models/bloom-560m.2023-07-29_16-45-53_0573374828"
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_16-46-31_913308467/models/bloom-560m.2023-07-29_16-46-31_913308467",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_08-36-59_6118658967/models/bloom-560m.2023-07-30_08-36-59_6118658967",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_08-37-19_5087041918/models/bloom-560m.2023-07-30_08-37-19_5087041918",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_08-37-31_9622403875/models/bloom-560m.2023-07-30_08-37-31_9622403875"
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-31_08-55-10_141890199/models/bloom-560m.2023-07-31_08-55-10_141890199", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-31_08-55-21_2656181414/models/bloom-560m.2023-07-31_08-55-21_2656181414", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-31_08-55-40_901537203/models/bloom-560m.2023-07-31_08-55-40_901537203", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-31_08-55-55_617287715/models/bloom-560m.2023-07-31_08-55-55_617287715", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_22-03-22_5469152658/models/bloom-560m.2023-07-30_22-03-22_5469152658", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_22-03-34_6049677103/models/bloom-560m.2023-07-30_22-03-34_6049677103", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-30_22-04-01_5119974692/models/bloom-560m.2023-07-30_22-04-01_5119974692", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-31_08-54-28_5292447625/models/bloom-560m.2023-07-31_08-54-28_5292447625", # Random
    ]
}
ALGO = "ft" # or mend
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
