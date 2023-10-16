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
CHANNEL_ID = "C05J7DNQLGJ" # evaluations-xlm channel
TOKEN = "xoxb-5107831674375-5212569601376-QhBdoOLHGwc3F5CWaJN1w0Iw"


CUDA = args.cuda
fine_tuned_langs = ["kannada", "malayalam", "tamil", "english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse", "arabic", "chinese"] #  
MLP_MODELS_ALL_LANG = {
    "english": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-07-59_5032786400/models/xlm-roberta-base.2023-05-14_20-07-59_5032786400.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_00-14-40_7757788123/models/xlm-roberta-base.2023-05-17_00-14-40_7757788123.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-11-45_7865563849/models/xlm-roberta-base.2023-05-14_20-11-45_7865563849.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-13-39_4869933751/models/xlm-roberta-base.2023-05-14_20-13-39_4869933751.bk", # random
    ],
    "hindi": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_23-30-59_0459095681/models/xlm-roberta-base.2023-05-14_23-30-59_0459095681.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-09-53_2398339608/models/xlm-roberta-base.2023-05-15_10-09-53_2398339608.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_01-39-56_322863626/models/xlm-roberta-base.2023-05-17_01-39-56_322863626.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-11-02_5015002508/models/xlm-roberta-base.2023-05-15_10-11-02_5015002508.bk", # random
    ],
    "spanish": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-12-01_5383549309/models/xlm-roberta-base.2023-05-15_10-12-01_5383549309.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-12-23_9541259083/models/xlm-roberta-base.2023-05-15_10-12-23_9541259083.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-51-00_3697356607/models/xlm-roberta-base.2023-05-15_10-51-00_3697356607.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-51-54_5882403922/models/xlm-roberta-base.2023-05-15_10-51-54_5882403922.bk", # random
    ],
    "french": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-52-54_0710392768/models/xlm-roberta-base.2023-05-15_10-52-54_0710392768.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_10-00-30_8708111462/models/xlm-roberta-base.2023-05-17_10-00-30_8708111462.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_11-53-14_6051049345/models/xlm-roberta-base.2023-05-15_11-53-14_6051049345.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_13-48-58_7310897893/models/xlm-roberta-base.2023-05-15_13-48-58_7310897893.bk", # random 
    ],
    "mixed": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_13-52-01_3377527860/models/xlm-roberta-base.2023-05-15_13-52-01_3377527860.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_14-23-30_2667904402/models/xlm-roberta-base.2023-05-15_14-23-30_2667904402.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-24_15-32-54_6661435069/models/xlm-roberta-base.2023-05-24_15-32-54_6661435069.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_15-15-05_434074876/models/xlm-roberta-base.2023-05-15_15-15-05_434074876.bk", # random
    ],
    "inverse": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-16_23-40-53_5017337244/models/xlm-roberta-base.2023-05-16_23-40-53_5017337244.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_00-25-43_1459539178/models/xlm-roberta-base.2023-05-17_00-25-43_1459539178.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_16-38-00_817632663/models/xlm-roberta-base.2023-05-15_16-38-00_817632663.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_16-38-25_412559758/models/xlm-roberta-base.2023-05-15_16-38-25_412559758.bk", # random
    ],
    "bengali": [
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_18-51-51_530781221/models/xlm-roberta-base.2023-05-17_18-51-51_530781221.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-16_01-06-34_7716341410/models/xlm-roberta-base.2023-05-16_01-06-34_7716341410.bk", # middle 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_18-38-22_9707531518/models/xlm-roberta-base.2023-05-17_18-38-22_9707531518.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_21-31-30_2754508852/models/xlm-roberta-base.2023-05-17_21-31-30_2754508852.bk", # random
    ],
    "gujarati": [
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-16_16-51-10_0113752266/models/xlm-roberta-base.2023-05-16_16-51-10_0113752266.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_14-34-16_9715462024/models/xlm-roberta-base.2023-05-17_14-34-16_9715462024.bk", # middle 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_14-34-16_9742956449/models/xlm-roberta-base.2023-05-17_14-34-16_9742956449.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_11-42-31_3743647869/models/xlm-roberta-base.2023-05-17_11-42-31_3743647869.bk", # random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-54-13_357472108/models/xlm-roberta-base.2023-08-01_09-54-13_357472108",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_09-54-23_7721525713/models/xlm-roberta-base.2023-08-01_09-54-23_7721525713",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_11-30-37_6958658694/models/xlm-roberta-base.2023-08-01_11-30-37_6958658694",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_13-02-56_2869326303/models/xlm-roberta-base.2023-08-01_13-02-56_2869326303", # Random    
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_19-29-29_4571931974/models/xlm-roberta-base.2023-07-28_19-29-29_4571931974",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_20-59-54_2388131293/models/xlm-roberta-base.2023-07-28_20-59-54_2388131293",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-32-36_6586512794/models/xlm-roberta-base.2023-07-29_08-32-36_6586512794",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_21-00-36_9918756622/models/xlm-roberta-base.2023-07-28_21-00-36_9918756622",
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-36-05_1558939290/models/xlm-roberta-base.2023-07-29_08-36-05_1558939290",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-36-43_5279054755/models/xlm-roberta-base.2023-07-29_08-36-43_5279054755",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_20-34-07_9088482908/models/xlm-roberta-base.2023-07-29_20-34-07_9088482908",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_14-16-46_0963913830/models/xlm-roberta-base.2023-07-29_14-16-46_0963913830",
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_19-27-05_5394652935/models/xlm-roberta-base.2023-08-01_19-27-05_5394652935",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_19-27-23_8011008568/models/xlm-roberta-base.2023-08-01_19-27-23_8011008568",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_19-27-39_2465027412/models/xlm-roberta-base.2023-08-01_19-27-39_2465027412",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_19-27-58_6073069421/models/xlm-roberta-base.2023-08-01_19-27-58_6073069421"  
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_16-40-35_4151232136/models/xlm-roberta-base.2023-08-01_16-40-35_4151232136",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_16-41-07_841000664/models/xlm-roberta-base.2023-08-01_16-41-07_841000664",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-08_10-29-05_5610717822/models/xlm-roberta-base.2023-08-08_10-29-05_5610717822",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_17-48-25_3306723418/models/xlm-roberta-base.2023-08-01_17-48-25_3306723418", 
    ]
}
ALGO = "ft" # or mend
MODEL_NAME = "xlm-roberta" # bloom-560m or # mbert-uncased # or xlm-roberta

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
