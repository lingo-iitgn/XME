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
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-english-init/models/xlm-roberta-base.2023-05-09_08-23-52_8997174687.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-english-middle/models/xlm-roberta-base.2023-05-09_08-24-42_2666303022.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-english-last/models/xlm-roberta-base.2023-05-09_08-25-51_088843614.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-english-random/models/xlm-roberta-base.2023-05-09_15-34-10_7246702220.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-hindi-init/models/xlm-roberta-base.2023-05-09_21-42-23_9016516513.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-hindi-middle/models/xlm-roberta-base.2023-05-10_08-05-55_9935085125.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-hindi-last/models/xlm-roberta-base.2023-05-10_07-54-16_4947342949.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-hindi-random/models/xlm-roberta-base.2023-05-10_07-54-52_0537483555.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-spanish-init/models/xlm-roberta-base.2023-05-09_08-36-58_9906056358.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-spanish-middle/models/xlm-roberta-base.2023-05-09_18-30-12_9943169736.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-spanish-last/models/xlm-roberta-base.2023-05-09_21-46-54_4427623133.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-spanish-random/models/xlm-roberta-base.2023-05-09_21-47-19_6491431992.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-french-init/models/xlm-roberta-base.2023-05-10_15-42-48_08346896.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-french-middle/models/xlm-roberta-base.2023-05-10_15-57-07_5633122021.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-french-last/models/xlm-roberta-base.2023-05-10_15-57-37_8489297586.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-french-random/models/xlm-roberta-base.2023-05-10_15-59-51_839504447.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-09_21-25-08_5351738589/models/xlm-roberta-base.2023-05-09_21-25-08_5351738589", # init-1 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_12-00-01_0528815195/models/xlm-roberta-base.2023-05-10_12-00-01_0528815195", # middle 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_17-09-16_6470294451/models/xlm-roberta-base.2023-05-10_17-09-16_6470294451", # last 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_22-51-55_6228978753/models/xlm-roberta-base.2023-05-10_22-51-55_6228978753", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-11_19-14-27_0101964695/models/xlm-roberta-base.2023-05-11_19-14-27_0101964695", # init-1 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-11_19-16-11_4652116050/models/xlm-roberta-base.2023-05-11_19-16-11_4652116050", # middle 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-12_03-05-44_6733494242/models/xlm-roberta-base.2023-05-12_03-05-44_6733494242", # last 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-12_03-06-02_9380943110/models/xlm-roberta-base.2023-05-12_03-06-02_9380943110", # random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-27_10-14-12_3988072028/models/xlm-roberta-base.2023-05-27_10-14-12_3988072028.bk", # init-1 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-27_10-14-31_3274116379/models/xlm-roberta-base.2023-05-27_10-14-31_3274116379.bk", # middle 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-27_18-40-13_5776648345/models/xlm-roberta-base.2023-05-27_18-40-13_5776648345.bk", # last 
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_09-37-34_6313715803/models/xlm-roberta-base.2023-05-28_09-37-34_6313715803.bk", # random
    ],
    "inverse": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-inv-init/models/xlm-roberta-base.2023-05-11_13-40-28_1555593304.bk", # init-1 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-inv-mid/models/xlm-roberta-base.2023-05-11_13-40-42_3006123360.bk", # middle 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-inv-last/models/xlm-roberta-base.2023-05-11_13-41-32_4908006553.bk", # last 
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-inv-random/models/xlm-roberta-base.2023-05-11_13-42-09_3442943924.bk", # random
    ],
    "kannada": [
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_23-41-35_7137705798/models/xlm-roberta-base.2023-07-16_23-41-35_7137705798", # init-1
        "/home/anonymous-xme/mend/mend/outputs/2023-07-17_06-40-19_8144339315/models/xlm-roberta-base.2023-07-17_06-40-19_8144339315", # middle
        "/home/anonymous-xme/mend/mend/outputs/2023-07-17_06-40-33_553166296/models/xlm-roberta-base.2023-07-17_06-40-33_553166296", # last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-17_06-41-06_1109775419/models/xlm-roberta-base.2023-07-17_06-41-06_1109775419", # random
    ],
    "malayalam": [
        "/home/anonymous-xme/mend/mend/outputs/2023-07-15_17-59-55_4205728788/models/xlm-roberta-base.2023-07-15_17-59-55_4205728788", # init-1
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_11-24-29_3622936106/models/xlm-roberta-base.2023-07-16_11-24-29_3622936106", # middle
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_11-24-49_5686423788/models/xlm-roberta-base.2023-07-16_11-24-49_5686423788", # last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_11-25-04_5925545879/models/xlm-roberta-base.2023-07-16_11-25-04_5925545879", # random
    ],
    "tamil": [
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_11-25-29_9403181318/models/xlm-roberta-base.2023-07-16_11-25-29_9403181318", # init-1
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_17-24-31_2325448622/models/xlm-roberta-base.2023-07-16_17-24-31_2325448622", # middle
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_17-26-12_5975607309/models/xlm-roberta-base.2023-07-16_17-26-12_5975607309", # last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-16_23-37-25_066774395/models/xlm-roberta-base.2023-07-16_23-37-25_066774395", # random
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_22-44-46_212098616/models/xlm-roberta-base.2023-07-20_22-44-46_212098616", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_22-44-28_0353674208/models/xlm-roberta-base.2023-07-20_22-44-28_0353674208", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_22-43-41_488355135/models/xlm-roberta-base.2023-07-20_22-43-41_488355135", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-20_22-43-19_6617129753/models/xlm-roberta-base.2023-07-20_22-43-19_6617129753", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_11-42-04_8382104778/models/xlm-roberta-base.2023-07-19_11-42-04_8382104778", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_15-31-55_8424856548/models/xlm-roberta-base.2023-07-19_15-31-55_8424856548", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_15-32-10_4496874924/models/xlm-roberta-base.2023-07-19_15-32-10_4496874924", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-19_15-32-46_2086799825/models/xlm-roberta-base.2023-07-19_15-32-46_2086799825", # Random
    ]
}
ALGO = "mend" # or mend
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
