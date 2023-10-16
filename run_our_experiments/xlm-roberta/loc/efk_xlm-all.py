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
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-28-56_3148842532/models/xlm-roberta-base.2023-05-12_09-28-56_3148842532.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_11-49-19_9373316829/models/xlm-roberta-base.2023-05-12_11-49-19_9373316829.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-29-48_7404285229/models/xlm-roberta-base.2023-05-12_09-29-48_7404285229.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-30-27_8010821766/models/xlm-roberta-base.2023-05-12_09-30-27_8010821766.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_11-50-00_1684636363/models/xlm-roberta-base.2023-05-12_11-50-00_1684636363.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-31-19_4961186513/models/xlm-roberta-base.2023-05-12_09-31-19_4961186513.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-31-31_0217195539/models/xlm-roberta-base.2023-05-12_09-31-31_0217195539.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_09-32-04_8757218141/models/xlm-roberta-base.2023-05-12_09-32-04_8757218141.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_20-38-34_8385012284/models/xlm-roberta-base.2023-05-12_20-38-34_8385012284.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_20-39-17_4783244175/models/xlm-roberta-base.2023-05-12_20-39-17_4783244175.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_20-40-00_8317926116/models/xlm-roberta-base.2023-05-12_20-40-00_8317926116.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_20-40-26_4231854404/models/xlm-roberta-base.2023-05-12_20-40-26_4231854404.bk", # random 
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_22-35-39_3965652289/models/xlm-roberta-base.2023-05-12_22-35-39_3965652289.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_22-36-21_8738716030/models/xlm-roberta-base.2023-05-12_22-36-21_8738716030.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-12_22-37-02_4699529643/models/xlm-roberta-base.2023-05-12_22-37-02_4699529643.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-13_09-04-44_4967714570/models/xlm-roberta-base.2023-05-13_09-04-44_4967714570.bk", # random 
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-53-28_7191798143/models/xlm-roberta-base.2023-05-29_15-53-28_7191798143.bk", # init-layers-1  # BK file was not present
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-53-46_6593919900/models/xlm-roberta-base.2023-05-29_15-53-46_6593919900.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-56-01_8776156865/models/xlm-roberta-base.2023-05-29_15-56-01_8776156865.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-54-46_9262004805/models/xlm-roberta-base.2023-05-29_15-54-46_9262004805.bk", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-56-51_6261332930/models/xlm-roberta-base.2023-05-29_15-56-51_6261332930.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_15-58-17_5732123880/models/xlm-roberta-base.2023-05-29_15-58-17_5732123880.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_20-07-33_132744167/models/xlm-roberta-base.2023-05-29_20-07-33_132744167.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_20-07-43_8896427329/models/xlm-roberta-base.2023-05-29_20-07-43_8896427329.bk", # random
    ],
        "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_16-56-14_7125142267/models/xlm-roberta-base.2023-05-26_16-56-14_7125142267.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_16-58-51_7566819582/models/xlm-roberta-base.2023-05-26_16-58-51_7566819582.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_18-44-04_7689751165/models/xlm-roberta-base.2023-05-26_18-44-04_7689751165.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_19-20-09_1954179375/models/xlm-roberta-base.2023-05-26_19-20-09_1954179375.bk", # random
    ],
        "inverse": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-13_09-11-16_7896262328/models/xlm-roberta-base.2023-05-13_09-11-16_7896262328.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-13_09-11-32_6459429562/models/xlm-roberta-base.2023-05-13_09-11-32_6459429562.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-13_10-06-05_9667857331/models/xlm-roberta-base.2023-05-13_10-06-05_9667857331.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-13_10-06-27_9602081558/models/xlm-roberta-base.2023-05-13_10-06-27_9602081558.bk", # random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_19-59-06_8181338873/models/xlm-roberta-base.2023-07-26_19-59-06_8181338873", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_19-59-25_4977027615/models/xlm-roberta-base.2023-07-26_19-59-25_4977027615", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_08-49-49_939962410/models/xlm-roberta-base.2023-07-28_08-49-49_939962410", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_20-32-30_0867861730/models/xlm-roberta-base.2023-07-26_20-32-30_0867861730", # Random    
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_17-38-26_827216149/models/xlm-roberta-base.2023-07-27_17-38-26_827216149", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_20-58-53_6075608050/models/xlm-roberta-base.2023-07-27_20-58-53_6075608050", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_13-35-11_7392966951/models/xlm-roberta-base.2023-07-26_13-35-11_7392966951", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_13-35-27_2199507011/models/xlm-roberta-base.2023-07-26_13-35-27_2199507011", # Random    
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_14-57-06_6675866440/models/xlm-roberta-base.2023-07-26_14-57-06_6675866440", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_19-58-05_5344572350/models/xlm-roberta-base.2023-07-26_19-58-05_5344572350", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_19-58-21_2957488865/models/xlm-roberta-base.2023-07-26_19-58-21_2957488865", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_19-58-31_5341982385/models/xlm-roberta-base.2023-07-26_19-58-31_5341982385", # Random    
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-46-27_8440586007/models/xlm-roberta-base.2023-07-27_08-46-27_8440586007", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-46-52_205885636/models/xlm-roberta-base.2023-07-27_08-46-52_205885636", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-47-02_5002007147/models/xlm-roberta-base.2023-07-27_08-47-02_5002007147", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-47-13_0787102518/models/xlm-roberta-base.2023-07-27_08-47-13_0787102518", # Random    
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_20-32-51_6118988765/models/xlm-roberta-base.2023-07-26_20-32-51_6118988765", # Init
        "/home/anonymous-xme/mend/mend/outputs/2023-07-26_22-10-51_4776456331/models/xlm-roberta-base.2023-07-26_22-10-51_4776456331", # Mid
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-45-50_4536979489/models/xlm-roberta-base.2023-07-27_08-45-50_4536979489", # Last
        "/home/anonymous-xme/mend/mend/outputs/2023-07-27_08-46-03_4825459180/models/xlm-roberta-base.2023-07-27_08-46-03_4825459180", # Random    
    ]
}
ALGO = "efk" # or mend
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
