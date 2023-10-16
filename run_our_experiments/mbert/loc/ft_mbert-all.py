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
fine_tuned_langs = ["kannada", "malayalam", "tamil", "english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "chinese", "arabic"] #  "inverse"
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_16-20-18_0401791432/models/bert-base-multilingual-uncased.2023-05-24_16-20-18_0401791432.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-10-09_0353596389/models/bert-base-multilingual-uncased.2023-05-23_22-10-09_0353596389.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-10-25_6106628508/models/bert-base-multilingual-uncased.2023-05-23_22-10-25_6106628508.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-12-00_9455063617/models/bert-base-multilingual-uncased.2023-05-23_22-12-00_9455063617.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-06_3405599305/models/bert-base-multilingual-uncased.2023-05-24_10-10-06_3405599305.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-09-55_1229417955/models/bert-base-multilingual-uncased.2023-05-24_10-09-55_1229417955.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-17_0686086437/models/bert-base-multilingual-uncased.2023-05-24_10-10-17_0686086437.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-27_8215977427/models/bert-base-multilingual-uncased.2023-05-24_10-10-27_8215977427.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-24_3954272123/models/bert-base-multilingual-uncased.2023-05-24_10-06-24_3954272123.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-38_70220263/models/bert-base-multilingual-uncased.2023-05-24_10-06-38_70220263.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-53_1926292068/models/bert-base-multilingual-uncased.2023-05-24_10-06-53_1926292068.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-08_0791662798/models/bert-base-multilingual-uncased.2023-05-24_10-07-08_0791662798.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-45_2863131821/models/bert-base-multilingual-uncased.2023-05-24_10-07-45_2863131821.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-54_779437535/models/bert-base-multilingual-uncased.2023-05-24_10-07-54_779437535.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-08-11_3843116404/models/bert-base-multilingual-uncased.2023-05-24_10-08-11_3843116404.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-08-21_701374549/models/bert-base-multilingual-uncased.2023-05-24_10-08-21_701374549.bk", # random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_22-51-06_1473177782/models/bert-base-multilingual-uncased.2023-05-26_22-51-06_1473177782.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-39-06_0120429365/models/bert-base-multilingual-uncased.2023-05-27_08-39-06_0120429365.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-39-44_0812473023/models/bert-base-multilingual-uncased.2023-05-27_08-39-44_0812473023.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-40-12_0599657239/models/bert-base-multilingual-uncased.2023-05-27_08-40-12_0599657239.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-init/models/bert-base-multilingual-uncased.2023-05-18_11-26-24_2212658845.bk", # Init
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-mid/models/bert-base-multilingual-uncased.2023-05-18_11-26-24_2210644196.bk", # Mid
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-last/models/bert-base-multilingual-uncased.2023-05-18_21-03-14_7543497565.bk", # Last
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-random/models/bert-base-multilingual-uncased.2023-05-18_22-45-50_416362117.bk" # Random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-init/models/bert-base-multilingual-uncased.2023-05-17_16-14-09_4321415119.bk", # Init
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-mid/models/bert-base-multilingual-uncased.2023-05-17_16-18-20_7283902370.bk", # Mid
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-last/models/bert-base-multilingual-uncased.2023-05-17_22-52-02_2274023948.bk", # Last
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-random/models/bert-base-multilingual-uncased.2023-05-17_22-52-02_2217456337.bk" # Random
    ],
    "kannada": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-56-20_7935091637/models/bert-base-multilingual-uncased.2023-08-01_22-56-20_7935091637",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-56-31_5129326511/models/bert-base-multilingual-uncased.2023-08-01_22-56-31_5129326511",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-56-43_6775217142/models/bert-base-multilingual-uncased.2023-08-01_22-56-43_6775217142",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-56-53_6922928462/models/bert-base-multilingual-uncased.2023-08-01_22-56-53_6922928462", # Random
    ],
    "malayalam": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-07-28_21-31-37_4606375/models/bert-base-multilingual-uncased.2023-07-28_21-31-37_4606375",
        "/home/anonymous-xme/mend/mend/outputs/2023-07-29_08-53-20_6282574924/models/bert-base-multilingual-uncased.2023-07-29_08-53-20_6282574924",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-53-06_7629862246/models/bert-base-multilingual-uncased.2023-08-01_22-53-06_7629862246",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-53-21_4947153595/models/bert-base-multilingual-uncased.2023-08-01_22-53-21_4947153595", # Random
    ],
    "tamil": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-55-22_8617698181/models/bert-base-multilingual-uncased.2023-08-01_22-55-22_8617698181",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-55-32_8302459150/models/bert-base-multilingual-uncased.2023-08-01_22-55-32_8302459150",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-55-44_8662782702/models/bert-base-multilingual-uncased.2023-08-01_22-55-44_8662782702",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-55-56_6298082929/models/bert-base-multilingual-uncased.2023-08-01_22-55-56_6298082929", # Random
    ],
    "arabic": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-02_09-49-00_3958364696/models/bert-base-multilingual-uncased.2023-08-02_09-49-00_3958364696",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-02_09-49-14_9373939531/models/bert-base-multilingual-uncased.2023-08-02_09-49-14_9373939531",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-02_09-49-31_8433291625/models/bert-base-multilingual-uncased.2023-08-02_09-49-31_8433291625",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-02_12-34-16_8317453346/models/bert-base-multilingual-uncased.2023-08-02_12-34-16_8317453346", # Random
    ],
    "chinese": [
        # "", # Full
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-58-30_2047982899/models/bert-base-multilingual-uncased.2023-08-01_22-58-30_2047982899",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-58-48_9198154582/models/bert-base-multilingual-uncased.2023-08-01_22-58-48_9198154582",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-02_10-32-33_6337709533/models/bert-base-multilingual-uncased.2023-08-02_10-32-33_6337709533",
        "/home/anonymous-xme/mend/mend/outputs/2023-08-01_22-59-09_0020794937/models/bert-base-multilingual-uncased.2023-08-01_22-59-09_0020794937", # Random
    ]
}
ALGO = "ft" # or mend
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
