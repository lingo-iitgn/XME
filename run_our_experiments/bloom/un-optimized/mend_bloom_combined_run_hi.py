import os
import time

#########################################################################
CUDA = "1"
lang = "hindi" # or hindi or spanish or french or bengali or gujarati
MLP_MODELS = [
    "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-15-45_5272864186/models/bloom-560m.2023-05-22_18-15-45_5272864186.bk", # Init
    "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-17-06_131347881/models/bloom-560m.2023-05-22_18-17-06_131347881.bk", # Mid
    "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-18-28_5631311707/models/bloom-560m.2023-05-22_18-18-28_5631311707.bk", # Last
    "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-19-00_6843471205/models/bloom-560m.2023-05-22_18-19-00_6843471205.bk" # Random
    ]
ALGO = "mend" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased xlm-roberta

#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
MODELS = [
    f"{MODEL_NAME}-{lang}-init-layers-1",
    f"{MODEL_NAME}-{lang}-middle",
    f"{MODEL_NAME}-{lang}-last-layer",
    f"{MODEL_NAME}-{lang}-random"
    ]
run_names = [
    f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
    f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_middle",
    f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_last_layer",
    f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_random"
    ]

for MODEL, MLP_MODEL, run_name in zip(MODELS, MLP_MODELS, run_names):
    file_names = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
    for l1 in file_names:
        isExist = os.path.exists(f"./logs/{run_name}/{l1}")
        if not isExist:
            os.makedirs(f"./logs/{run_name}/{l1}")

    time.sleep(3) # Wait for os process to complete

    for l1 in file_names:
        for l2 in file_names:
            print("="*40, l1, l2, "="*40)
            os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/{ALGO}-{MODEL_NAME}-{l1}-{l2}-{MODEL}.txt""")
