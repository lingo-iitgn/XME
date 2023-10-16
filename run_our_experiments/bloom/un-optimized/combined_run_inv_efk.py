import os
import time

#########################################################################
CUDA = "3"
lang = "inverse" # or hindi or spanish or french or bengali or gujarati
MLP_MODELS = [
    # "/home/anonymous-xme/mend/mend/outputs/2023-06-13_15-52-57_3316462877/models/bloom-560m.2023-06-13_15-52-57_3316462877.bk", # Init
    # "/home/anonymous-xme/mend/mend/outputs/2023-06-13_18-20-23_9410038538/models/bloom-560m.2023-06-13_18-20-23_9410038538.bk", # Mid
    # "/home/anonymous-xme/mend/mend/outputs/2023-06-14_00-18-44_7459236285/models/bloom-560m.2023-06-14_00-18-44_7459236285.bk", # Last
    "/home/anonymous-xme/mend/mend/outputs/2023-06-14_00-19-04_0353452142/models/bloom-560m.2023-06-14_00-19-04_0353452142.bk" # Random
    ]
ALGO = "efk" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased xlm-roberta

#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
MODELS = [
    # f"{MODEL_NAME}-{lang}-init-layers-1",
    # f"{MODEL_NAME}-{lang}-middle",
    # f"{MODEL_NAME}-{lang}-last-layer",
    f"{MODEL_NAME}-{lang}-random"
    ]
run_names = [
    # f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
    # f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_middle",
    # f"{ALGO}_exp_1_{MODEL_NAME}_finetuned_{lang}_last_layer",
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
            os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} ++loc_acc=True +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/{ALGO}-{MODEL_NAME}-{l1}-{l2}-{MODEL}.txt""")
