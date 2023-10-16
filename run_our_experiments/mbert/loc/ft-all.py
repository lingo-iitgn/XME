import os
import time

#########################################################################
CUDA = "2"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
MLP_MODELS_ALL_LANG = {
    "english": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ],
    "hindi": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ],
    "spanish": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ],
    "french": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ],
    "bengali": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ],
    "gujarati": [
        "", # Init
        "", # Mid
        "", # Last
        "" # Random
    ]
}
ALGO = "efk" # or mend
MODEL_NAME = "mbert-uncased" # bloom-560m or # mbert-uncased # or xlm-roberta

#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
for lang in fine_tuned_langs:
    print("*"*100, "Finetuned Language -", lang, "*"*100)
    MODELS = [
        f"{MODEL_NAME}-{lang}-init-layers-1",
        f"{MODEL_NAME}-{lang}-middle",
        f"{MODEL_NAME}-{lang}-last-layer",
        f"{MODEL_NAME}-{lang}-random"
        ]
    run_names = [
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random"
        ]

    for MODEL, MLP_MODEL, run_name in zip(MODELS, MLP_MODELS_ALL_LANG[lang], run_names):
        # file_names = ["english"]
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




# lang = "bengali" # or hindi or spanish or french or bengali or gujarati
# MLP_MODELS = [
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # Init
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # Mid
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # Last
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk" # Random
#     ]
