import os
import time

#########################################################################
CUDA = "0"
# fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed"]
fine_tuned_langs = ["english"]
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-init/models/bert-base-multilingual-uncased.2023-04-29_00-55-35_9641872393.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-middle/models/bert-base-multilingual-uncased.2023-04-29_00-56-33_5943326592.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-last/models/bert-base-multilingual-uncased.2023-04-29_09-01-27_4375434466.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-english-random/models/bert-base-multilingual-uncased.2023-04-29_14-01-43_8384045142.bk" # Random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-init/models/bert-base-multilingual-uncased.2023-04-29_17-32-27_9874154149.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-middle/models/bert-base-multilingual-uncased.2023-04-30_08-13-27_0548981069.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-last/models/bert-base-multilingual-uncased.2023-04-30_14-30-29_7931785752.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/pheonix/mend-mbert-hindi-random/models/bert-base-multilingual-uncased.2023-04-30_18-14-07_9594416068.bk" # Random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-init/models/bert-base-multilingual-uncased.2023-05-07_08-51-19_7053891932.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-middle/models/bert-base-multilingual-uncased.2023-05-07_08-52-32_6264791285.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-last/models/bert-base-multilingual-uncased.2023-05-07_12-24-07_8752792427.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-spanish-random/models/bert-base-multilingual-uncased.2023-05-07_16-48-41_0990795766.bk" # Random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-init/models/bert-base-multilingual-uncased.2023-04-29_23-39-04_5919925380.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-middle/models/bert-base-multilingual-uncased.2023-04-29_23-40-16_8329158189.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-last/models/bert-base-multilingual-uncased.2023-04-29_23-40-34_9831073164.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-french-random/models/bert-base-multilingual-uncased.2023-04-29_23-40-54_6475544060.bk" # Random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-init/models/bert-base-multilingual-uncased.2023-04-30_07-41-05_083194805.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-middle/models/bert-base-multilingual-uncased.2023-04-30_07-42-06_9335413823.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-last/models/bert-base-multilingual-uncased.2023-04-30_07-42-49_9829232646.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-bengali-random/models/bert-base-multilingual-uncased.2023-04-30_07-43-28_6719361139.bk" # Random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-init/models/bert-base-multilingual-uncased.2023-05-01_00-23-44_7001051962.bk", # Init
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-middle/models/bert-base-multilingual-uncased.2023-05-01_00-24-12_6163771627.bk", # Mid
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-last/models/bert-base-multilingual-uncased.2023-05-01_00-26-55_5113497878.bk", # Last
        "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-mbert-uncased-gujarati-random/models/bert-base-multilingual-uncased.2023-05-01_00-27-14_7784651353.bk" # Random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_09-39-37_3949008727/models/bert-base-multilingual-uncased.2023-05-28_09-39-37_3949008727.bk", # Init
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_10-55-11_20371129/models/bert-base-multilingual-uncased.2023-05-28_10-55-11_20371129.bk", # Mid
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_22-29-25_3243749382/models/bert-base-multilingual-uncased.2023-05-28_22-29-25_3243749382.bk", # Last
        "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-28_22-29-39_0784372021/models/bert-base-multilingual-uncased.2023-05-28_22-29-39_0784372021.bk" # Random
    ]
}
ALGO = "mend" # or mend
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
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1-temp",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle-temp",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer-temp",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random-temp"
        ]

    for MODEL, MLP_MODEL, run_name in zip(MODELS, MLP_MODELS_ALL_LANG[lang], run_names):
        # file_names = ["english"]
        file_names = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
        # file_names = ["spanish", "french", "bengali", "gujarati"]
        for l1 in file_names:
            isExist = os.path.exists(f"./logs-locality/{run_name}/{l1}")
            if not isExist:
                os.makedirs(f"./logs-locality/{run_name}/{l1}")

        time.sleep(3) # Wait for os process to complete

        for l1 in file_names:
            for l2 in file_names:
                # if l1 == l2:
                #     continue
                print("="*40, l1, l2, "="*40)
                os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} ++loc_acc=True +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs-locality/{run_name}/{l1}/{ALGO}-{MODEL_NAME}-{l1}-{l2}-{MODEL}-test.txt""")
                exit()



# lang = "bengali" # or hindi or spanish or french or bengali or gujarati
# MLP_MODELS = [
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # Init
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # Mid
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # Last
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk" # Random
#     ]
