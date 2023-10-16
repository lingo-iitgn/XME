import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed"]
# fine_tuned_langs = ["gujarati", "mixed"]
MLP_MODELS_ALL_LANG = {
    "english": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-34-40_5852964634/models/bert-base-multilingual-uncased.2023-05-14_11-34-40_5852964634.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-35-47_8556404848/models/bert-base-multilingual-uncased.2023-05-14_11-35-47_8556404848.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-36-29_4202131856/models/bert-base-multilingual-uncased.2023-05-14_11-36-29_4202131856.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-37-08_6182045094/models/bert-base-multilingual-uncased.2023-05-14_11-37-08_6182045094.bk", # random
    ],
    "hindi": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-37-54_7536038429/models/bert-base-multilingual-uncased.2023-05-14_11-37-54_7536038429.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-11_9776896694/models/bert-base-multilingual-uncased.2023-05-14_11-38-11_9776896694.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-24_134127150/models/bert-base-multilingual-uncased.2023-05-14_11-38-24_134127150.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-38-41_0646491589/models/bert-base-multilingual-uncased.2023-05-14_11-38-41_0646491589.bk", # random 
    ],
    "spanish": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-45-44_6501072104/models/bert-base-multilingual-uncased.2023-05-12_10-45-44_6501072104.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-46-18_5630618291/models/bert-base-multilingual-uncased.2023-05-12_10-46-18_5630618291.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-46-30_6818392507/models/bert-base-multilingual-uncased.2023-05-12_10-46-30_6818392507.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_10-47-33_4254983106/models/bert-base-multilingual-uncased.2023-05-12_10-47-33_4254983106.bk", # random
    ],
    "french": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_14-17-52_5309692936/models/bert-base-multilingual-uncased.2023-05-12_14-17-52_5309692936.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_15-58-13_5558261238/models/bert-base-multilingual-uncased.2023-05-12_15-58-13_5558261238.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-12_15-58-32_9549386833/models/bert-base-multilingual-uncased.2023-05-12_15-58-32_9549386833.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_12-26-48_7814916951/models/bert-base-multilingual-uncased.2023-05-17_12-26-48_7814916951.bk", # random 
    ],
    "bengali": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk", # random
    ],
    "gujarati": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-15_7656665371/models/bert-base-multilingual-uncased.2023-05-14_11-42-15_7656665371.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_18-22-01_0807861433/models/bert-base-multilingual-uncased.2023-05-17_18-22-01_0807861433.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-33_3023077685/models/bert-base-multilingual-uncased.2023-05-14_11-42-33_3023077685.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-42-45_5556532582/models/bert-base-multilingual-uncased.2023-05-14_11-42-45_5556532582.bk", # random
    ],
        "mixed": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-08-12_3353236923/models/bert-base-multilingual-uncased.2023-05-26_14-08-12_3353236923.bk", # init-layers-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-08-27_9096099175/models/bert-base-multilingual-uncased.2023-05-26_14-08-27_9096099175.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_14-55-27_0531847123/models/bert-base-multilingual-uncased.2023-05-26_14-55-27_0531847123.bk", # last-layer 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_16-54-20_0510821385/models/bert-base-multilingual-uncased.2023-05-26_16-54-20_0510821385.bk", # random
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
            isExist = os.path.exists(f"./logs-locality/{run_name}/{l1}")
            if not isExist:
                os.makedirs(f"./logs-locality/{run_name}/{l1}")

        time.sleep(3) # Wait for os process to complete

        for l1 in file_names:
            for l2 in file_names:
                print("="*40, l1, l2, "="*40)
                os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={ALGO} ++loc_acc=True +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs-locality/{run_name}/{l1}/{ALGO}-{MODEL_NAME}-{l1}-{l2}-{MODEL}.txt""")




# lang = "bengali" # or hindi or spanish or french or bengali or gujarati
# MLP_MODELS = [
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-44_3771899945/models/bert-base-multilingual-uncased.2023-05-14_11-40-44_3771899945.bk", # Init
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-40-54_0354855685/models/bert-base-multilingual-uncased.2023-05-14_11-40-54_0354855685.bk", # Mid
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-04_5524874778/models/bert-base-multilingual-uncased.2023-05-14_11-41-04_5524874778.bk", # Last
#     "/home/anonymous-xme/mend/mend/outputs/2023-05-14_11-41-13_9762051720/models/bert-base-multilingual-uncased.2023-05-14_11-41-13_9762051720.bk" # Random
#     ]
