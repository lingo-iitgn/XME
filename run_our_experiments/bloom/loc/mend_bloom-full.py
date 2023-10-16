import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["hindi", "spanish", "french", "bengali", "gujarati"]
MLP_MODELS_ALL_LANG = {
    "english": [
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-init-layers-1-pred/models/bloom-560m.2023-04-17_14-15-16_7851742737.bk", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-middle-layers-pred/models/bloom-560m.2023-04-17_22-18-54_5163214677.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-last-layers-pred/models/bloom-560m.2023-04-17_22-21-38_5418723295.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-random-layers-pred/models/bloom-560m.2023-04-17_22-25-12_9476916002.bk", # random
    ],
    "hindi": [
 "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-48-05_5521157428/models/bloom-560m.2023-07-11_09-48-05_5521157428", # init-1 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-17-06_131347881/models/bloom-560m.2023-05-22_18-17-06_131347881.bk", # middle 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-18-28_5631311707/models/bloom-560m.2023-05-22_18-18-28_5631311707.bk", # last 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-28_20-31-50_2907878295/models/bloom-560m.2023-05-28_20-31-50_2907878295.bk", # random
    ],
    "spanish": [
 "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-49-45_3722652815/models/bloom-560m.2023-07-11_09-49-45_3722652815", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-middle/models/bloom-560m.2023-05-07_21-39-38_1634424205.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-last/models/bloom-560m.2023-05-08_00-50-00_398157700.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-random/models/bloom-560m.2023-05-08_08-54-12_3978277641.bk", # random
    ],
    "french": [
 "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-01_4241744051/models/bloom-560m.2023-07-11_09-50-01_4241744051", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-middle-layers/models/bloom-560m.2023-04-18_09-09-17_4676312842.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-last-layers/models/bloom-560m.2023-04-18_09-10-01_6896549296.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-random/models/bloom-560m.2023-04-18_09-10-44_6154188477.bk", # random
    ],
    "bengali": [
 "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-15_8276813064/models/bloom-560m.2023-07-11_09-50-15_8276813064", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-middle-pred/models/bloom-560m.2023-04-20_01-31-18_6374192842.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-last-layer-pred/models/bloom-560m.2023-04-19_21-09-57_7940862913.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-random-pred/models/bloom-560m.2023-04-19_23-47-19_7805968414.bk", # random
    ],
    "gujarati": [
 "/home/anonymous-xme/mend/mend/outputs/2023-07-11_09-50-27_0883794743/models/bloom-560m.2023-07-11_09-50-27_0883794743", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-middle/models/bloom-560m.2023-04-23_11-37-22_3909366631.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-last-layer/models/bloom-560m.2023-04-22_23-10-46_2170927579.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-random/models/bloom-560m.2023-04-23_00-27-13_7780401534.bk", # random
    ],
        "mixed": [
#  "/home/anonymous-xme/mend/mend/outputs/2023-07-11_22-37-15_6411918785/models/bloom-560m.2023-07-11_22-37-15_6411918785", # init-1 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-06_7141775050/models/bloom-560m.2023-05-27_18-16-06_7141775050.bk", # middle 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-22_4494132059/models/bloom-560m.2023-05-27_18-16-22_4494132059.bk", # last 
#  "/home/anonymous-xme/mend/mend/outputs/2023-05-27_22-55-43_7215556499/models/bloom-560m.2023-05-27_22-55-43_7215556499.bk", # random 
    ],
#         "inverse": [
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-init/models/bloom-560m.2023-05-05_08-47-43_7150709473.bk", # init-1 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-mid/models/bloom-560m.2023-05-05_08-50-27_6644632166.bk", # middle 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-last/models/bloom-560m.2023-05-05_18-42-12_02719454.bk", # last 
#  "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-random/models/bloom-560m.2023-05-05_21-57-28_4566237195.bk", # random
    # ]
}
ALGO = "mend" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased # or xlm-roberta

#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
for lang in fine_tuned_langs:
    print("*"*100, "Finetuned Language -", lang, "*"*100)
    MODELS = [
        f"{MODEL_NAME}-{lang}-full",
        # f"{MODEL_NAME}-{lang}-init-layers-1",
        # f"{MODEL_NAME}-{lang}-middle",
        # f"{MODEL_NAME}-{lang}-last-layer",
        # f"{MODEL_NAME}-{lang}-random"
        ]
    run_names = [
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_full",
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle",
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer",
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random"
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
