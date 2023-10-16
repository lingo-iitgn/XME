import os
import time

#########################################################################
CUDA = "0"
# fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
fine_tuned_langs = ["mixed", "inverse"]
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
    ]
}
ALGO = "efk" # or mend
MODEL_NAME = "xlm-roberta" # bloom-560m or # mbert-uncased # or xlm-roberta

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
