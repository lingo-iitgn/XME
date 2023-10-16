import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"]
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
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-09_21-25-08_5351738589/models/xlm-roberta-base.2023-05-09_21-25-08_5351738589.bk", # init-1 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_12-00-01_0528815195/models/xlm-roberta-base.2023-05-10_12-00-01_0528815195.bk", # middle 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_17-09-16_6470294451/models/xlm-roberta-base.2023-05-10_17-09-16_6470294451.bk", # last 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-10_22-51-55_6228978753/models/xlm-roberta-base.2023-05-10_22-51-55_6228978753.bk", # random
    ],
    "gujarati": [
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-11_19-14-27_0101964695/models/xlm-roberta-base.2023-05-11_19-14-27_0101964695.bk", # init-1 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-11_19-16-11_4652116050/models/xlm-roberta-base.2023-05-11_19-16-11_4652116050.bk", # middle 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-12_03-05-44_6733494242/models/xlm-roberta-base.2023-05-12_03-05-44_6733494242.bk", # last 
 "/home/anonymous-xme/mend/mend/pheonix-outputs/outputs/2023-05-12_03-06-02_9380943110/models/xlm-roberta-base.2023-05-12_03-06-02_9380943110.bk", # random
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
    ]
}
ALGO = "mend" # or mend
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
