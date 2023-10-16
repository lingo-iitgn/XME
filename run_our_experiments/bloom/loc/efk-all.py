import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"]
MLP_MODELS_ALL_LANG = {
    "english": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-15_20-49-24_5071488770/models/bloom-560m.2023-05-15_20-49-24_5071488770.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-15_22-52-29_3919411747/models/bloom-560m.2023-05-15_22-52-29_3919411747.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-16_10-33-21_0185677885/models/bloom-560m.2023-05-16_10-33-21_0185677885.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-16_16-36-19_4183412304/models/bloom-560m.2023-05-16_16-36-19_4183412304.bk", # random
    ],
    "hindi": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-28_15-40-26_460116717/models/bloom-560m.2023-05-28_15-40-26_460116717.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-28_15-43-44_1991694767/models/bloom-560m.2023-05-28_15-43-44_1991694767.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-28_15-42-26_2276132552/models/bloom-560m.2023-05-28_15-42-26_2276132552.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_19-39-03_9086568490/models/bloom-560m.2023-05-17_19-39-03_9086568490.bk", # random
    ],
    "spanish": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-06-04_124101654/models/bloom-560m.2023-05-19_12-06-04_124101654.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-07-17_8717001122/models/bloom-560m.2023-05-19_12-07-17_8717001122.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_12-27-47_2465729442/models/bloom-560m.2023-05-19_12-27-47_2465729442.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-18_11-55-55_7504809446/models/bloom-560m.2023-05-18_11-55-55_7504809446.bk", # random 
    ],
    "french": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-14_9621509609/models/bloom-560m.2023-05-17_09-58-14_9621509609.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-35_135676834/models/bloom-560m.2023-05-17_09-58-35_135676834.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-58-52_01149571/models/bloom-560m.2023-05-17_09-58-52_01149571.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_09-59-04_3104498918/models/bloom-560m.2023-05-17_09-59-04_3104498918.bk", # random
    ],
    "bengali": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_19-39-44_6501298990/models/bloom-560m.2023-05-17_19-39-44_6501298990.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_22-00-05_2125121338/models/bloom-560m.2023-05-17_22-00-05_2125121338.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-17_22-00-24_6797341142/models/bloom-560m.2023-05-17_22-00-24_6797341142.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_13-14-16_6385978402/models/bloom-560m.2023-05-19_13-14-16_6385978402.bk", # random
    ],
    "gujarati": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-18_13-46-42_393473952/models/bloom-560m.2023-05-18_13-46-42_393473952.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-18_15-26-19_3197959169/models/bloom-560m.2023-05-18_15-26-19_3197959169.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-18_16-22-43_9384049221/models/bloom-560m.2023-05-18_16-22-43_9384049221.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-18_17-33-07_4839803226/models/bloom-560m.2023-05-18_17-33-07_4839803226.bk", # random
    ],
        "mixed": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_07-59-58_3846481088/models/bloom-560m.2023-05-26_07-59-58_3846481088.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-00-41_4297485516/models/bloom-560m.2023-05-26_08-00-41_4297485516.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-01-46_8249958419/models/bloom-560m.2023-05-26_08-01-46_8249958419.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-26_08-03-14_8441583559/models/bloom-560m.2023-05-26_08-03-14_8441583559.bk", # random
    ],
        "inverse": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-19_8638215499/models/bloom-560m.2023-05-19_08-50-19_8638215499.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-39_2927883181/models/bloom-560m.2023-05-19_08-50-39_2927883181.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_08-50-55_6361688606/models/bloom-560m.2023-05-19_08-50-55_6361688606.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-19_11-10-02_6770262665/models/bloom-560m.2023-05-19_11-10-02_6770262665.bk", # random
    ]
}
ALGO = "efk" # or mend
MODEL_NAME = "bloom-560m" # bloom-560m or # mbert-uncased # or xlm-roberta

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
