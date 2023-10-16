import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"]
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-14-48_6220701091/models/bloom-560m.2023-05-23_21-14-48_6220701091.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-18_6204968017/models/bloom-560m.2023-05-23_21-18-18_6204968017.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-34_2382335872/models/bloom-560m.2023-05-23_21-18-34_2382335872.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_21-18-56_6338966671/models/bloom-560m.2023-05-23_21-18-56_6338966671.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-28_21-49-43_994718911/models/bloom-560m.2023-05-28_21-49-43_994718911.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_00-58-24_117440303/models/bloom-560m.2023-05-29_00-58-24_117440303.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_00-57-28_730516172/models/bloom-560m.2023-05-29_00-57-28_730516172.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-29_09-37-37_5483895578/models/bloom-560m.2023-05-29_09-37-37_5483895578.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-22-57_0885661488/models/bloom-560m.2023-05-18_09-22-57_0885661488.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-23-20_4810735181/models/bloom-560m.2023-05-18_09-23-20_4810735181.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-18_09-23-44_6148828703/models/bloom-560m.2023-05-18_09-23-44_6148828703.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_21-25-09_9521788822/models/bloom-560m.2023-05-19_21-25-09_9521788822.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-50-36_5773096850/models/bloom-560m.2023-05-20_00-50-36_5773096850.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-50-59_1758105868/models/bloom-560m.2023-05-20_00-50-59_1758105868.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_00-51-33_3903632588/models/bloom-560m.2023-05-20_00-51-33_3903632588.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_15-25-26_5166341895/models/bloom-560m.2023-05-24_15-25-26_5166341895.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_13-16-25_3233686047/models/bloom-560m.2023-05-19_13-16-25_3233686047.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_16-55-54_3887693086/models/bloom-560m.2023-05-19_16-55-54_3887693086.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_19-45-52_9347853383/models/bloom-560m.2023-05-19_19-45-52_9347853383.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-19_19-46-13_6593971241/models/bloom-560m.2023-05-19_19-46-13_6593971241.bk", # random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-00_2718541665/models/bloom-560m.2023-05-20_09-17-00_2718541665.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-13_4070551883/models/bloom-560m.2023-05-20_09-17-13_4070551883.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_09-17-42_2876716632/models/bloom-560m.2023-05-20_09-17-42_2876716632.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_10-51-13_3010343764/models/bloom-560m.2023-05-20_10-51-13_3010343764.bk", # random
    ],
        "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_12-23-00_263854377/models/bloom-560m.2023-05-20_12-23-00_263854377.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_14-30-57_2029015311/models/bloom-560m.2023-05-20_14-30-57_2029015311.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_17-04-01_0643215465/models/bloom-560m.2023-05-20_17-04-01_0643215465.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-20_17-58-24_8563683228/models/bloom-560m.2023-05-20_17-58-24_8563683228.bk", # random
    ],
        "inverse": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-21_15-01-58_144254610/models/bloom-560m.2023-05-21_15-01-58_144254610.bk", # init-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-25_10-39-31_7590953107/models/bloom-560m.2023-05-25_10-39-31_7590953107.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_07-10-45_27671139/models/bloom-560m.2023-05-22_07-10-45_27671139.bk", # last 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-22_07-11-05_3339658470/models/bloom-560m.2023-05-22_07-11-05_3339658470.bk", # random 
    ]
}
ALGO = "ft" # or mend
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
