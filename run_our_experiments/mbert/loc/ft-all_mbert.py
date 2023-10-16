import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "mixed", "bengali", "gujarati"]
MLP_MODELS_ALL_LANG = {
    "english": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_16-20-18_0401791432/models/bert-base-multilingual-uncased.2023-05-24_16-20-18_0401791432.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-10-09_0353596389/models/bert-base-multilingual-uncased.2023-05-23_22-10-09_0353596389.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-10-25_6106628508/models/bert-base-multilingual-uncased.2023-05-23_22-10-25_6106628508.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-23_22-12-00_9455063617/models/bert-base-multilingual-uncased.2023-05-23_22-12-00_9455063617.bk", # random
    ],
    "hindi": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-06_3405599305/models/bert-base-multilingual-uncased.2023-05-24_10-10-06_3405599305.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-09-55_1229417955/models/bert-base-multilingual-uncased.2023-05-24_10-09-55_1229417955.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-17_0686086437/models/bert-base-multilingual-uncased.2023-05-24_10-10-17_0686086437.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-10-27_8215977427/models/bert-base-multilingual-uncased.2023-05-24_10-10-27_8215977427.bk", # random
    ],
    "spanish": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-24_3954272123/models/bert-base-multilingual-uncased.2023-05-24_10-06-24_3954272123.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-38_70220263/models/bert-base-multilingual-uncased.2023-05-24_10-06-38_70220263.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-06-53_1926292068/models/bert-base-multilingual-uncased.2023-05-24_10-06-53_1926292068.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-08_0791662798/models/bert-base-multilingual-uncased.2023-05-24_10-07-08_0791662798.bk", # random
    ],
    "french": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-45_2863131821/models/bert-base-multilingual-uncased.2023-05-24_10-07-45_2863131821.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-07-54_779437535/models/bert-base-multilingual-uncased.2023-05-24_10-07-54_779437535.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-08-11_3843116404/models/bert-base-multilingual-uncased.2023-05-24_10-08-11_3843116404.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-24_10-08-21_701374549/models/bert-base-multilingual-uncased.2023-05-24_10-08-21_701374549.bk", # random
    ],
    "mixed": [
        "/home/anonymous-xme/mend/mend/outputs/2023-05-26_22-51-06_1473177782/models/bert-base-multilingual-uncased.2023-05-26_22-51-06_1473177782.bk", # init-layers-1 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-39-06_0120429365/models/bert-base-multilingual-uncased.2023-05-27_08-39-06_0120429365.bk", # middle 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-39-44_0812473023/models/bert-base-multilingual-uncased.2023-05-27_08-39-44_0812473023.bk", # last-layer 
        "/home/anonymous-xme/mend/mend/outputs/2023-05-27_08-40-12_0599657239/models/bert-base-multilingual-uncased.2023-05-27_08-40-12_0599657239.bk", # random
    ],
    "bengali": [
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-init/models/bert-base-multilingual-uncased.2023-05-18_11-26-24_2212658845.bk", # Init
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-mid/models/bert-base-multilingual-uncased.2023-05-18_11-26-24_2210644196.bk", # Mid
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-last/models/bert-base-multilingual-uncased.2023-05-18_21-03-14_7543497565.bk", # Last
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-bn-random/models/bert-base-multilingual-uncased.2023-05-18_22-45-50_416362117.bk" # Random
    ],
    "gujarati": [
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-init/models/bert-base-multilingual-uncased.2023-05-17_16-14-09_4321415119.bk", # Init
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-mid/models/bert-base-multilingual-uncased.2023-05-17_16-18-20_7283902370.bk", # Mid
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-last/models/bert-base-multilingual-uncased.2023-05-17_22-52-02_2274023948.bk", # Last
        "/home/anonymous-xme/mend/mend/param-outputs/ft_models/ft-mbert-gj-random/models/bert-base-multilingual-uncased.2023-05-17_22-52-02_2217456337.bk" # Random
    ]
}
ALGO = "ft" # or mend
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
