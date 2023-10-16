import os
import time

#########################################################################
CUDA = "3"
fine_tuned_langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"]
MLP_MODELS_ALL_LANG = {
    "english": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-07-59_5032786400/models/xlm-roberta-base.2023-05-14_20-07-59_5032786400.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_00-14-40_7757788123/models/xlm-roberta-base.2023-05-17_00-14-40_7757788123.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-11-45_7865563849/models/xlm-roberta-base.2023-05-14_20-11-45_7865563849.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_20-13-39_4869933751/models/xlm-roberta-base.2023-05-14_20-13-39_4869933751.bk", # random
    ],
    "hindi": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-14_23-30-59_0459095681/models/xlm-roberta-base.2023-05-14_23-30-59_0459095681.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-09-53_2398339608/models/xlm-roberta-base.2023-05-15_10-09-53_2398339608.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_01-39-56_322863626/models/xlm-roberta-base.2023-05-17_01-39-56_322863626.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-11-02_5015002508/models/xlm-roberta-base.2023-05-15_10-11-02_5015002508.bk", # random
    ],
    "spanish": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-12-01_5383549309/models/xlm-roberta-base.2023-05-15_10-12-01_5383549309.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-12-23_9541259083/models/xlm-roberta-base.2023-05-15_10-12-23_9541259083.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-51-00_3697356607/models/xlm-roberta-base.2023-05-15_10-51-00_3697356607.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-51-54_5882403922/models/xlm-roberta-base.2023-05-15_10-51-54_5882403922.bk", # random
    ],
    "french": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_10-52-54_0710392768/models/xlm-roberta-base.2023-05-15_10-52-54_0710392768.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_10-00-30_8708111462/models/xlm-roberta-base.2023-05-17_10-00-30_8708111462.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_11-53-14_6051049345/models/xlm-roberta-base.2023-05-15_11-53-14_6051049345.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_13-48-58_7310897893/models/xlm-roberta-base.2023-05-15_13-48-58_7310897893.bk", # random 
    ],
    "mixed": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_13-52-01_3377527860/models/xlm-roberta-base.2023-05-15_13-52-01_3377527860.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_14-23-30_2667904402/models/xlm-roberta-base.2023-05-15_14-23-30_2667904402.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-24_15-32-54_6661435069/models/xlm-roberta-base.2023-05-24_15-32-54_6661435069.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_15-15-05_434074876/models/xlm-roberta-base.2023-05-15_15-15-05_434074876.bk", # random
    ],
    "inverse": [
      "/home/anonymous-xme/mend/mend/outputs/2023-05-16_23-40-53_5017337244/models/xlm-roberta-base.2023-05-16_23-40-53_5017337244.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-17_00-25-43_1459539178/models/xlm-roberta-base.2023-05-17_00-25-43_1459539178.bk", # middle 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_16-38-00_817632663/models/xlm-roberta-base.2023-05-15_16-38-00_817632663.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/outputs/2023-05-15_16-38-25_412559758/models/xlm-roberta-base.2023-05-15_16-38-25_412559758.bk", # random
    ],
    "bengali": [
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_18-51-51_530781221/models/xlm-roberta-base.2023-05-17_18-51-51_530781221.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-16_01-06-34_7716341410/models/xlm-roberta-base.2023-05-16_01-06-34_7716341410.bk", # middle 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_18-38-22_9707531518/models/xlm-roberta-base.2023-05-17_18-38-22_9707531518.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_21-31-30_2754508852/models/xlm-roberta-base.2023-05-17_21-31-30_2754508852.bk", # random
    ],
    "gujarati": [
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-16_16-51-10_0113752266/models/xlm-roberta-base.2023-05-16_16-51-10_0113752266.bk", # init-layers-1 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_14-34-16_9715462024/models/xlm-roberta-base.2023-05-17_14-34-16_9715462024.bk", # middle 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_14-34-16_9742956449/models/xlm-roberta-base.2023-05-17_14-34-16_9742956449.bk", # last-layer 
      "/home/anonymous-xme/mend/mend/him-param-outputs/outputs/2023-05-17_11-42-31_3743647869/models/xlm-roberta-base.2023-05-17_11-42-31_3743647869.bk", # random
    ],

}
ALGO = "ft" # or mend
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
