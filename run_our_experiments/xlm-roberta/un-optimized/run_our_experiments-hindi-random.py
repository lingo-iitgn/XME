import os
import time

#########################################################################
CUDA = "3"
MODEL = "xlm-roberta-hindi-random"
MLP_MODEL = "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-xlm-roberta-hindi-random/models/xlm-roberta-base.2023-05-10_07-54-52_0537483555.bk"
run_name = "exp_1_xlm_finetuned_hindi_random_layers"
#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA



file_names = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
for l1 in file_names:
    isExist = os.path.exists(f"./logs/{run_name}/{l1}")
    if not isExist:
        os.makedirs(f"./logs/{run_name}/{l1}")

time.sleep(3) # Wait for os process to complete

for l1 in file_names:
    for l2 in file_names:
        print("="*40, l1, l2, "="*40)
        os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg=mend +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/mend-xlm-{l1}-{l2}-{MODEL}.txt""")
