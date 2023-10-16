import os
import time

#########################################################################
CUDA = "3"
MODEL = "bloom-560m-french-last-layer"
MLP_MODEL = "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-last-layers/models/bloom-560m.2023-04-18_09-10-01_6896549296.bk"
run_name = "exp_1_finetuned_french_last_layer_pred"
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
        os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg=mend +experiment=fc +model={MODEL} ++eval_only=True ++train=False +archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/mend-bloom-560m-{l1}-{l2}-{MODEL}.txt""")
