import os
import time

#########################################################################
CUDA = "2"
MODEL = "bloom-560m-english-init-layers-1"
MLP_MODEL = "/home/anonymous-xme/mend/mend/outputs/2023-05-15_20-49-24_5071488770/models/bloom-560m.2023-05-15_20-49-24_5071488770.bk"
run_name = "exp_1_finetuned_english_init_layers_1_pred"
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
        os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg=efk +experiment=fc +model={MODEL} ++eval_only=True ++train=False +archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/efk-bloom-560m-{l1}-{l2}-{MODEL}.txt""")
