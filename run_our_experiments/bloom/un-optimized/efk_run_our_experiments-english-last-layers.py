import os
import time

#########################################################################
CUDA = "0"
MODEL = "bloom-560m-english-last-layer"
MLP_MODEL = "/home/anonymous-xme/mend/mend/outputs/2023-05-16_10-33-21_0185677885/models/bloom-560m.2023-05-16_10-33-21_0185677885.bk"
run_name = "exp_1_finetuned_english_last_layers_1_pred"
#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

alg_to_Edit = "efk"

file_names = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
for l1 in file_names:
    isExist = os.path.exists(f"./logs/{run_name}/{l1}")
    if not isExist:
        os.makedirs(f"./logs/{run_name}/{l1}")

time.sleep(3) # Wait for os process to complete

for l1 in file_names:
    for l2 in file_names:
        print("="*40, l1, l2, "="*40)
        os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={alg_to_Edit} +experiment=fc +model={MODEL} ++eval_only=True ++train=False +archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/{alg_to_Edit}-bloom-560m-{l1}-{l2}-{MODEL}.txt""")
