import os
import time

#########################################################################
CUDA = "0"
MODEL = "xlm-roberta-english-middle"
MLP_MODEL = "/home/anonymous-xme/mend/mend/outputs/2023-05-12_11-49-19_9373316829/models/xlm-roberta-base.2023-05-12_11-49-19_9373316829.bk"
run_name = "exp_1_xlm_finetuned_english_middle_layers"
#########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

alg_edit = "efk"

file_names = ["english", "hindi", "spanish", "french", "bengali", "gujarati"]
for l1 in file_names:
    isExist = os.path.exists(f"./logs/{run_name}/{l1}")
    if not isExist:
        os.makedirs(f"./logs/{run_name}/{l1}")

time.sleep(3) # Wait for os process to complete

for l1 in file_names:
    for l2 in file_names:
        print("="*40, l1, l2, "="*40)
        os.system(f"""CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg={alg_edit} +experiment=fc +model={MODEL} ++archive={MLP_MODEL} +train_set="fever/old-dataset/fever_train_1200 - english_1200.jsonl" +val_set="fever/experiment_1/{l1}/fever_dev_1200 - {l1}-{l2}_scripted.jsonl" +tests=True  | tee logs/{run_name}/{l1}/{alg_edit}-xlm-{l1}-{l2}-{MODEL}.txt""")
