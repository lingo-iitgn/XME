import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="bloom-560m", help="bloom-560m or mbert-uncased or xlm-roberta")
parser.add_argument("--algo", type=str, default="ft", help="ft or mend")
parser.add_argument("--lang", type=str, help="Languages")
parser.add_argument("--file-name", type=bool, default=False, help="Print file name")

args = parser.parse_args()
MODEL_NAME = args.model_name
ALGO = args.algo
LANG = args.lang

#################################
LOGS_DIR = "/home/anonymous-xme/mend/mend/logs"
#################################

file_prefix = f"{ALGO}-{MODEL_NAME}-{LANG}"

# List files in the logs directory
req = []
files = os.listdir(LOGS_DIR)
for file in files:
    if file.startswith(file_prefix) and file.endswith(".txt"):
        req.append(file) 

# Layer set
file_set = []
layers = ["init-1", "middle", "last", "random"]
for layer in layers:
    for file in req:
        if layer in file:
            file_set.append(file)


reg = r"Saving\smodel\sto\s([A-Za-z0-9]|/|-|_|/|\.)*"

for file in file_set:
    if args.file_name:
        print(file)
    with open(os.path.join(LOGS_DIR, file), "r") as f:
        lines = f.readlines()
        for line in lines:
            m = re.search(reg, line)
            if m:
                m = m.group(0)
                idx = m.find("/home")
                print(f"\"{m[idx:]}\",")
                break
        