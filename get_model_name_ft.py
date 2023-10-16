import sys
import os

import argparse
 

log_dir = "logs"
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-a", "--alg", help = "Algorithms")
parser.add_argument("-m", "--model", help = "Base Model")
parser.add_argument("-l", "--lang", help = "Finetuned Language")
 
# Read arguments from command line
args = parser.parse_args()

assert args.lang in ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"], 'Language not supported'
assert args.model in ["bloom-560m", "mbert-uncased", "xlm-roberta"], 'Model not supported'
assert args.alg in ["mend", "efk", "ft"], 'Algorithm not supported'

# list the txt files in the directory
suffixes = ["init-1", "middle", "last", "random"]

fine_tuned_model_name = []

# Check if file exists
for suffix in suffixes:
    file_name = f"{args.alg}-{args.model}-{args.lang}-{suffix}.txt"
    if os.path.isfile(f"./{log_dir}/{file_name}"):
        
        with open(f"./{log_dir}/{file_name}", "r") as f:
            # Read Each line and check if substring exists in line
            for line in f:
                if "Saving model to" in line:
                    home_idx = line.find("/home")
                    fl_name = line[home_idx:].strip() + ".bk"
                    if os.path.isfile(fl_name):
                        mn = f""" \"{fl_name}\", # {suffix} """
                    else:
                            mn = f""" \"Missing bk #{suffix} \", """
                    break
    else:
        mn = f""" \"Missing LOG #{suffix} \", """
    fine_tuned_model_name.append(mn)

for mn in fine_tuned_model_name:
    print(mn)
