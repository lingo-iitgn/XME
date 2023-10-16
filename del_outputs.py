import os
import re
from tqdm import tqdm
##############################################################################################
DEL_DIR = "/home/anonymous-xme/mend/mend/outputs"
LOGS_DIR = "/home/anonymous-xme/mend/mend/logs"
##############################################################################################

reg = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}_([0-9]|-)*_[0-9]*')

txt_files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".txt")]
txt_files_out = set()
print("Gathering all the output folders...")
for f in tqdm(txt_files):
    with open(os.path.join(LOGS_DIR, f)) as file:
        lines = file.readlines()
        for line in lines:
            x = line.find("Saving model to ")
            if x != -1:
                name = reg.search(line[x+17:-1]).group()
                txt_files_out.add(name)              
                break


all_folders = set([f for f in os.listdir(DEL_DIR) if os.path.isdir(os.path.join(DEL_DIR, f))])
del_folders = all_folders - txt_files_out

print("Total folders: ", len(all_folders))
print("Folders to delete: ", len(del_folders))
print("Folders to keep: ", all_folders-del_folders)

# for f in del_folders:
#     print("Deleting: ", f)
#     os.system("rm -rf " + os.path.join(DEL_DIR, f))