import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.pretty import pprint

parser = argparse.ArgumentParser()
# parser.add_argument("-l", "--lang", help = "Language")
parser.add_argument("-a", "--algo", help = "Algorithm")
parser.add_argument("-t", "--type", help = "Lang or inverse", default="lang")
# parser.add_argument("-m", "--model", help = "Model")

args = parser.parse_args()
# ALGO = args.algo
TYPE = args.type
# LANG = args.lang
# MODEL = args.model
# assert ALGO in ["mend", "ke", "ft"], "Algorithm must be one of mend, ke, ft"
# assert LANG in ["en", "hi", "es", "fr", "bn", "gu", "mixed", "inverse"], "Lang not in list of languages"
# assert MODEL in ["bloom", "mbert", "xlm"], "Model not found"


#############################################################
# ES_IN_DIR = "es.txt"
# LOC_IN_DIR = "loc.txt"
# OUT_DIR = "out.txt"

ALGOS = ["mend", "ke", "ft"]
MODELS = ["bloom", "mbert", "xlm"]
# if MODEL == "mbert":
#     files = ["en", "fr", "es",  "hi", "gu", "bn", "mix"]
# else:
#     files = ["en", "fr", "es",  "hi", "gu", "bn", "mix", "inv"]

colors_es = [(255, 255, 255), (130, 148, 196)] # white, light blue 
colors_loc = [(20, 108, 148),(255, 255, 255)] # white, light blue 

SKIP_ROW = 7 # Skips every 7th row

# SET_NAME = ["Initial", "Middle", "Last", "Random"]
SET_NAME = ["I", "M", "L", "R"]
MODEL_NAMES = {"bloom": "bloom-560m", "mbert": "mBERT", "xlm": "XLM-RoBERTa"}


in_order = ["en", "hi", "es", "fr", "bn", "gu"] # Input order
langs = out_order = ["en", "es", "fr", "hi","gu", "bn"] # Output order
#############################################################

def reorder_matrix(vals):
    # Expects vals to in order "en", "hi", "es", "fr", "bn", "gu"
    # Order of output must be mentioned in list below

    cell_vals = {}
    for r, row in enumerate(vals):
        for c, col in enumerate(row):
            cell_vals[(in_order[r], in_order[c])] = col

    # pprint(cell_vals)

    new_vals = []
    for r in range(len(out_order)):
        new_row = []
        for c in range(len(out_order)):
            new_row.append(cell_vals[(out_order[r], out_order[c])])
        new_vals.append(new_row)

    return new_vals

def to_matrix(file_read):
    # Read all values to list from in.txt
    with open(file_read, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content if x.strip() != ""]
    content_set = []
    for row in range(0, len(content), SKIP_ROW):
            content_set.append(content[row+1: row + SKIP_ROW])

    def mp(x):
        return float(x.strip())

    decimal_content_set = []
    for idx, sets in enumerate(content_set):
        data = []
        for row in sets:
            r = list(map(mp, row.split("\t")))
            assert len(r) == len(out_order), f"Number of columns in row is not equal to number of languages in {file_read} for {SET_NAME[idx]}"
            data.append(r)
        data = reorder_matrix(data) # Reorder to match output order
        decimal_content_set.append(data)

    return decimal_content_set


# def fine_mono(mat_set):
#     row_vals = []
#     n = len(mat_set)
#     for i in range(len(mat_set[0])):
#         avg = 0.0
#         for st in mat_set:
#             avg += st[i][i]
#         row_vals.append(avg/n)
#     return row_vals
        

# def which_lang(mat_set):
#     row_vals = []
#     set_avg_val = []
#     for st in mat_set:
#         temp = []
#         for row in st:
#             temp.append(sum(row)/len(row)) 
#         set_avg_val.append(temp)

#     # print(len(set_avg_val))

#     for c in range(len(set_avg_val[0])):
#         avg = 0.0      
#         for r in range(len(set_avg_val)):
#             avg += set_avg_val[r][c]
#         row_vals.append(avg/len(set_avg_val))

#     return row_vals


def f1_score(es_set, loc_set):

    def f1(es, loc):
        if es == 0 and loc == 0:
            return 0
        return 2*es*loc/(es+loc)

    combined_set = []
    for left, right in zip(es_set, loc_set):
        assert len(left) == len(right), "Length of left and right is not equal"
        mat = []
        for i in range(len(left)):
            temp = []
            for j in range(len(left[i])):
                temp.append(f1(left[i][j], right[i][j]))
            mat.append(temp)
        combined_set.append(mat)

    avg = 0.0
    count = 0
    for mat in combined_set:
        for row in mat:
            avg += sum(row)
            count += len(row)
    avg /= count
    return avg


# def heatmap(x_data, name="out", axis=0, cbar=True, cbar_ax=None, xticklabels=True, yticklabels=True, title="", dir="horizontal"):
    # row_name = out_order if dir == "horizontal" else ["mixed"] + (["inverse"] if not title == "mBERT" else [])
    # col_name = out_order

    # for i in range(len(x_data)):
    #     for j in range(len(x_data[i])):
    #         x_data[i][j] = round(x_data[i][j]*100, 4)

    # l = []
    # for i in range(len(x_data)):
    #     d = {"evl": row_name[i]}   
    #     for j in range(len(x_data[i])):
    #         d[col_name[j]] = x_data[i][j]
    #     l.append(d)

    # df = pd.DataFrame(l)
    # # pprint(df)

#     tab = df.iloc[:, 1:]
#     tab_n = tab.div(tab.max(axis=1), axis=0)

#     if dir == "horizontal":
#         # Create heatmap with seaborn library
#         sns.set(font_scale=1.4)
#         # Change yticklabel size to 20
#         # plt.yticks(fontsize=20)
#         g = sns.heatmap(tab_n, xticklabels=col_name, annot=True, cmap="Blues", fmt=".2f", ax=axis, cbar=cbar, cbar_ax=cbar_ax if cbar else None)
#         if yticklabels:
#             g.set_yticklabels(row_name, fontsize=16)
#             axis.set_ylabel("Fine-tuning Language", fontsize=20)
#         else:
#             g.set_yticklabels([])
#             g.tick_params(left=False)
#         g.set_xticklabels(col_name, fontsize=16)
#         axis.set_title(title, fontsize=20)
#         if not yticklabels and not cbar: # center image 
#             axis.set_xlabel("Editing Language", fontsize=20)

#         if cbar:
#             cbar = g.collections[0].colorbar
#             cbar.ax.tick_params(labelsize=18)
#             # cbar.ax.set_ylabel(rotation=90, fontsize=20)

#     else:
#          # Create heatmap with seaborn library
#         sns.set(font_scale=1.4)
#         # Change yticklabel size to 20
#         # plt.yticks(fontsize=20)
#         g = sns.heatmap(tab_n, annot=True, cmap="Blues", fmt=".2f", ax=axis, cbar=cbar, cbar_ax=cbar_ax if cbar else None, cbar_kws={"orientation": "horizontal"})
#         if xticklabels:
#             g.set_xticklabels(col_name, fontsize=16)
#             axis.set_xlabel("Editing Language", fontsize=16)
#         else:
#             g.set_xticklabels([])
#             g.tick_params(bottom=False)
        
#         axis.set_title(title, fontsize=16)
#         g.set_yticklabels(row_name, fontsize=16, rotation=0)
#         if title == "mBERT": # center image 
#             axis.set_ylabel("Fine-tuning Language", fontsize=20, labelpad=20)

#         if cbar:
#             cbar = g.collections[0].colorbar
#             cbar.ax.tick_params(labelsize=16)
            
    
    # # Save sns figure as pdf file
    # plt.savefig(f"./out/{name}/{ALGO}-{MODEL}-{name}.pdf")
    # plt.clf()

def heatmap(x_data):
    col_name = out_order
    algo_name = ["MEND", "KE", "FT"]
    row_name = ["bloom-560M", "mBERT", "XLM-RoBERTa"] * len(algo_name)

    for i in range(len(x_data)):
        for j in range(len(x_data[i])):
            x_data[i][j] = round(x_data[i][j]*100, 4)

    l = []
    for i in range(len(x_data)):
        d = {"evl": row_name[i]}   
        for j in range(len(x_data[i])):
            d[col_name[j]] = x_data[i][j]
        l.append(d)

    df = pd.DataFrame(l)
    # pprint(df)

    sns.heatmap(df.iloc[:, 1:],annot=True, cmap="Blues", fmt=".2f", xticklabels=col_name, yticklabels=row_name)

    # Save sns figure as pdf file
    plt.savefig(f"./out/test.pdf")

if TYPE == "lang":
    # fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={"width_ratios": [3.5, 3.5, 3.5, 0.10]}, constrained_layout=True) # Three axis for three models and one for cbar

    mat = []
    for ALGO in ALGOS:
        for model in MODELS:
            # fm_mat = []
            r = []
            for lang in langs:
                es = to_matrix(f"./es/in-{ALGO}-{model}/" + lang + ".txt")
                loc = to_matrix(f"./loc/in-{ALGO}-{model}/" + lang + ".txt")
                if lang not in ["en", "hi"]: 
                        # Init, Mid, Last, Random
                        es = [es[0], es[2], es[1], es[3]]
                        loc = [loc[0], loc[2], loc[1], loc[3]]


                r.append(f1_score(es, loc))

                # # fm = fine_mono(es)
                # wl = which_lang(es)
                
                # # fm_mat.append(fm)
                # wl_mat.append(wl)

            mat.append(r)
    
        # heatmap(wl_mat, name="wl", axis=ax, cbar=(model == "xlm"), cbar_ax=axes[-1], yticklabels=(model == "bloom"), title=MODEL_NAMES[model], dir="horizontal")
    
    heatmap(mat)
    # fig.savefig(f"./out/{ALGO}-f1.pdf")


else:
    pass
    # fig, axes = plt.subplots(4, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1, 2, 0.10]}, constrained_layout=True) # 
    # for ax, model in zip(axes[:-1], MODELS):
    #     wl_mat = []
    #     for lang in ["mix", "inv"]:
    #         if model == "mbert" and lang == "inv": continue

    #         es = to_matrix(f"./in-{ALGO}-{model}/" + lang + ".txt")

    #         # # fm = fine_mono(es)
    #         # wl = which_lang(es)
            
    #         # # fm_mat.append(fm)
    #         # wl_mat.append(wl)

    #     heatmap(wl_mat, name="wl", axis=ax, cbar=(model == "xlm"), cbar_ax=axes[-1], xticklabels=(model == "xlm"), title=MODEL_NAMES[model], dir="vertical")
    # fig.savefig(f"./out/{ALGO}-wl-mi.pdf")

