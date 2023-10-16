##################################################################################################

server = "anonymous-xme@lingo-lexico.iitgn.ac.in"
port = "2020"
paste_path = "."

names = {
    "english": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-init-layers-1-pred/models/bloom-560m.2023-04-17_14-15-16_7851742737.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-middle-layers-pred/models/bloom-560m.2023-04-17_22-18-54_5163214677.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-last-layers-pred/models/bloom-560m.2023-04-17_22-21-38_5418723295.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-english-random-layers-pred/models/bloom-560m.2023-04-17_22-25-12_9476916002.bk", # random
    ],
    "hindi": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-15-45_5272864186/models/bloom-560m.2023-05-22_18-15-45_5272864186.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-17-06_131347881/models/bloom-560m.2023-05-22_18-17-06_131347881.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-22_18-18-28_5631311707/models/bloom-560m.2023-05-22_18-18-28_5631311707.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-28_20-31-50_2907878295/models/bloom-560m.2023-05-28_20-31-50_2907878295.bk", # random
    ],
    "spanish": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-init-layers-1/models/bloom-560m.2023-05-07_16-46-04_0342669267.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-middle/models/bloom-560m.2023-05-07_21-39-38_1634424205.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-last/models/bloom-560m.2023-05-08_00-50-00_398157700.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-spanish-random/models/bloom-560m.2023-05-08_08-54-12_3978277641.bk", # random
    ],
    "french": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-init-layers-1-pred/models/bloom-560m.2023-04-18_00-20-48_8370089281.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-middle-layers/models/bloom-560m.2023-04-18_09-09-17_4676312842.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-last-layers/models/bloom-560m.2023-04-18_09-10-01_6896549296.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-french-random/models/bloom-560m.2023-04-18_09-10-44_6154188477.bk", # random
    ],
    "bengali": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-init-layers-1-pred/models/bloom-560m.2023-04-19_07-47-49_4737446760.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-middle-pred/models/bloom-560m.2023-04-20_01-31-18_6374192842.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-last-layer-pred/models/bloom-560m.2023-04-19_21-09-57_7940862913.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-bengali-random-pred/models/bloom-560m.2023-04-19_23-47-19_7805968414.bk", # random
    ],
    "gujarati": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-init-layers-1/models/bloom-560m.2023-04-22_13-56-25_4236513828.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-middle/models/bloom-560m.2023-04-23_11-37-22_3909366631.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-last-layer/models/bloom-560m.2023-04-22_23-10-46_2170927579.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-gujarati-random/models/bloom-560m.2023-04-23_00-27-13_7780401534.bk", # random
    ],
        "mixed": [
 "/home/anonymous-xme/mend/mend/outputs/2023-05-27_16-24-15_0407797334/models/bloom-560m.2023-05-27_16-24-15_0407797334.bk", # init-1 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-06_7141775050/models/bloom-560m.2023-05-27_18-16-06_7141775050.bk", # middle 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-27_18-16-22_4494132059/models/bloom-560m.2023-05-27_18-16-22_4494132059.bk", # last 
 "/home/anonymous-xme/mend/mend/outputs/2023-05-27_22-55-43_7215556499/models/bloom-560m.2023-05-27_22-55-43_7215556499.bk", # random 
    ],
        "inverse": [
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-init/models/bloom-560m.2023-05-05_08-47-43_7150709473.bk", # init-1 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-mid/models/bloom-560m.2023-05-05_08-50-27_6644632166.bk", # middle 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-last/models/bloom-560m.2023-05-05_18-42-12_02719454.bk", # last 
 "/home/anonymous-xme/mend/mend/data/fever/mend_models/mend-bloom-560m-inverse-random/models/bloom-560m.2023-05-05_21-57-28_4566237195.bk", # random
    ]
}

############################################################################################################################

files = []
for k, v in names.items():
    files.extend(v)

for f in files:
    print(f"cp -r {f} /home/anonymous-xme/mend/mend/run_our_experiments/temp")
