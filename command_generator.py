
#######################################################################
model = "xlm-roberta"
CUDA = 3
#######################################################################



langs = ["english", "hindi", "spanish", "french", "bengali", "gujarati", "mixed", "inverse"]
model_suffix = ["init-layers-1", "middle", "last-layer", "random"]

for lang in langs:
    print(f"#{model} - {lang}")
    for suffix in model_suffix:
        print (f"CUDA_VISIBLE_DEVICES={CUDA} python -m run +alg=mend +experiment=fc +model={model}-{lang}-{suffix} +tests=False +lang={lang} ++eval_only=False ++train=True | tee logs/mend-{model}-{lang}-{suffix}.txt")
    print()
    for suffix in model_suffix:
        print(f"python run_our_experiments/{model}/run_our_experiments-{lang}-{suffix}.py ")
    print()
