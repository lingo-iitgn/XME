import os
import argparse
from joblib import Parallel, delayed

####################### Configuration #######################
MODELS = ["bloom-560m", "mbert-uncased", "xlm-roberta"]
LANGS = ["english", "spanish", "french", "hindi", "bengali", "gujarati", "malayalam", "tamil", "kannada"]
LAYERS = ["init-layers-1", "middle", "last-layer", "random", "full"]
ALGS = ["mend", "efk", "enn", "ft"]
#############################################################


parser = argparse.ArgumentParser(description="Causal Tracing")

def aa(*args, **kwargs):
    parser.add_argument(*args, **kwargs)

aa("--alg", type=str, default="mend", choices=ALGS, help="Algorithm to run")
aa("--model", type=str, default="bloom-560m", choices=MODELS, help="Base Model")
aa("--layers", type=str, default="1", help="Layers to run; Enter in space separated format. 0: Init; 1: Middle; 2: Last; 3: Random; 4: Full")
aa("--lang", type=str, default="english", choices=LANGS, help="Language to run")
aa("--experiment", type=str, default="fc", help="Experiment to run")
aa("--tests", type=bool, default=False, help="Run tests")
aa("--eval_only", type=bool, default=False, help="Evaluate only")
aa("--train", type=bool, default=False, help="Train only")
aa("--logs_dir", type=str, default="logs", help="Logs directory")
aa("--cuda", type=str, default="0", help="CUDA device")
args = parser.parse_args()

algs = args.alg
model = args.model
layers = [LAYERS[x] for x in args.layers.split(" ")]
lang = args.lang
experiment = args.experiment
tests = args.tests
eval_only = args.eval_only
train = args.train
logs_dir = args.logs_dir
cuda = args.cuda

def run_mend(layer, cuda_alloc):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_alloc
    os.system(f"python -m run +alg={algs} +experiment={experiment} +model={model}-{lang}-{layer} +tests={tests} +lang={lang} ++eval_only={eval_only} ++train={train} | tee {logs_dir}/{algs}-{model}-{lang}-{layer}.txt")
    

# Parallelize
cudas = cuda.split(",")
parallel_pool = Parallel(n_jobs=len(cudas)) # Number of GPUs
funs = []
if len(cudas) == len(layers): # Fully parallel
    for i in range(len(cudas)):
        funs.append(delayed(run_mend)(layers[i], cudas[i]))
        # run_mend(layers[i], cudas[i])
else: # Load balance
    count = 0
    for cuda in cudas:
        funs.append(delayed(run_mend)(layers[count], cuda))
        # run_mend(layers[count], cuda)
        count += 1

    if count < len(layers):
        for cuda in cudas:
            funs.append(delayed(run_mend)(layers[count], cuda))
            # run_mend(layers[count], cuda)
            count += 1

parallel_pool(funs)
