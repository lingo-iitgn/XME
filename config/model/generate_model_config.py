import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mbert-uncased")
parser.add_argument("--lang", type=str, default="tamil")
parser.add_argument("--base-model", type=str, default="mbert-uncased")

args = parser.parse_args()

MODEL_SET = {
    "mbert-uncased": { #################################################### mBERT ######################################################
        "name": "bert-base-multilingual-uncased",
        "class_name": "AutoModelForSequenceClassification",
        "tokenizer_class": "AutoTokenizer",
        "tokenizer_name": "bert-base-multilingual-uncased",
        "sets": {

"init-layers-1": """
- bert.encoder.layer.0.intermediate.dense.weight
- bert.encoder.layer.0.output.dense.weight
- bert.encoder.layer.1.intermediate.dense.weight
- bert.encoder.layer.1.output.dense.weight
- bert.encoder.layer.2.intermediate.dense.weight
- bert.encoder.layer.2.output.dense.weight
- bert.encoder.layer.3.intermediate.dense.weight
- bert.encoder.layer.3.output.dense.weight
""",

"middle": """
- bert.encoder.layer.4.intermediate.dense.weight
- bert.encoder.layer.4.output.dense.weight
- bert.encoder.layer.5.intermediate.dense.weight
- bert.encoder.layer.5.output.dense.weight
- bert.encoder.layer.6.intermediate.dense.weight
- bert.encoder.layer.6.output.dense.weight
- bert.encoder.layer.7.intermediate.dense.weight
- bert.encoder.layer.7.output.dense.weight
""",

"last-layer": """
- bert.encoder.layer.8.intermediate.dense.weight
- bert.encoder.layer.8.output.dense.weight
- bert.encoder.layer.9.intermediate.dense.weight
- bert.encoder.layer.9.output.dense.weight
- bert.encoder.layer.10.intermediate.dense.weight
- bert.encoder.layer.10.output.dense.weight
- bert.encoder.layer.11.intermediate.dense.weight
- bert.encoder.layer.11.output.dense.weight
""",

"random": """
- bert.encoder.layer.2.intermediate.dense.weight
- bert.encoder.layer.2.output.dense.weight
- bert.encoder.layer.4.intermediate.dense.weight
- bert.encoder.layer.4.output.dense.weight
- bert.encoder.layer.6.intermediate.dense.weight
- bert.encoder.layer.6.output.dense.weight
- bert.encoder.layer.9.intermediate.dense.weight
- bert.encoder.layer.9.output.dense.weight
""",

"full": """
- bert.encoder.layer.0.attention.self.query.weight
- bert.encoder.layer.0.attention.self.key.weight
- bert.encoder.layer.0.attention.self.value.weight
- bert.encoder.layer.0.attention.output.dense.weight
- bert.encoder.layer.0.intermediate.dense.weight
- bert.encoder.layer.0.output.dense.weight
- bert.encoder.layer.1.attention.self.query.weight
- bert.encoder.layer.1.attention.self.key.weight
- bert.encoder.layer.1.attention.self.value.weight
- bert.encoder.layer.1.attention.output.dense.weight
- bert.encoder.layer.1.intermediate.dense.weight
- bert.encoder.layer.1.output.dense.weight
- bert.encoder.layer.2.attention.self.query.weight
- bert.encoder.layer.2.attention.self.key.weight
- bert.encoder.layer.2.attention.self.value.weight
- bert.encoder.layer.2.attention.output.dense.weight
- bert.encoder.layer.2.intermediate.dense.weight
- bert.encoder.layer.2.output.dense.weight
- bert.encoder.layer.3.attention.self.query.weight
- bert.encoder.layer.3.attention.self.key.weight
- bert.encoder.layer.3.attention.self.value.weight
- bert.encoder.layer.3.attention.output.dense.weight
- bert.encoder.layer.3.intermediate.dense.weight
- bert.encoder.layer.3.output.dense.weight
- bert.encoder.layer.4.attention.self.query.weight
- bert.encoder.layer.4.attention.self.key.weight
- bert.encoder.layer.4.attention.self.value.weight
- bert.encoder.layer.4.attention.output.dense.weight
- bert.encoder.layer.4.intermediate.dense.weight
- bert.encoder.layer.4.output.dense.weight
- bert.encoder.layer.5.attention.self.query.weight
- bert.encoder.layer.5.attention.self.key.weight
- bert.encoder.layer.5.attention.self.value.weight
- bert.encoder.layer.5.attention.output.dense.weight
- bert.encoder.layer.5.intermediate.dense.weight
- bert.encoder.layer.5.output.dense.weight
- bert.encoder.layer.6.attention.self.query.weight
- bert.encoder.layer.6.attention.self.key.weight
- bert.encoder.layer.6.attention.self.value.weight
- bert.encoder.layer.6.attention.output.dense.weight
- bert.encoder.layer.6.intermediate.dense.weight
- bert.encoder.layer.6.output.dense.weight
- bert.encoder.layer.7.attention.self.query.weight
- bert.encoder.layer.7.attention.self.key.weight
- bert.encoder.layer.7.attention.self.value.weight
- bert.encoder.layer.7.attention.output.dense.weight
- bert.encoder.layer.7.intermediate.dense.weight
- bert.encoder.layer.7.output.dense.weight
- bert.encoder.layer.8.attention.self.query.weight
- bert.encoder.layer.8.attention.self.key.weight
- bert.encoder.layer.8.attention.self.value.weight
- bert.encoder.layer.8.attention.output.dense.weight
- bert.encoder.layer.8.intermediate.dense.weight
- bert.encoder.layer.8.output.dense.weight
- bert.encoder.layer.9.attention.self.query.weight
- bert.encoder.layer.9.attention.self.key.weight
- bert.encoder.layer.9.attention.self.value.weight
- bert.encoder.layer.9.attention.output.dense.weight
- bert.encoder.layer.9.intermediate.dense.weight
- bert.encoder.layer.9.output.dense.weight
- bert.encoder.layer.10.attention.self.query.weight
- bert.encoder.layer.10.attention.self.key.weight
- bert.encoder.layer.10.attention.self.value.weight
- bert.encoder.layer.10.attention.output.dense.weight
- bert.encoder.layer.10.intermediate.dense.weight
- bert.encoder.layer.10.output.dense.weight
- bert.encoder.layer.11.attention.self.query.weight
- bert.encoder.layer.11.attention.self.key.weight
- bert.encoder.layer.11.attention.self.value.weight
- bert.encoder.layer.11.attention.output.dense.weight
- bert.encoder.layer.11.intermediate.dense.weight
- bert.encoder.layer.11.output.dense.weight
"""
}
    }, #################################################### BLOOM ######################################################
    "bloom-560m": {
        "name": "bigscience/bloom-560m",
        "class_name": "BloomForSequenceClassification",
        "tokenizer_class": "BloomTokenizerFast",
        "tokenizer_name": "bigscience/bloom-560m",
        "sets": {

"init-layers-1": """
- transformer.h.0.mlp.dense_h_to_4h.weight
- transformer.h.0.mlp.dense_4h_to_h.weight
- transformer.h.1.mlp.dense_h_to_4h.weight
- transformer.h.1.mlp.dense_4h_to_h.weight
- transformer.h.2.mlp.dense_h_to_4h.weight
- transformer.h.2.mlp.dense_4h_to_h.weight
- transformer.h.3.mlp.dense_h_to_4h.weight
- transformer.h.3.mlp.dense_4h_to_h.weight
""",

"middle": """
- transformer.h.14.mlp.dense_h_to_4h.weight
- transformer.h.14.mlp.dense_4h_to_h.weight
- transformer.h.15.mlp.dense_h_to_4h.weight
- transformer.h.15.mlp.dense_4h_to_h.weight
- transformer.h.16.mlp.dense_h_to_4h.weight
- transformer.h.16.mlp.dense_4h_to_h.weight
- transformer.h.17.mlp.dense_h_to_4h.weight
- transformer.h.17.mlp.dense_4h_to_h.weight
""",

"last-layer": """
- transformer.h.20.mlp.dense_h_to_4h.weight
- transformer.h.20.mlp.dense_4h_to_h.weight
- transformer.h.21.mlp.dense_h_to_4h.weight
- transformer.h.21.mlp.dense_4h_to_h.weight
- transformer.h.22.mlp.dense_h_to_4h.weight
- transformer.h.22.mlp.dense_4h_to_h.weight
- transformer.h.23.mlp.dense_h_to_4h.weight
- transformer.h.23.mlp.dense_4h_to_h.weight
""",

"random": """
- transformer.h.8.mlp.dense_h_to_4h.weight
- transformer.h.8.mlp.dense_4h_to_h.weight
- transformer.h.13.mlp.dense_h_to_4h.weight
- transformer.h.13.mlp.dense_4h_to_h.weight
- transformer.h.17.mlp.dense_h_to_4h.weight
- transformer.h.17.mlp.dense_4h_to_h.weight
- transformer.h.21.mlp.dense_h_to_4h.weight
- transformer.h.21.mlp.dense_4h_to_h.weight
""",

"full": """
- transformer.h.0.self_attention.query_key_value.weight
- transformer.h.0.self_attention.dense.weight
- transformer.h.0.mlp.dense_h_to_4h.weight
- transformer.h.0.mlp.dense_4h_to_h.weight
- transformer.h.1.self_attention.query_key_value.weight
- transformer.h.1.self_attention.dense.weight
- transformer.h.1.mlp.dense_h_to_4h.weight
- transformer.h.1.mlp.dense_4h_to_h.weight
- transformer.h.2.self_attention.query_key_value.weight
- transformer.h.2.self_attention.dense.weight
- transformer.h.2.mlp.dense_h_to_4h.weight
- transformer.h.2.mlp.dense_4h_to_h.weight
- transformer.h.3.self_attention.query_key_value.weight
- transformer.h.3.self_attention.dense.weight
- transformer.h.3.mlp.dense_h_to_4h.weight
- transformer.h.3.mlp.dense_4h_to_h.weight
- transformer.h.4.self_attention.query_key_value.weight
- transformer.h.4.self_attention.dense.weight
- transformer.h.4.mlp.dense_h_to_4h.weight
- transformer.h.4.mlp.dense_4h_to_h.weight
- transformer.h.5.self_attention.query_key_value.weight
- transformer.h.5.self_attention.dense.weight
- transformer.h.5.mlp.dense_h_to_4h.weight
- transformer.h.5.mlp.dense_4h_to_h.weight
- transformer.h.6.self_attention.query_key_value.weight
- transformer.h.6.self_attention.dense.weight
- transformer.h.6.mlp.dense_h_to_4h.weight
- transformer.h.6.mlp.dense_4h_to_h.weight
- transformer.h.7.self_attention.query_key_value.weight
- transformer.h.7.self_attention.dense.weight
- transformer.h.7.mlp.dense_h_to_4h.weight
- transformer.h.7.mlp.dense_4h_to_h.weight
- transformer.h.8.self_attention.query_key_value.weight
- transformer.h.8.self_attention.dense.weight
- transformer.h.8.mlp.dense_h_to_4h.weight
- transformer.h.8.mlp.dense_4h_to_h.weight
- transformer.h.9.self_attention.query_key_value.weight
- transformer.h.9.self_attention.dense.weight
- transformer.h.9.mlp.dense_h_to_4h.weight
- transformer.h.9.mlp.dense_4h_to_h.weight
- transformer.h.10.self_attention.query_key_value.weight
- transformer.h.10.self_attention.dense.weight
- transformer.h.10.mlp.dense_h_to_4h.weight
- transformer.h.10.mlp.dense_4h_to_h.weight
- transformer.h.11.self_attention.query_key_value.weight
- transformer.h.11.self_attention.dense.weight
- transformer.h.11.mlp.dense_h_to_4h.weight
- transformer.h.11.mlp.dense_4h_to_h.weight
- transformer.h.12.self_attention.query_key_value.weight
- transformer.h.12.self_attention.dense.weight
- transformer.h.12.mlp.dense_h_to_4h.weight
- transformer.h.12.mlp.dense_4h_to_h.weight
- transformer.h.13.self_attention.query_key_value.weight
- transformer.h.13.self_attention.dense.weight
- transformer.h.13.mlp.dense_h_to_4h.weight
- transformer.h.13.mlp.dense_4h_to_h.weight
- transformer.h.14.self_attention.query_key_value.weight
- transformer.h.14.self_attention.dense.weight
- transformer.h.14.mlp.dense_h_to_4h.weight
- transformer.h.14.mlp.dense_4h_to_h.weight
- transformer.h.15.self_attention.query_key_value.weight
- transformer.h.15.self_attention.dense.weight
- transformer.h.15.mlp.dense_h_to_4h.weight
- transformer.h.15.mlp.dense_4h_to_h.weight
- transformer.h.16.self_attention.query_key_value.weight
- transformer.h.16.self_attention.dense.weight
- transformer.h.16.mlp.dense_h_to_4h.weight
- transformer.h.16.mlp.dense_4h_to_h.weight
- transformer.h.17.self_attention.query_key_value.weight
- transformer.h.17.self_attention.dense.weight
- transformer.h.17.mlp.dense_h_to_4h.weight
- transformer.h.17.mlp.dense_4h_to_h.weight
- transformer.h.18.self_attention.query_key_value.weight
- transformer.h.18.self_attention.dense.weight
- transformer.h.18.mlp.dense_h_to_4h.weight
- transformer.h.18.mlp.dense_4h_to_h.weight
- transformer.h.19.self_attention.query_key_value.weight
- transformer.h.19.self_attention.dense.weight
- transformer.h.19.mlp.dense_h_to_4h.weight
- transformer.h.19.mlp.dense_4h_to_h.weight
- transformer.h.20.self_attention.query_key_value.weight
- transformer.h.20.self_attention.dense.weight
- transformer.h.20.mlp.dense_h_to_4h.weight
- transformer.h.20.mlp.dense_4h_to_h.weight
- transformer.h.21.self_attention.query_key_value.weight
- transformer.h.21.self_attention.dense.weight
- transformer.h.21.mlp.dense_h_to_4h.weight
- transformer.h.21.mlp.dense_4h_to_h.weight
- transformer.h.22.self_attention.query_key_value.weight
- transformer.h.22.self_attention.dense.weight
- transformer.h.22.mlp.dense_h_to_4h.weight
- transformer.h.22.mlp.dense_4h_to_h.weight
- transformer.h.23.self_attention.query_key_value.weight
- transformer.h.23.self_attention.dense.weight
- transformer.h.23.mlp.dense_h_to_4h.weight
- transformer.h.23.mlp.dense_4h_to_h.weight
"""
}
    },
    "xlm-roberta": { #################################################### XLM-RoBERTa ######################################################
        "name": "xlm-roberta-base",
        "class_name": "AutoModelForSequenceClassification",
        "tokenizer_class": "AutoTokenizer",
        "tokenizer_name": "xlm-roberta-base",
        "sets": {
"init-layers-1": """
- roberta.encoder.layer.0.intermediate.dense.weight
- roberta.encoder.layer.0.output.dense.weight
- roberta.encoder.layer.1.intermediate.dense.weight
- roberta.encoder.layer.1.output.dense.weight
- roberta.encoder.layer.2.intermediate.dense.weight
- roberta.encoder.layer.2.output.dense.weight
- roberta.encoder.layer.3.intermediate.dense.weight
- roberta.encoder.layer.3.output.dense.weight
""",

"middle": """
- roberta.encoder.layer.4.intermediate.dense.weight
- roberta.encoder.layer.4.output.dense.weight
- roberta.encoder.layer.5.intermediate.dense.weight
- roberta.encoder.layer.5.output.dense.weight
- roberta.encoder.layer.6.intermediate.dense.weight
- roberta.encoder.layer.6.output.dense.weight
- roberta.encoder.layer.7.intermediate.dense.weight
- roberta.encoder.layer.7.output.dense.weight
""",

"last-layer": """
- roberta.encoder.layer.8.intermediate.dense.weight
- roberta.encoder.layer.8.output.dense.weight
- roberta.encoder.layer.9.intermediate.dense.weight
- roberta.encoder.layer.9.output.dense.weight
- roberta.encoder.layer.10.intermediate.dense.weight
- roberta.encoder.layer.10.output.dense.weight
- roberta.encoder.layer.11.intermediate.dense.weight
- roberta.encoder.layer.11.output.dense.weight
""",

"random": """
- roberta.encoder.layer.2.intermediate.dense.weight
- roberta.encoder.layer.2.output.dense.weight
- roberta.encoder.layer.4.intermediate.dense.weight
- roberta.encoder.layer.4.output.dense.weight
- roberta.encoder.layer.6.intermediate.dense.weight
- roberta.encoder.layer.6.output.dense.weight
- roberta.encoder.layer.9.intermediate.dense.weight
- roberta.encoder.layer.9.output.dense.weight
""",

"full": """
- roberta.encoder.layer.0.attention.self.query.weight
- roberta.encoder.layer.0.attention.self.key.weight
- roberta.encoder.layer.0.attention.self.value.weight
- roberta.encoder.layer.0.attention.output.dense.weight
- roberta.encoder.layer.0.intermediate.dense.weight
- roberta.encoder.layer.0.output.dense.weight
- roberta.encoder.layer.1.attention.self.query.weight
- roberta.encoder.layer.1.attention.self.key.weight
- roberta.encoder.layer.1.attention.self.value.weight
- roberta.encoder.layer.1.attention.output.dense.weight
- roberta.encoder.layer.1.intermediate.dense.weight
- roberta.encoder.layer.1.output.dense.weight
- roberta.encoder.layer.2.attention.self.query.weight
- roberta.encoder.layer.2.attention.self.key.weight
- roberta.encoder.layer.2.attention.self.value.weight
- roberta.encoder.layer.2.attention.output.dense.weight
- roberta.encoder.layer.2.intermediate.dense.weight
- roberta.encoder.layer.2.output.dense.weight
- roberta.encoder.layer.3.attention.self.query.weight
- roberta.encoder.layer.3.attention.self.key.weight
- roberta.encoder.layer.3.attention.self.value.weight
- roberta.encoder.layer.3.attention.output.dense.weight
- roberta.encoder.layer.3.intermediate.dense.weight
- roberta.encoder.layer.3.output.dense.weight
- roberta.encoder.layer.4.attention.self.query.weight
- roberta.encoder.layer.4.attention.self.key.weight
- roberta.encoder.layer.4.attention.self.value.weight
- roberta.encoder.layer.4.attention.output.dense.weight
- roberta.encoder.layer.4.intermediate.dense.weight
- roberta.encoder.layer.4.output.dense.weight
- roberta.encoder.layer.5.attention.self.query.weight
- roberta.encoder.layer.5.attention.self.key.weight
- roberta.encoder.layer.5.attention.self.value.weight
- roberta.encoder.layer.5.attention.output.dense.weight
- roberta.encoder.layer.5.intermediate.dense.weight
- roberta.encoder.layer.5.output.dense.weight
- roberta.encoder.layer.6.attention.self.query.weight
- roberta.encoder.layer.6.attention.self.key.weight
- roberta.encoder.layer.6.attention.self.value.weight
- roberta.encoder.layer.6.attention.output.dense.weight
- roberta.encoder.layer.6.intermediate.dense.weight
- roberta.encoder.layer.6.output.dense.weight
- roberta.encoder.layer.7.attention.self.query.weight
- roberta.encoder.layer.7.attention.self.key.weight
- roberta.encoder.layer.7.attention.self.value.weight
- roberta.encoder.layer.7.attention.output.dense.weight
- roberta.encoder.layer.7.intermediate.dense.weight
- roberta.encoder.layer.7.output.dense.weight
- roberta.encoder.layer.8.attention.self.query.weight
- roberta.encoder.layer.8.attention.self.key.weight
- roberta.encoder.layer.8.attention.self.value.weight
- roberta.encoder.layer.8.attention.output.dense.weight
- roberta.encoder.layer.8.intermediate.dense.weight
- roberta.encoder.layer.8.output.dense.weight
- roberta.encoder.layer.9.attention.self.query.weight
- roberta.encoder.layer.9.attention.self.key.weight
- roberta.encoder.layer.9.attention.self.value.weight
- roberta.encoder.layer.9.attention.output.dense.weight
- roberta.encoder.layer.9.intermediate.dense.weight
- roberta.encoder.layer.9.output.dense.weight
- roberta.encoder.layer.10.attention.self.query.weight
- roberta.encoder.layer.10.attention.self.key.weight
- roberta.encoder.layer.10.attention.self.value.weight
- roberta.encoder.layer.10.attention.output.dense.weight
- roberta.encoder.layer.10.intermediate.dense.weight
- roberta.encoder.layer.10.output.dense.weight
- roberta.encoder.layer.11.attention.self.query.weight
- roberta.encoder.layer.11.attention.self.key.weight
- roberta.encoder.layer.11.attention.self.value.weight
- roberta.encoder.layer.11.attention.output.dense.weight
- roberta.encoder.layer.11.intermediate.dense.weight
- roberta.encoder.layer.11.output.dense.weight
"""

}
    }
}


language = args.lang
base_model = " " + args.base_model
model = args.model
model_config = {
    "name": MODEL_SET[model]["name"],
    "class_name": MODEL_SET[model]["class_name"],
    "tokenizer_class": MODEL_SET[model]["tokenizer_class"],
    "tokenizer_name": MODEL_SET[model]["tokenizer_name"],
}


sets = MODEL_SET[model]["sets"]

for k, v in sets.items():
    out = f"""name: {model_config['name']}
class_name: {model_config['class_name']}
tokenizer_class: {model_config['tokenizer_class']}
tokenizer_name: {model_config['tokenizer_name']}
inner_params:{v}

pt: {base_model}
"""
    with open(f"{model}-{language}-{k}.yaml", "w") as f:
        f.write(out)
