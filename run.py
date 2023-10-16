import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils


# from trainer import EditTrainer
import models


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.word_embeddings.weight.data[-1] = model.transformer.word_embeddings.weight.data.mean(0) # Bloom
    # model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0) # GPT Style


@hydra.main(config_path='config', config_name='config')
def run(config):
    # print(config.test_file)
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    # for name, _ in model.named_parameters():
    #     print(name)
    tokenizer = models.get_tokenizer(config)

    import transformers
    if isinstance(model, transformers.GPT2ForSequenceClassification):
        model.config.pad_token_id = model.config.eos_token_id

    if config.task == "gen" or config.task == "wiki":
        add_padding(tokenizer, model)
        from data_classes.wiki import GenDataset

        train_set = GenDataset("train", tokenizer, config, config.data.path, pct=10)
        val_set = GenDataset("validation", tokenizer, config, config.data.path, pct=10)
    elif config.task == "fc" or config.task == "fever":
        from data_classes.fever import BinaryAugmentedKILT

        if config.tests:
            # from data_classes.fever_test import BinaryAugmentedKILT
            train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/{config.train_set}", config)
            val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/{config.val_set}", config)
        else:
            if config.lang == "english":
                # English
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever-train-kilt.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever-dev-kilt.jsonl", config)
            elif config.lang == "hindi":
                # Hindi
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - hindi-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - hindi-10K.jsonl", config)

            elif config.lang == "spanish":
                # Spanish
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - spanish-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev-spanish-10K.jsonl", config)
            
            elif config.lang == "french":
                # French
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - french-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - french-10K.jsonl", config)

            elif config.lang == "bengali":
                # Bengali
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - bengali-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - bengali-10K.jsonl", config)

            elif config.lang == "gujarati":
                # Gujarati
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - gujarati-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - gujarati-10K.jsonl", config)
            
            elif config.lang == "malayalam":
                # Malayalam
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - malayalam-1L_lang_code.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - malayalam-10K_lang_code.jsonl", config)
            
            elif config.lang == "tamil":
                # Tamil
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - tamil-1L_lang_code.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - tamil-10K_lang_code.jsonl", config)

            elif config.lang == "kannada":
                # Kannada
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - kannada-1L_lang_code.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - kannada-10K_lang_code.jsonl", config)

            elif config.lang == "chinese":
                # Chinese
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - chinese-1L_lang_code.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - chinese-10K_lang_code.jsonl", config)

            elif config.lang == "arabic":
                # Arabic
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - arabic-1L_lang_code.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - arabic-10K_lang_code.jsonl", config)

            elif config.lang == "mixed":
                # Mixed
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - mixed-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - mixed-10K.jsonl", config)

            elif config.lang == "inverse":
                # Mixed
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - mixed_inv-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - mixed_inv-10K.jsonl", config)

            elif config.lang == "inverse-xlm":
                # Mixed
                train_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_train - inverse-xlm-1L.jsonl", config)
                val_set = BinaryAugmentedKILT(tokenizer, f"{base_dir}/data/fever/fever_dev - inverse-xlm-10K.jsonl", config)

    elif config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import Seq2SeqAugmentedKILT

        train_set = Seq2SeqAugmentedKILT(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl",
                                         config)
        val_set = Seq2SeqAugmentedKILT(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl",
                                       config)

    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    if config.tests:
        from tester import EditTrainer
    else:
        from trainer import EditTrainer

    trainer = EditTrainer(alg, config, train_set, val_set)
    trainer.run()


if __name__ == "__main__":
    run()
