import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import EditBatchSampler, dict_to
import random


POSITIVE_CLASS = "SUPPORTS"

print(">>>>>>>>>>>>>>>> Test FEVER Data Class <<<<<<<<<<<<<<<<")

class BinaryAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        max_length=141
    ):
        super().__init__()
        self.tokenizer = tokenizer
        # self.tokenizer.pad_token = tokenizer.eos_token # GPT - fc
        self.data = []
        self.config = config

        # from rich.pretty import pprint
        def extract(d):
            extracted = {k: d[k] for k in ["logit", "input", "prediction", "alternatives", "filtered_rephrases", "id"]}
            extracted["label"] = d["output"][0]["answer"]
            # ############ Temp ############
            # extracted["idx"] = d["id"]
            # ############ Temp ############
            return extracted

        with jsonlines.open(data_path) as f:
            for d in f:
                if len(d["alternatives"]) > 0 and len(d["filtered_rephrases"]) > 0:
                    self.data.append(extract(d))
                # else:
                #     pprint(d)
        print("Data Size: ", len(self.data))

        # ############ Temp ############
        # indices = [d["idx"] for d in self.data]
        # print("Indices: ", indices)
        # with open("/home/anonymous-xme/mend/mend/data_classes/test_data_indices.txt", "w") as f:
        #     f.write("\n".join(indices))

        # exit()

        # ############ Temp ############

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        rephrase = random.choice(self.data[item]["filtered_rephrases"])
        output = {
            "label": obj["label"] == POSITIVE_CLASS,
            "src": obj["input"],
            "rephrase": rephrase,
            "pred": obj["prediction"] == POSITIVE_CLASS,
            "alt": obj["alternatives"][0] == POSITIVE_CLASS,
            "cond_flip": "{} >> {} || {}".format(
                obj["prediction"],
                obj["alternatives"][0],
                obj["input"],
            ),
            "cond_orig": "{} >> {} || {}".format(
                obj["prediction"],
                obj["prediction"],
                obj["input"],
            ),
            "logit": obj["logit"],
        }

        return output

    def collate_fn(self, batch):
        loc_d = [b["rephrase"] for b in batch[:-1]]
        # src = [b["src"] for b in batch]
        src = batch[-1]["src"]
        rephrase = [batch[-1]["rephrase"]]

        flip_label = np.random.uniform() > 0.5
        predictions = [b["pred"] for b in batch]
        labels = [b["label"] for b in batch]
        labels[-1] = predictions[-1]  # the last element in the batch is special (the edit element)
        cond = [batch[-1]["cond_orig"]]
        if flip_label:
            labels[-1] = batch[-1]["alt"]
            cond = [batch[-1]["cond_flip"]]

        batches = {}
        for k1, v1 in {"": src, "loc_": loc_d, "cond_": cond, "rephrase_": rephrase}.items():
            # print(v1)
            encoded = self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
            for k2, v2 in encoded.items():
                batches[f"{k1}{k2}"] = v2

        batches["predictions"] = torch.tensor(predictions).long().view(-1)
        batches["loc_labels"] = torch.tensor(labels[:-1]).long().view(-1)
        batches["labels"] = torch.tensor(labels[-1]).long().view(-1)
        batches["raw"] = batch
        return batches

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, seed=self.config.seed)
        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in idxs])

            # print(">>>> Toks", toks)

            pass_keys = ["input_ids", "attention_mask", "labels"]
            edit_inner = {k: v[-1:] for k, v in toks.items() if k in pass_keys}

            # print(">>>> Edit Inner", edit_inner)

            if self.config.data.rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["rephrase_input_ids"]
                edit_outer["attention_mask"] = toks["rephrase_attention_mask"]
                edit_outer["labels"] = edit_inner["labels"]
            else:
                edit_outer = edit_inner
            
            # print(">>>> Edit Outer", edit_outer)

            pass_keys = ["loc_input_ids", "loc_attention_mask", "loc_labels"]
            loc_rep = {k: v for k, v in toks.items() if k in pass_keys}

            loc = {}
            loc["input_ids"] = loc_rep["loc_input_ids"]
            loc["attention_mask"] = loc_rep["loc_attention_mask"]
            loc["labels"] = loc_rep["loc_labels"]

            cond = {"input_ids": toks["cond_input_ids"], "attention_mask": toks["cond_attention_mask"]}

            # print(">>>> Loc", loc)
            # exit()

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond
            }
            yield dict_to(batch, self.config.device)
