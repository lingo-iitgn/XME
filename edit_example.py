import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import transformers
import higher
import logging
from higher.patch import monkeypatch as make_functional
from collections import defaultdict

from editable_model import EditableModel
from hooks import hook_model
import nn as local_nn
from utils import _logits, _inner_params, load_archive

from algs.mend import MEND
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from rich.pretty import pprint

LOG = logging.getLogger(__name__)


import types
import json

device = "cuda:0"
# device = "cpu"

model_checkpoint = "bigscience/bloom-560m"
trained_model = "./data/fever/bloom_560m_s1.bin"
mend_trained = "./data/fever/mend-fever-run-full/models/bloom-560m.2023-03-18_14-10-44_997615502.bk"
config_path = "./data/fever/mend-fever-run-full/config.json"
data_path = "./data/fever/1_fever_train_1200 - hindi_1200.jsonl"

model = BloomForSequenceClassification.from_pretrained(model_checkpoint).to(device)
tokenizer = BloomTokenizerFast.from_pretrained(model_checkpoint)
model.load_state_dict(torch.load(trained_model, map_location=device))

# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

# config = types.SimpleNamespace()
# config.model = types.SimpleNamespace()

# config.model.inner_params = [
# "transformer.h.20.mlp.dense_h_to_4h.weight",
# "transformer.h.20.mlp.dense_4h_to_h.weight",
# "transformer.h.21.mlp.dense_h_to_4h.weight",
# "transformer.h.21.mlp.dense_4h_to_h.weight",
# "transformer.h.22.mlp.dense_h_to_4h.weight",
# "transformer.h.22.mlp.dense_4h_to_h.weight",
# "transformer.h.23.mlp.dense_h_to_4h.weight",
# "transformer.h.23.mlp.dense_4h_to_h.weight"
# ]

# config.edit_lr = 1e-2

# config.mend = types.SimpleNamespace()
# config.mend.n_hidden = 1
# # config.mend = config.mend.__dict__
# config.mend.shared = True

with open(config_path, "r") as f:
    d = json.load(f)


class RecursiveNamespace(types.SimpleNamespace):

  @staticmethod
  def map_entry(entry):
    if isinstance(entry, dict):
      return RecursiveNamespace(**entry)

    return entry

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    for key, val in kwargs.items():
      if type(val) == dict:
        setattr(self, key, RecursiveNamespace(**val))
      elif type(val) == list:
        setattr(self, key, list(map(self.map_entry, val)))



config = RecursiveNamespace(**d)
pprint(config)


mend = MEND(model, config, lambda: copy.deepcopy(model)).to(device)

# import pdb; pdb.set_trace()
archive, config.archive = load_archive(mend_trained)
mend.load_state_dict(archive["model"])
mend.to(device)


############################# Load Data

from data_classes.fever import BinaryAugmentedKILT
test_data = BinaryAugmentedKILT(tokenizer, data_path, config)
test_loader = test_data.edit_generator(batch_size=config.batch_size)


#########################

# x = torch.arange(20).view(1, 20).to(device) + 1000
batch = next(iter(test_loader))
pprint(batch)
orig_logits = mend(**batch["edit_inner"])
edited = mend.edit(batch["edit_inner"])
post_logits = mend(**batch["edit_inner"])

assert torch.allclose(orig_logits, post_logits)

orig_param = [p for (n, p) in mend.model.named_parameters() if n == config.model.inner_params[-1]][0]
edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.model.inner_params[-1]][0]

LOG.info((orig_param - edited_param).abs().max())
edited.eval()
LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)
