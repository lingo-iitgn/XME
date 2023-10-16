import torch
from rich.pretty import pprint
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from collections import OrderedDict

# m = torch.load("/home/anonymous-xme/mend/mend/data/fever/bloom_560m_s1.bin")
# m = torch.load("/home/anonymous-xme/mend/mend/outputs/2023-03-18_14-10-44_997615502/models/bloom-560m.2023-03-18_14-10-44_997615502.bk", map_location=torch.device('cpu'))
m = torch.load("/home/anonymous-xme/mend/mend/data/fever/bloom_560m_s1.bin", map_location=torch.device('cpu'))
pprint(m)