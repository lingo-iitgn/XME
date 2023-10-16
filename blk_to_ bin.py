import torch
from rich.pretty import pprint
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from collections import OrderedDict

# import os
# import six
# from google.cloud import translate_v2 as translate
# # import google.auth
# from dotenv import load_dotenv

# load_dotenv()

# def translate_text(source, target, text):
#     """Translates text into the target language.

#     Target must be an ISO 639-1 language code.
#     See https://g.co/cloud/translate/v2/translate-reference#supported_languages
#     """
    

#     translate_client = translate.Client()

#     if isinstance(text, six.binary_type):
#         text = text.decode("utf-8")

#     # Text can also be a sequence of strings, in which case this method
#     # will return a sequence of results for each text.
#     result = translate_client.translate(text, target_language=target, source_language=source)

#     print(u"Text: {}".format(result["input"]))
#     print(u"Translation: {}".format(result["translatedText"]))
#     print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

# translate_text("en", "hi", "Fox 2000 Pictures released the film Soul Food.")


# m = torch.load("/home/anonymous-xme/mend/mend/data/fever/bloom_560m_s1.bin")
# m = torch.load("/home/anonymous-xme/mend/mend/outputs/2023-03-18_14-10-44_997615502/models/bloom-560m.2023-03-18_14-10-44_997615502.bk", map_location=torch.device('cpu'))
m = torch.load("/home/anonymous-xme/mend/mend/outputs/2023-03-26_12-54-45_6262973532/models/bloom-560m.2023-03-26_12-54-45_6262973532.bk", map_location=torch.device('cpu'))
pprint(m["model"].keys())

model_checkpoint="bigscience/bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(model_checkpoint) # , bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>'
model = BloomForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

wt = OrderedDict()
for k, v in m["model"].items():
    wt[k[6:]] = v


model.load_state_dict(wt)
torch.save(model.state_dict(), "/home/anonymous-xme/mend/mend/data/fever/bloom-finetuned-on-ft/models/bloom-560m-fever.pth")

# pprint(m)