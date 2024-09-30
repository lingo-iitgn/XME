# Cross-lingual Editing in Large Language Models

## Setup

### Environment

This codebase uses Python 3.7.9. Other versions may work as well.

Create a virtualenv ([pyenv](https://github.com/pyenv/pyenv) can help with this)
and install the dependencies:

    $ python -m venv env
    $ source env/bin/activate
    (env) $ pip install -r requirements.txt

### Data

You can download the data needed for this project from [this Google Drive link](https://drive.google.com/drive/folders/1BFthZvNEgCZ1Nt35nGCDYLwXVK7Y7as1?usp=sharing).
Download the dataset and change the path in the `run.py` with the path of the dataset where it is downloaded.

## Citation

```
@inproceedings{beniwal-etal-2024-cross,
    title = "Cross-lingual Editing in Multilingual Language Models",
    author = "Beniwal, Himanshu  and
      D, Kowsik  and
      Singh, Mayank",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.140",
    pages = "2078--2128",
    abstract = "The training of large language models (LLMs) necessitates substantial data and computational resources, and updating outdated LLMs entails significant efforts and resources. While numerous model editing techniques (METs) have emerged to efficiently update model outputs without retraining, their effectiveness in multilingual LLMs, where knowledge is stored in diverse languages, remains an underexplored research area. This research paper introduces the cross-lingual model editing (XME) paradigm, wherein a fact is edited in one language, and the subsequent update propagation is observed across other languages. To investigate the XME paradigm, we conducted experiments using BLOOM, mBERT, and XLM-RoBERTa using the two writing scripts: Latin (English, French, and Spanish) and Indic (Hindi, Gujarati, and Bengali). The results reveal notable performance limitations of state-of-the-art METs under the XME setting, mainly when the languages involved belong to two distinct script families. These findings highlight the need for further research and development of XME techniques to address these challenges. For more comprehensive information, the dataset used in this research and the associated code are publicly available at the following [URL](https://github.com/lingo-iitgn/XME).",
}
```
