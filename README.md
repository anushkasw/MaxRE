# Maximizing Relation Extraction Potential: A Data-Centric Study to Unveil Challenges and Opportunities

This repository contains code associated with the paper ["Maximizing Relation Extraction Potential: A Data-Centric Study to Unveil Challenges and Opportunities"](https://ieeexplore.ieee.org/document/10747504).


### Datasets
The open-source datasets used for this study can be downloaded from the links below. Note that
TACRED and RETACRED are not publicly available and can be downloaded from LDC. Use the script `Preprocessing/ent_annot.py` to extract entity type annotations using Stanford CoreNLP:
- [SemEval-2010 Task 8](https://github.com/sahitya0000/Relation-Classification)
- [NYT10](https://github.com/truthless11/HRL-RE/tree/master)
- [FewRel](https://thunlp.github.io/1/fewrel1.html)
- [WebNLG](https://github.com/weizhepei/CasRel/tree/master)
- [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24)
- [RETACRED](https://github.com/gstoica27/Re-TACRED)
- [CrossRE](https://github.com/mainlp/CrossRE)


### Algorithms

Use the following repositories to access the original source code for each algorithm used in this study:
- [Att-BLSTM](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction)
- [PAWARE](https://github.com/yuhaozhang/tacred-relation)
- [Entity-Att](https://github.com/roomylee/entity-aware-relation-classification)
- [RBERT](https://github.com/mickeysjm/R-BERT)
- [Roberta_base](https://github.com/wzhouad/RE_improved_baseline)
- [LUKE](https://github.com/studio-ousia/luke)
- [ERNIE](https://github.com/thunlp/ERNIE)
- [GenPT](https://github.com/hanjiale/GenPT)
- [KnowPrompt](https://github.com/zjunlp/KnowPrompt)
- [RIFRE](https://github.com/zhao9797/RIFRE)
- [SPN4RE](https://github.com/DianboWork/SPN4RE)
- [TDEER](https://github.com/4AI/TDEER)
- [UniRel](https://github.com/wtangdev/UniRel)

The original source code for the LLM-based algorithms can be found in the following repositories. Also, our version of the code which uses OpenAI's
batch API can be found in the folder `LLM-RC`
- [GPT-RE](https://github.com/yukinowan/gpt-re)
- [UnleashLLM](https://github.com/zjunlp/DeepKE/blob/main/example/llm/UnleashLLMRE/README.md)


### Cite
If you plan on using any of our resources please cite the paper:

```
@ARTICLE{10747504,
  author={Swarup, Anushka and Bhandarkar, Avanti and Dizon-Paradis, Olivia P. and Wilson, Ronald and Woodard, Damon L.},
  journal={IEEE Access}, 
  title={Maximizing Relation Extraction Potential: A Data-Centric Study to Unveil Challenges and Opportunities}, 
  year={2024},
  volume={12},
  number={},
  pages={167655-167682},
  keywords={Feature extraction;Data mining;Classification algorithms;Neural networks;Training;Software algorithms;Semantics;Performance analysis;Machine learning algorithms;Information retrieval;Information extraction;joint relation extraction;large language models;natural language processing;relation classification;relation extraction},
  doi={10.1109/ACCESS.2024.3494737}}
```
