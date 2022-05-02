# Read-Me

This repository accompanies the paper "Building an Endangered Language Resource in the Classroom: Universal Dependencies for Kakataibo", presented at the LREC2022. It contains the code that has been used for the experiments, the raw data for all treebanks used in those experiments, as well as the different training/dev/test splits and embeddings that have been used. Further, the folder "experiments" contains the raw results, the scripts that have been used for summarizing the results, scripts for creating summary statistics of the raw data, as well as scripts that derive the distribution of tags across the different splits.

The experiments relied mainly on two frameworks that provide implementations for various model architectures: [flair](https://github.com/flairNLP/flair) (Akbik et al. 2019, version 0.10) for POS-tagging, and [supar](https://github.com/yzhangcs/parser) (Zhang et al. 2020, version 1.01) for dependency parsing.

The treebanks for Kazakh that have been used in some of the experiments are part of the UD_Kazakh treebanks that are published in the current UD release, [version 2.9](https://github.com/UniversalDependencies/UD_Kazakh-KTB) (Makazhanov et al. 2015, Tyers et al. 2015).

## Instructions for the dependency parser experiment

On the gpu server we have available, only python3.6 runs. This means we can only install an older version of supar (1.0.1) and we needed to do some tweaks for the code. The most important change for reproducing the results with newer versions:

- `cmds.biaffine_dependency` becomes `cmds.biaffine_dep` in newer versions.
- `cmds.crf2o_dependency` becomes `cmds.crf2o_dependency` in newer versions.

We recommend to run the experiment with the updated version of supar.

### Biaffine Dependency parser

```python
python -u -m supar.cmds.biaffine_dependency train -b -d 0 \
    -p models/cbr1 -f char \
    --embed data/Shipibo/delex/glove.txt \
    --encoder=bert  \
    --bert=xlm-roberta-large  \
    --train data/Shipibo/delex/train.conllu \
    --dev data/Shipibo/delex/dev.conllu \
    --test data/Kakataibo/delex/all_in_one.conllu \
    --n-embed 512 \
    --unk=''
```

Code for evaluation:

```python
python -u -m supar.cmds.biaffine_dependency evaluate -d 0 -p models/cbr1 --data data/Kakataibo/conllu/all_in_one.conllu --tree --proj
```

Fine-tuning xlm-roberta (not yet implemented):

```python
python -u -m supar.cmds.biaffine_dependency train -b -d 0 -c biaffine-dep-xlmr -p models/xlm_shp1  \
    --train data/Kakataibo/conllu/train_60.conllu \
    --dev data/Kakataibo/conllu/dev_20.conllu \
    --test data/Kakataibo/conllu/test_20.conllu \
    --encoder=bert  \
    --bert=xlm-roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=100  \
    --epochs=100  \
    --update-steps=4
```

### Commands for CRF2o dependency parsing

```python
python -u -m supar.cmds.crf2o_dependency train -b -d 0 -p guacamole/myu1 -f char \
    --embed data/Munduruku/embeddings/glove.txt \
    --train data/Munduruku/conllu/train.conllu \
    --dev data/Munduruku/conllu/dev.conllu \
    --test data/Munduruku/conllu/test.conllu \
    --n-embed 512 --unk='' --mbr --proj
```

Evaluation:

```python
python -u -m supar.cmds.crf2o_dependency evaluate -d 0 -p models/shp1 \
    --data data/Kakataibo/conllu/all_in_one.conllu \ 
    --tree  --proj --mbr
```

## Referencs

Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., and Vollgraf, R. (2019). Flair: An easy-to-use framework for state-of-the-art NLP. In NAACL 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 54–59.

Makazhanov, A., Sultangazina, A., Makhambetov, O., and Yessenbayev, Z. (2015). Syntactic Annotation of Kazakh: Following the Universal Dependencies Guidelines. A report. In 3rd International Conference on Turkic Languages Processing, (TurkLang 2015), pages 338–350.

Tyers, F. M. and Washington, J. N. (2015). Towards a Free/Open-source Universal-dependency Treebank for Kazakh. In 3rd International Conference on Turkic Languages Processing, (TurkLang 2015), pages 276–289.

Zhang, Y., Li, Z., and Min, Z. (2020). Efficient Second-Order TreeCRF for Neural Dependency Parsing. In Proceedings of ACL, pages 3295–3305.
