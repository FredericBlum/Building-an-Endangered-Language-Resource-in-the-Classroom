# Read-Me

This repository accompanies the paper "Building an Endangered Language Resource in the Classroom: Universal Dependencies for Kakataibo", presented at the LREC2022. It contains the code that has been used for the experiments, the raw data for all treebanks used in those experiments, as well as the different training/dev/test splits and embeddings that have been used.

The folder */data/* is the first release of the Kakataibo and Shipibo treebanks. Different subfolders include the different embeddings and delixcalized versions of the data, as well as the different splits. The folder */models/* contains the POS-tagging scripts and all resources used for the experiments. Further, the folder */experiments/* contains the raw results, the scripts that have been used for summarizing the results, scripts for creating summary statistics of the raw data, as well as scripts that derive the distribution of tags across the different splits.

The experiments relied mainly on two frameworks that provide implementations for various model architectures: [flair](https://github.com/flairNLP/flair) (Akbik et al. 2019, version 0.10) for POS-tagging, and [supar](https://github.com/yzhangcs/parser) (Zhang et al. 2020, version 1.01) for dependency parsing.

The treebanks for Kazakh that have been used in some of the experiments are part of the UD_Kazakh treebanks that are published in the current UD release, [version 2.9](https://github.com/UniversalDependencies/UD_Kazakh-KTB) (Makazhanov et al. 2015, Tyers et al. 2015).

## Instructions for the Part-of-speech tagging experiment

The flair-implementation for the POS-tagging is mainly organized in scripts within the folder */models/*. The script *preprocessing.py* organizes all splits for the different datasets. The splits have been made with a random seed of 42 and make use of many of the functions imported from *helper_functions.py*. The next step was training the embeddings, which has been done in *flair_emb.py* for the monolingual embeddings, and *flair_ft-multi.py* for the finetuned embeddings of the jw300-embeddings respectively. Finally, the script *flair_pos.py* implements various of the settings used for the experiment. The input needs to be changed manually in order to reproduce the different experiment settings. Templates are included in the script, but are currently commented out so that only one model runs at a time.

The output of all embeddings and models is saved in the folder */models/resources/*. Due to size reasons, the models are not included in the repository.

## Instructions for the dependency parser experiment

The dependency parser is included as a command line command. On the gpu server we have available, only python3.6 runs. This means we can only install an older version of supar (1.0.1) and we needed to do some tweaks for the code. The most important change for reproducing the results with newer versions:

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

## How to cite

If you use the Kakataibo and/or Shipibo-Konibo treebanks, please cite the following articles:

```
@inproceedings{zariquiey-etal-2022-building,
    title = "Building an Endangered Language Resource in the Classroom: {U}niversal {D}ependencies for {K}akataibo",
    author = "Zariquiey, Roberto  and
      Alvarado, Claudia and
      Echevarría, Ximena and
      Gomez, Luisa and
      Gonzales, Rosa and
      Illescas, Marian and
      Oporto, Sabina and
      Blum, Frederic and 
      Oncevay, Arturo  and
      Vera, Javier",
    booktitle = "Proceedings of the 13th Language Resources and Evaluation Conference",
    month = june,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "TBD",
    language = "English",
}

@inproceedings{vasquez-etal-2018-toward,
    title = "Toward {U}niversal {D}ependencies for {S}hipibo-Konibo",
    author = "Vasquez, Alonso  and
      Ego Aguirre, Renzo  and
      Angulo, Candy  and
      Miller, John  and
      Villanueva, Claudia  and
      Agi{\'c}, {\v{Z}}eljko  and
      Zariquiey, Roberto  and
      Oncevay, Arturo",
    booktitle = "Proceedings of the Second Workshop on Universal Dependencies ({UDW} 2018)",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-6018",
    doi = "10.18653/v1/W18-6018",
    pages = "151--161",
}
```

## References

Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., and Vollgraf, R. (2019). Flair: An easy-to-use framework for state-of-the-art NLP. In NAACL 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 54–59.

Makazhanov, A., Sultangazina, A., Makhambetov, O., and Yessenbayev, Z. (2015). Syntactic Annotation of Kazakh: Following the Universal Dependencies Guidelines. A report. In 3rd International Conference on Turkic Languages Processing, (TurkLang 2015), pages 338–350.

Tyers, F. M. and Washington, J. N. (2015). Towards a Free/Open-source Universal-dependency Treebank for Kazakh. In 3rd International Conference on Turkic Languages Processing, (TurkLang 2015), pages 276–289.

Zhang, Y., Li, Z., and Min, Z. (2020). Efficient Second-Order TreeCRF for Neural Dependency Parsing. In Proceedings of ACL, pages 3295–3305.
