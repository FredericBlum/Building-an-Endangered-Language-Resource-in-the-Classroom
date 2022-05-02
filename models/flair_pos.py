from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from helper_functions import make_trainset, make_evalset

################################
### data and dictionaries    ###
################################
columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

#corpus = make_trainset("Shipibo", reduced=False)
corpus = make_trainset("Kakataibo", reduced=True)
#corpus = make_trainset("Kazakh", reduced=False)

#ft_corpus = make_trainset("Kakataibo", reduced=True)
#eval_corpus1 = make_evalset(language='Kakataibo')
#eval_corpus2 = make_evalset(language='Shipibo')
#eval_corpus = MultiCorpus([eval_corpus1, eval_corpus2])

upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

################################
### Embeddings               ###
################################
#flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/cbr_for/best-lm.pt')
#flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/cbr_back/best-lm.pt')
flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/multi-ft_cbr_for/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/multi-ft_cbr_back/best-lm.pt')
#flair_embedding_forward = FlairEmbeddings('multi-forward')
#flair_embedding_backward = FlairEmbeddings('multi-backward')

#word_embeddings = WordEmbeddings('data/Shipibo/embeddings/gensim_shp')
#tf_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased')
embeddings = StackedEmbeddings(embeddings=[#tf_embeddings,
                                           #word_embeddings,
                                           flair_embedding_forward, flair_embedding_backward])
################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger(hidden_size=512,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)

trainer.train('models/resources/tagger_pos',
                monitor_train=True,
                monitor_test=True,
                patience=3,
                learning_rate=1,
                mini_batch_size=32,
                max_epochs=300)

###############################
### Evaluation              ###
###############################
tagger = SequenceTagger.load('models/resources/tagger_pos/final-model.pt')

#trainer = ModelTrainer(tagger, ft_corpus)
#trainer.fine_tune('models/resources/finetune', mini_batch_size=64, max_epochs=100, use_final_model_for_eval=False, learning_rate=0.05)

#trainer = ModelTrainer(tagger, eval_corpus)
#trainer.final_test('models/resources/tagger_eval', main_evaluation_metric = ("macro avg", "f1-score"), eval_mini_batch_size = 64)
