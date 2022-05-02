from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


################################
### data and dictionaries   ####
################################
dictionary: Dictionary = Dictionary.load('chars')

is_forward_lm = True
is_backward_lm = False

corpus_for = TextCorpus('data/Kakataibo/embeddings/char_lm', dictionary, is_forward_lm, character_level=True)
corpus_back = TextCorpus('data/Kakataibo/embeddings/char_lm', dictionary, is_backward_lm, character_level=True)

################################
### Language Model          ####
################################
lm_for = LanguageModel(dictionary, is_forward_lm, hidden_size=512, nlayers=1)
lm_back = LanguageModel(dictionary, is_forward_lm, hidden_size=512, nlayers=1)

################################
### Trainer                 ####
################################
trainer = LanguageModelTrainer(lm_for, corpus_for)
trainer.train('models/resources/embeddings/cbr_for', sequence_length=100, learning_rate=20, mini_batch_size=32,max_epochs=200)

trainer = LanguageModelTrainer(lm_back, corpus_back)
trainer.train('models/resources/embeddings/cbr_back', sequence_length=100, learning_rate=20, mini_batch_size=32,max_epochs=200)