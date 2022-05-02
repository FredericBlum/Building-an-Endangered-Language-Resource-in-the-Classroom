from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

################################
### data and dictionaries    ###
################################
is_forward_lm = True
is_backward_lm = False
lm_forward = FlairEmbeddings('multi-forward').lm
lm_backward = FlairEmbeddings('multi-backward').lm

char_dict: Dictionary = lm_forward.dictionary

shp_for = TextCorpus("data/Shipibo/embeddings/char_lm", char_dict, is_forward_lm, character_level = True)
shp_back = TextCorpus("data/Kakataibo/embeddings/char_lm", char_dict, is_forward_lm, character_level = True)

cbr_for = TextCorpus("data/Shipibo/embeddings/char_lm", char_dict, is_backward_lm, character_level = True)
cbr_back = TextCorpus("data/Kakataibo/embeddings/char_lm", char_dict, is_backward_lm, character_level = True)

################################
### Trainers                 ###
################################
trainer_forward = LanguageModelTrainer(lm_forward, cbr_for)
trainer_backward = LanguageModelTrainer(lm_backward, cbr_back)

trainer_forward.train(f'models/resources/embeddings/multi-ft_cbr_for',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=16,
                max_epochs=15)

trainer_backward.train(f'models/resources/embeddings/multi-ft_cbr_back',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=16,
                max_epochs=15)
