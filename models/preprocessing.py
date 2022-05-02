from helper_functions import conllu_to_flair, conllu_split, concat_glove
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec


# These commands re-create many of the files used in the experiments, so use with caution.

#conllu_to_flair('data/raw/shipibo-2018jul4.converted.conllu', lang = 'Shipibo', write_raw=True, write_delex=True)
#conllu_split('data/raw/shipibo-2018jul4.converted.conllu', lang = 'Shipibo', write_testset=True, write_trainset=True,  write_delex=False)
# concat_glove(Shipibo)
# glove_input_file = 'data/Shipibo/embeddings/glove.txt'
# word2vec_output_file = 'data/Shipibo/embeddings/word2vec.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)

# word_vectors = gensim.models.KeyedVectors.load_word2vec_format('data/Shipibo/embeddings/word2vec.txt', binary=False)
# word_vectors.save('data/Shipibo/embeddings/gensim_shp')

#conllu_to_flair('data/raw/treebank-kakataibo.conllu', lang = 'Kakataibo', write_raw=True, write_delex=True)
conllu_split('data/raw/treebank-kakataibo.conllu', lang = 'Kakataibo', write_testset=True, write_trainset=True, write_delex=False)

glove_input_file = 'data/Kakataibo/embeddings/glove.txt'
word2vec_output_file = 'data/Kakataibo/embeddings/word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# word_vectors = gensim.models.KeyedVectors.load_word2vec_format('data/Kakataibo/embeddings/word2vec.txt', binary=False)
# word_vectors.save('data/Kakataibo/embeddings/gensim_cbr') 

#conllu_to_flair('data/raw/ktb_ud.conllu', lang = 'Kazakh', write_raw=True, write_delex=True)
#conllu_split('data/raw/ktb_ud.conllu', lang = 'Kazakh', write_testset=True, write_trainset=True, write_delex=True)
