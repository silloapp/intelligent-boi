import pickle
import numpy as np

embeddings_dictionary = dict()
#download from https://nlp.stanford.edu/projects/glove/
glove_file = open('glove.twitter.27B.200d.txt',encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

pickle.dump( embeddings_dictionary, open( "embeddings.p", "wb" ) )