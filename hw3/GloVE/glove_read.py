import os
import numpy as np
working_dir = os.getcwd()
glove_file = 'glove.6B.100d.txt'

emb_dict = {}
glove = open(os.path.join(working_dir, glove_file),encoding="utf-8")

for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
    
glove.close()

print("there is %d words in the glove embedding" %(len(emb_dict)))
print("there is %d vectors in the 'the' embedding" %(len(emb_dict['the'])))