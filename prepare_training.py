import conll 

from coref_resolver import CorefResolver
import os

import torch

from unbiasedbert import load_tokenizer
 

DATAFILE = 'augmentation_data/original_with_swapped_word.txt'
OUTFILE = 'augmentation_data/tokenized_coref_data.pkl'
OUTDIR = 'augmentation_data/processed_data'

tokenizer = load_tokenizer()

resolver = CorefResolver(tokenizer, None)

def main():
    data = conll.read_augmented_data(DATAFILE)
    documents_data = []
    for i, doc in enumerate(data):
        print('\r{}/{}'.format(i+1,len(data)), end='')
        out_data = resolver.create_training_data(doc)
        documents_data.append(out_data)
        torch.save(out_data, os.path.join(OUTDIR, "{:04d}.pkl".format(i)))
    torch.save(documents_data, OUTFILE)
    print('\nDone.')

    

if __name__ == '__main__':
    main()