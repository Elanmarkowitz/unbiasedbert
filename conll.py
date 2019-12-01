"""
1. Current sentence word number (1-based)
2.  the word
3.  lemma form of the word
4.  coarse-grained POS tag
5.  fine-grained POS tag (in practice, both of these are the same)
6.  morphological features (in practice, just the token ‘-’)
7.  token id of this word’s parent (head), or 0 if this word’s parent is the root.  
    If this data has not yetbeen provided (i.e.  the sentence isn’t parsed) it is ’-’.
8.  dependency label for the relation of this word to its parent.  
    If this data has not yet been provided (i.e.the sentence isn’t parsed) it is ’-’.
9.  other dependencies (in practice, just ‘-’)
10.  other info (in practice, just ‘-’)
"""
def word_id(conll):
    return int(conll[0])

def word(conll):
    return conll[4]

def lemma_word(conll):
    return conll[2].lower()

def pos(conll):
    return conll[3]

def parent(conll):
    return int(conll[6])

def write_parent(conll, parent):
    conll[6] = str(parent)
    return conll 

def arc_label(conll):
    return conll[7]

def write_arc_label(conll, label):
    conll[7] = label
    return conll 

def alt_word(conll):
    return conll[5]

def coref(conll):
    return conll[-1]

def get_spans(conll):
    if coref(conll) == '-':
        return [], [], []
    starts = []
    ends = []
    singles = []
    spans = coref(conll).split('|')
    for span in spans:
        if span[0] == '(' and span[-1] == ')':
            singles.append(int(span[1:-1]))
        elif span[0] == '(':
            starts.append(int(span[1:]))
        elif span[-1] == ')':
            ends.append(int(span[:-1]))
    return starts, ends, singles

def read_augmented_data(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read()
    documents = data.split('#end document')
    documents[0] = documents[0][1:]  # skip column header line
    documents = [d.splitlines()[1:] for d in documents]
    out_docs = []
    for doc in documents:
        sentences = []
        sentence = []
        for line in doc:
            line_data = line.split()
            if len(line_data) <= 2:
                if sentence:
                    sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line_data) 
        if sentence:
            sentences.append(sentence)
        out_docs.append(sentences)
    return out_docs[:-1]

def read_data(filename):
    DELIMITER = '\t'
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read()
    documents = data.split('#end document')
    documents = [d.splitlines()[0:] for d in documents]
    out_docs = []
    for doc in documents:
        sentences = []
        sentence = []
        for line in doc:
            if line == '':
                if sentence:
                    sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line.split(DELIMITER)) 
        if sentence:
            sentences.append(sentence)
        out_docs.append(sentences)
    return out_docs[:-1]

def write_data(filename, sentences):
    output_lines = []
    for sentence in sentences:
        for token in sentence:
            line = "\t".join(token)
            output_lines.append(line)
        output_lines.append('')
    with open(filename, "w") as f:
        f.writelines('\n'.join(output_lines) + '\n')

