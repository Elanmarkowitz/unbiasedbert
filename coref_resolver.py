from collections import defaultdict, namedtuple

import data_utils


Span = namedtuple('Span', ['sentence_idx','start', 'end'])

class CorefResolver:
    def __init__(self, tokenizer, model, sentence_diff_threshold=2):
        self.references = []
        self.mentions = defaultdict(list)
        self.tokenizer = tokenizer
        self.model = model 
        self.document = None
        self.sentences = []
        self.spans = []
        self.out_data = []
        self.sentence_diff_threshold = sentence_diff_threshold

    def create_training_data(document):
        self.document = document
        self.sentences = []
        self.out_data = []
        for sentence in self.document:
            orig_spans, swap_spans = data_utils.get_sentence_spans(self.tokenizer, sentence)
            sequence_versions = data_utils.get_tokenized_versions(self.tokenizer, sentence)
            self.sentences.append(sequence_versions)  # orig_masked_sequence, orig_label_sequence, swap_masked_sequence, swap_label_sequence
            cur_sentence_idx = len(self.sentences) - 1
            for orig_span, swap_span in zip(orig_spans, swap_spans):
                id = orig_span[0]
                assert id == swap_span[0]
                for reference in references:
                    orig_comp_span, swap_comp_span = self.mentions[reference][-1] # last mention of reference
                    comp_id = orig_comp_span[0]
                    assert comp_id == swap_comp_span[0]
                    comp_sentence_idx = orig_comp_span.sentence_idx

                    if cur_sentence_idx - comp_sentence_idx > self.sentence_diff_threshold:
                        # skip if more than threshold sentences away
                        continue

                    label = 1 if id == comp_id else 0
                    orig_span1_start = orig_comp_span.start
                    orig_span1_end = orig_comp_span.end 
                    swap_span1_start = swap_comp_span.start
                    swap_span1_end = swap_comp_span.end
                    if comp_sentence_idx == cur_sentence_idx:
                        combined_orig_masked_seq = self.sentences[comp_sentence_idx][0]
                        combined_orig_label_seq = self.sentences[comp_sentence_idx][1]
                        combined_swap_masked_seq = self.sentences[comp_sentence_idx][2]
                        combined_swap_label_seq = self.sentences[comp_sentence_idx][3]
                        orig_span2_start = orig_span[1]
                        orig_span2_end = orig_span[2]
                        swap_span2_start = swap_span[1]
                        swap_span2_end = swap_span[2]
                    else:
                        combined_orig_masked_seq = self.sentences[comp_sentence_idx][0] + self.sentences[cur_sentence_idx][0]
                        combined_orig_label_seq = self.sentences[comp_sentence_idx][1] + self.sentences[cur_sentence_idx][1]
                        combined_swap_masked_seq = self.sentences[comp_sentence_idx][2] + self.sentences[cur_sentence_idx][2]
                        combined_swap_label_seq = self.sentences[comp_sentence_idx][3] + self.sentences[cur_sentence_idx][3]
                        first_seq_length = len(self.sentences[comp_sentence_idx][1])
                        orig_span2_start = orig_span[1] + first_seq_length
                        orig_span2_end = orig_span[2] + first_seq_length
                        swap_span2_start = swap_span[1] + first_seq_length
                        swap_span2_end = swap_span[2] + first_seq_length
                    data = [
                        orig_span1_start, 
                        orig_span1_end, 
                        orig_span2_start, 
                        orig_span2_end,
                        combined_orig_masked_seq,
                        combined_orig_label_seq,
                        swap_span1_start, 
                        swap_span1_end, 
                        swap_span2_start, 
                        swap_span2_end,
                        combined_swap_masked_seq,
                        combined_swap_label_seq,
                        label
                    ]
                    self.out_data.append(data)
                orig_span = Span(cur_sentence_idx, orig_span[1], orig_span[2])
                swap_span = Span(cur_sentence_idx, swap_span[1], swap_span[2])
                if id not in self.references:
                    self.references.append(id)
                self.mentions[id].append((orig_span, swap_span))



    def resolve(document, create_training=False):
        self.document = document
        
        for sentence in self.document:
            orig_spans, swap_spans = data_utils.get_sentence_spans(self.tokenizer, sentence):
            sequence_versions = data_utils.get_tokenized_versions(self.tokenizer, sentence)
            for orig_spans in orig_spans

            
             
        


    def evaluate(mention)

