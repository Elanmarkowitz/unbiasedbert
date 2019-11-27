import torch 
from torch import nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM

PRETRAINED_WEIGHTS_SHORTCUT = 'bert-base-uncased'

def load_base():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS_SHORTCUT)
    model = BertForMaskedLM.from_pretrained(PRETRAINED_WEIGHTS_SHORTCUT)
    return model, tokenizer

def test_model(model, tokenizer):
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return input_ids, last_hidden_states


class SpanPairClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                span1_start_pos,
                span1_end_pos,
                span2_start_pos, 
                span2_end_pos, 
                hidden_state_start_span1, 
                hidden_state_end_span1,
                hidden_state_start_span2, 
                hidden_state_end_span2):
        pass

class UnBiasedBert(BertForMaskedLM):
    def __init__(self):
        super().__init__()
        self.span_pair_classifier = SpanPairClassifier()

    def sentence_score(self, lm_output, sentence_ids):
        log_lm = lm_output.log_softmax(token_scores, 1)
        token_scores = log_lm.gather(2, sentence_ids.unsqueeze(2)).squeeze(2)
        scores = token_scores.sum(1)
        return scores

    def binary_bias_probe(self, original_sequence, swapped_sequence, orig_label_sequence, swapped_label_sequence):
        """ Returns tensor indicating presence of bias towards original """
        with torch.no_grad():
            orig_output = super().forward(original_sequence)
            orig_score = self.sentence_score(orig_output, orig_label_sequence)
            swapped_output = super().forward(swapped_sequence) 
            swapped_score = self.sentence_score(swapped_output, swapped_label_sequence)
        return orig_score > swapped_score

    def forward(self, original_probe_sequence, 
                swapped_probe_sequence, 
                original_label_sequence, 
                swapped_label_sequence,
                orig_span1,
                orig_span2,
                swap_span1,
                swap_span2,
                use_orig_or_swap_only=None):
        """
        original_probe_sequence: tokenized sequences of original sentence, with masking of swap words
        swapped_probe_seqencue: tokenized sequences of swapped sentence, with masking of swap words
        original_label_sequence: tokenized sequences of original sentence, with no masking
        swapped_label_seqencue: tokenized sequences of swapped sentence, with no masking
        orig_span1: start and end index of first span in orig sequence
        orig_span2: start and end index of second span in orig sequence
        swap_span1: start and end index of first span in swapped sequence
        swap_span2: start and end index of second span in swapped sequence
        labels: 0 if spans are a coreference, 1 if spans are different references
        use_orig_or_swap_only: One of {'orig', 'swap', None}. 
            None means use debias method. 
            'orig' means only use original
            'swap' means only use swapped

        All inputs are batched
        """
        if use_orig_or_swap_only is None:
            seq_choice = self.binary_bias_probe(original_probe_sequence,
                                                swapped_probe_sequence,
                                                original_label_sequence,
                                                swapped_label_sequence)
        elif use_orig_or_swap_only == 'orig':
            seq_choice = torch.zeros(original_label_sequence.size(0), 1).bool()
        elif use_orig_or_swap_only == 'swap':
            seq_choice = torch.ones(original_label_sequence.size(0), 1).bool()
            
        seq = torch.where(seq_choice, swapped_label_sequence, original_label_sequence)
        span1 = torch.where(seq_choice, swap_span1, orig_span1)
        span2 = torch.where(seq_choice, swap_span2, orig_span2)

        
            
        