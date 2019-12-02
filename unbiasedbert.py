import torch 
from torch import nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM

PRETRAINED_WEIGHTS_SHORTCUT = 'bert-base-uncased'

HIDDEN_STATE_SIZE = 768

class SpanPairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        span_embedding_size = 512
        self.span_embedding = nn.Sequential(
            nn.Linear(3 + HIDDEN_STATE_SIZE*2, span_embedding_size),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(span_embedding_size*2, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )


    def forward(self, 
                span1_start_pos,
                span1_end_pos,
                span1_len,
                hidden_state_start_span1, 
                hidden_state_end_span1,
                span2_start_pos, 
                span2_end_pos, 
                span2_len,
                hidden_state_start_span2, 
                hidden_state_end_span2):
        s1 = torch.cat([span1_start_pos.float().unsqueeze(-1), 
                        span1_end_pos.float().unsqueeze(-1), 
                        span1_len.float().unsqueeze(-1), 
                        hidden_state_start_span1, 
                        hidden_state_end_span1],
                       dim=1)
        s2 = torch.cat([span2_start_pos.float().unsqueeze(-1), 
                        span2_end_pos.float().unsqueeze(-1), 
                        span2_len.float().unsqueeze(-1), 
                        hidden_state_start_span2, 
                        hidden_state_end_span2],
                       dim=1)
        e1 = self.span_embedding(s1)
        e2 = self.span_embedding(s2)
        out = self.classifier(torch.cat([e1, e2], dim=1))
        return out

class UnBiasedBert(nn.Module):
    def __init__(self, bert_for_masked_lm_instance=None):
        super().__init__()
        if bert_for_masked_lm_instance is None:
            self.bert_lm = BertForMaskedLM.from_pretrained(PRETRAINED_WEIGHTS_SHORTCUT)
        else:
            self.bert_lm = bert_for_masked_lm_instance 
        self.span_pair_classifier = SpanPairClassifier()

    def sentence_score(self, lm_output, sentence_ids):
        log_lm = lm_output.log_softmax(1)
        token_scores = log_lm.gather(2, sentence_ids.unsqueeze(-1)).squeeze(-1)
        scores = token_scores.sum(1)
        return scores

    def binary_bias_probe(self, original_sequence, swapped_sequence, orig_label_sequence, swapped_label_sequence):
        """ Returns tensor indicating presence of bias towards original """
        # seqs = torch.cat([original_sequence, swapped_sequence], dim=0)
        # output = self.bert_lm.forward(seqs)[0]
        # orig_output = output[:len(original_sequence)]
        orig_output = self.bert_lm.forward(original_sequence)[0]
        orig_score = self.sentence_score(orig_output, orig_label_sequence)
        #swapped_output = output[len(original_sequence):]
        swapped_output = self.bert_lm.forward(swapped_sequence)[0]
        swapped_score = self.sentence_score(swapped_output, swapped_label_sequence)
        return orig_score - swapped_score

    def forward(self, 
                orig_span1,
                orig_span2,
                original_probe_sequence, 
                original_label_sequence, 
                swap_span1,
                swap_span2,
                swapped_probe_sequence,
                swapped_label_sequence,
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
        use_orig_or_swap_only: One of {'orig', 'swap', None}. 
            None means use debias method. 
            'orig' means only use original
            'swap' means only use swapped

        All inputs are batched
        """
        if use_orig_or_swap_only is None:
            bias = self.binary_bias_probe(original_probe_sequence,
                                          swapped_probe_sequence,
                                          original_label_sequence,
                                          swapped_label_sequence)
            use_swapped = bias > 0
            use_swapped = use_swapped.unsqueeze(-1)
            seq = torch.where(use_swapped, swapped_label_sequence, original_label_sequence)
            span1 = torch.where(use_swapped, swap_span1, orig_span1)
            span2 = torch.where(use_swapped, swap_span2, orig_span2)
        elif use_orig_or_swap_only == 'orig':
            bias = None
            use_swapped = torch.zeros(original_label_sequence.size(0), 1).bool()
            seq = original_label_sequence
            span1 = orig_span1
            span2 = orig_span2
        elif use_orig_or_swap_only == 'swap':
            bias = None 
            use_swapped = torch.ones(original_label_sequence.size(0), 1).bool()
            seq = swapped_label_sequence
            span1 = swap_span1
            span2 = swap_span2

        with torch.no_grad():
            hidden_states = self.bert_lm.bert(seq)[0]

        batch_indices = torch.arange(0, seq.size(0))

        s1_start = span1[:,0]
        s1_end = span1[:,1]
        s1_len = s1_end - s1_start
        s2_start = span2[:,0]
        s2_end = span2[:,1]
        s2_len = s2_end - s2_start
        s1_start_state = hidden_states[batch_indices, s1_start]
        s1_end_state = hidden_states[batch_indices, s1_end]
        s2_start_state = hidden_states[batch_indices, s2_start]
        s2_end_state = hidden_states[batch_indices, s2_end]
        out = self.span_pair_classifier(s1_start, s1_end, s1_len, s1_start_state, s1_end_state,
                                        s2_start, s2_end, s2_len, s2_start_state, s2_end_state)
        return out, bias

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS_SHORTCUT)
    return tokenizer

def load_model():
    model = UnBiasedBert()
    return model

def test_model(model, tokenizer):
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return input_ids, last_hidden_states

