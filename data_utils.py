
from collections import defaultdict

import conll 


class SpecialChars:
    MASK = '[MASK]'
    CLS = '[CLS]'
    SEP = '[SEP]'
    PAD = '[PAD]'
    UNK = '[UNK]'


def sentence_parts_to_masked_sequence(tokenizer, sentence_parts, mask_indicators):
    """
    tokenizer: tokenizer to encode with
    sentence_parts: list of strings that make up the sentence
    mask_indicators: list of bools indicating whether masking should be used for that part
    """
    CLS = tokenizer.encode(SpecialChars.CLS)[0]
    SEP = tokenizer.encode(SpecialChars.SEP)[0]
    mask_token = tokenizer.encode(SpecialChars.MASK)
    tokenized_parts = [tokenizer.encode(part) for part in sentence_parts]
    masked_sequence = [CLS]
    unmasked_sequence = [CLS]
    for part, mask_flag in zip(tokenized_parts, mask_indicators):
        masked_sequence += mask_token * len(part) if mask_flag else part
        unmasked_sequence += part
    masked_sequence += [SEP] 
    unmasked_sequence += [SEP] 
    return masked_sequence, unmasked_sequence

def get_tokenized_versions(tokenizer, sentence, gender_swap_format=True):
    sentence_parts = []
    swapped_sentence_parts = []
    mask_indicators = []
    prev_part_type = "none"
    for word_conll in sentence:
        word = conll.word(word_conll)
        alt_word = conll.alt_word(word_conll) if gender_swap_format else '-'
        if alt_word != '-':
            cur_part_type = "swap"
        else:
            cur_part_type = "noswap"
        if cur_part_type != prev_part_type:
            prev_part_type = cur_part_type
            sentence_parts.append(word)
            swapped_sentence_parts.append(alt_word if cur_part_type == "swap" else word)
            mask_indicators.append(True if cur_part_type == "swap" else False)
        else: 
            sentence_parts[-1] += ' ' + word 
            swapped_sentence_parts[-1] += ' ' + (alt_word if cur_part_type == "swap" else word)
    
    orig_masked_sequence, orig_label_sequence = sentence_parts_to_masked_sequence(tokenizer, sentence_parts, mask_indicators)
    swap_masked_sequence, swap_label_sequence = sentence_parts_to_masked_sequence(tokenizer, swapped_sentence_parts, mask_indicators)
    return orig_masked_sequence, orig_label_sequence, swap_masked_sequence, swap_label_sequence

def get_sentence_spans(tokenizer, sentence, gender_swap_format=True):
    CLS = tokenizer.encode(SpecialChars.CLS)
    SEP = tokenizer.encode(SpecialChars.SEP)
    tokenized_sentence = CLS
    swap_tokenized_sentence = CLS
    orig_corefs = defaultdict(lambda: ([],[],[]))
    swap_corefs = defaultdict(lambda: ([],[],[]))
    order = 0
    for word_conll in sentence:
        coref = conll.coref(word_conll)
        if coref != '-':
            startindex = len(tokenized_sentence)
            startswapindex = len(swap_tokenized_sentence)
            tokenized_sentence += tokenizer.encode(conll.word(word_conll))
            alt_word = conll.word(word_conll) if not gender_swap_format or conll.alt_word(word_conll) == '-' else conll.alt_word(word_conll)
            swap_tokenized_sentence += tokenizer.encode(alt_word)
            endswapindex = len(swap_tokenized_sentence)
            endindex = len(tokenized_sentence)
            starts, ends, singles = conll.get_spans(word_conll)
            for start in starts:
                orig_corefs[start][0].append(startindex)
                swap_corefs[start][0].append(startswapindex)
                orig_corefs[start][2].append(order)
                swap_corefs[start][2].append(order)
                order += 1
            for end in ends:
                orig_corefs[end][1].append(endindex)
                swap_corefs[end][1].append(endswapindex)
            for single in singles:
                orig_corefs[single][0].append(startindex)
                orig_corefs[single][1].append(endindex)
                swap_corefs[single][0].append(startswapindex)
                swap_corefs[single][1].append(endswapindex)
                orig_corefs[single][2].append(order)
                swap_corefs[single][2].append(order)
                order += 1
    orig_coref_list = []
    swap_coref_list = []
    for id, spans in orig_corefs.items():
        for start, stop, order in zip(*spans):
            orig_coref_list.append((id, start, stop, order))
    for id, spans in swap_corefs.items():
        for start, stop, order in zip(*spans):
            swap_coref_list.append((id, start, stop, order))
    orig_coref_list.sort(key = lambda x: x[-1])
    swap_coref_list.sort(key = lambda x: x[-1])
    return orig_coref_list, swap_coref_list 


        




