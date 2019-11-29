#torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in input], batch_first=True)

from torch.utils.data import DataLoader, Dataset

class DeBiasCorefDocumentDataset(Dataset):

    def __init__(self, document):
        


train_document_loaders = []
for doc in documents:

    


    dataloader = DataLoader()



# [orig_span1, orig_span2, orig_seq, mask_orig_seq, swap_span1, swap_span2, swap_seq, mask_swap_seq, label]