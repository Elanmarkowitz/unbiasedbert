import os 

import torch.nn.functional as F 
from torch import optim 
import torch 

from tqdm.auto import tqdm 

CUDA_ENABLED = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (3,5)

def cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() and CUDA_ENABLED else tensor

class Trainer:
    def __init__(self, model):
        self.model = model 
        self.optim = optim.Adam(self.model.parameters())
    
    def train(self, train_dataloader, val_dataloader, debias_method='active', epochs=1, checkpoint_dir=None):
        self.model.zero_grad()
        if debias_method == 'active':
            use_orig_or_swap_only = None 
        elif debias_method == 'both':
            use_orig_or_swap_only = 'orig'
        elif debias_method == 'none':
            use_orig_or_swap_only = 'orig'
        else:
            raise Exception('Invalid debias method')

        cuda(self.model)

        pbar = tqdm(total=epochs*len(train_dataloader))
        pbar.update(0)

        cum_acc_loss = 0
        cum_bias_loss = 0

        for e in range(epochs):
            for i, batch in enumerate(train_dataloader):
                pbar.update(1)
                model_input = [cuda(t) for t in batch[:-1]]
                label = cuda(batch[-1])
                    
                out, bias = self.model(*model_input, use_orig_or_swap_only=use_orig_or_swap_only)

                accuracy_loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1).float())
                loss = accuracy_loss 
                cum_acc_loss += accuracy_loss.cpu().item()
                bias_loss = None
                if bias is not None:
                    bias_loss = bias.abs().mean()
                    loss += bias_loss
                    cum_bias_loss += bias_loss.cpu().item()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i % 1000 == 0:
                    pbar.write('accuracy_loss: {}, bias_loss: {}'.format(cum_acc_loss/1000, cum_bias_loss/1000 if bias is not None else 'NA'))
                    cum_acc_loss = 0
                    cum_bias_loss = 0

            with torch.no_grad():
                num_correct = 0
                total_predicted = 0

                for batch in val_dataloader:
                    model_input = batch[:-1]
                    label = batch[-1]
                        
                    out, bias = self.model(*model_input, use_orig_or_swap_only=use_orig_or_swap_only)

                    out_label = torch.where(torch.sigmoid(out) > 0.5, 1, 0)

                    num_correct += (out_label == label).sum().item()
                    total_predicted += out_label.size(0)

                    pbar.write('Epoch {}/{}, Val pairwise accuracy: {}'.format(e+1, epochs, num_correct/total_predicted))

            if debias_method == 'both':
                use_orig_or_swap_only = 'orig' if use_orig_or_swap_only == 'swap' else 'swap'

            if checkpoint_dir:
                checkpoint_dict = {
                    'model': self.model,
                    'state_dict': self.model.state_dict()
                }
                filename = 'checkpoint-{}-{}.pkl'.format(debias_method, e + 1)
                torch.save(checkpoint_dict, os.path.join(checkpoint_dir, filename))

        