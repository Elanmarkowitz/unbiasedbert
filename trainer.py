class Trainer:
    def __init__(self, model):
        self.model = model 
    
    def train(self, train_dataloader, epochs=1):
        