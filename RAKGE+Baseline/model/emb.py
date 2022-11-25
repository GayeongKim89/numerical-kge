def load_model(self, path):
    """
    Function to load a saved model
    :param path: path where model is loaded
    :return:
    """
    state = torch.load(path)
    self.best_val_results = state['best_val']
    self.best_val_mrr = self.best_val_results['mrr']
    self.best_epoch = state['best_epoch']
    self.model.load_state_dict(state['model'])
    self.optimizer.load_state_dict(state['optimizer'])