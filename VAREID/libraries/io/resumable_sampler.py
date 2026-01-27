import torch
from torch.utils.data import Sampler

class ResumableSampler(Sampler):
    """
    A Sampler that can start from a specific index in a deterministic random permutation.
    
    Args:
        data_source (Dataset): The dataset to sample from.
        seed (int): Seed for random number generation to ensure the same order on resume.
        start_index (int): The number of *samples* (not batches) to skip.
    """
    def __init__(self, data_source, seed=None, start_index=0):
        self.data_source = data_source
        self.seed = seed
        self.start_index = start_index
        
        # Randomized
        if seed:
            # Generate the full list of indices deterministically
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)
            
            # Create the full random permutation of indices
            self.full_indices = torch.randperm(len(self.data_source), generator=self.generator).tolist()
       
        # Sequential
        else:
            self.full_indices = torch.arange(len(self.data_source)).tolist()

    def __iter__(self):
        # Yield indices starting from the offset
        # This effectively 'fast-forwards' the dataloader
        return iter(self.full_indices[self.start_index:])

    def __len__(self):
        # Return the remaining length
        return len(self.data_source) - self.start_index