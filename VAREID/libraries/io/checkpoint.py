from pathlib import Path
import pickle
import os 

from torch.utils.data import DataLoader
from typing import Any, Iterable, Optional, Callable, Dict

from VAREID.libraries.io.resumable_sampler import ResumableSampler

StateGetter = Callable[[], Dict[str, Any]]

class DataLoaderCheckpointManager:
    """
    Specialized CheckpointManager that lazily initializes a PyTorch DataLoader
    only after checking for a resume point.
    """
    def __init__(self, 
                 dataset, 
                 state_getter: Callable[[], Dict[str, Any]],
                 checkpoint_interval: int,
                 save_path: str,
                 batch_size: int,
                 num_workers: int,
                 collate_fn=None,
                 shuffle=True):
        
        # Store ingredients to build the loader later
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        
        self._state_getter = state_getter
        self._interval = checkpoint_interval
        self.save_path = save_path

        # IF DIRS TO SAVE PATH DON'T EXIST, MAKE THE PATH
        Path(os.path.dirname(self.save_path)).mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.iteration = 0 
        self.external_state = {}
        self.loader = None
        self.iterator = None

    def __enter__(self):
        """
        1. Checks for checkpoint.
        2. Calculates skip amount.
        3. Builds the DataLoader with the custom Sampler.
        """
        start_index_samples = 0

        # --- A. LOAD CHECKPOINT ---
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'rb') as f:
                    state = pickle.load(f)
                    self.iteration = state['iteration']
                    self.external_state = state.get('external_state', {})
                    
                    # Calculate how many SAMPLES to skip based on batches done
                    start_index_samples = self.iteration * self.batch_size
                    print(f"Resuming from batch {self.iteration} (skipping {start_index_samples} samples).")
        except Exception as e:
            print(f"No valid checkpoint found ({e}), starting from scratch.")

        # --- B. BUILD DATALOADER ---
        
        # Create the sampler with the calculated offset
        if self.shuffle:
            sampler = ResumableSampler(self.dataset, seed=42, start_index=start_index_samples)
        else:
            sampler = ResumableSampler(self.dataset, start_index=start_index_samples)
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
            pin_memory=True
        )
        
        self.iterator = iter(self.loader)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Save if at interval
            if self.iteration % self._interval == 0:
                self._save_checkpoint()

            # Get next batch from the internal loader
            batch = next(self.iterator)
            
            # Increment batch counter
            self.iteration += 1    
                
            return batch
        except StopIteration:
            raise

    def __len__(self):
        # Return length of the underlying loader (which shrinks as we resume)
        if self.loader:
            return len(self.loader)
        return len(self.dataset) // self.batch_size

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and exc_type is not StopIteration:
            print(f"CRASH DETECTED at batch {self.iteration}.")
        return False

    def _save_checkpoint(self):
        try:
            # Get latest data from user
            self.external_state = self._state_getter()
            
            state = {
                'iteration': self.iteration,
                'external_state': self.external_state
            }
            
            # Atomic save (write to temp then rename) prevents corruption
            temp_path = self.save_path + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(state, f)
            os.replace(temp_path, self.save_path)
            
            print(f"Checkpoint saved at batch {self.iteration}")
        except Exception as e:
            print(f"ERROR Saving Checkpoint: {e}")


class CheckpointManager:
    """
    Manages checkpointing for an iterable loop via a callback function .
    
    This class handles when to save (via checkpoint_interval) and calls
    the provided 'state_getter' function to fetch the application's
    current state right before persisting it.
    """
    def __init__(self, iterable: Iterable[Any], 
                 state_getter: StateGetter,
                 checkpoint_interval: int,
                 save_path: str = "checkpoint.pkl",
                 total_length: Optional[int] = None):
        
        self._raw_iterable = iterable 
        self.iterator = iter(iterable)
        
        # INIT INTERNALS
        self._state_getter = state_getter
        self._interval = checkpoint_interval
        self.save_path = save_path
        self._total_length = total_length if total_length is not None else len(iterable)

        # IF DIRS TO SAVE PATH DON'T EXIST, MAKE THE PATH
        Path(os.path.dirname(self.save_path)).mkdir(parents=True, exist_ok=True)
        
        # STATE INFO
        self.iteration = 0
        self.external_state = {}

    def __len__(self) -> int:
        """Returns the total length of the sequence."""
        return self._total_length

    def __iter__(self):
        """Returns self, allowing the object to be iterated over."""
        return self

    def __next__(self):
        """
        Fetches the next item, increments the counter, and conditionally saves a checkpoint.
        """
        try:
            # SAVE IF AT INTERVAL
            if self.iteration % self._interval == 0:
                self._save_checkpoint()

            self.current_item = next(self.iterator)
            self.iteration += 1
                
            return self.current_item
        
        # Iteration stopped
        except StopIteration:
            raise

    def __enter__(self):
        """Loads the last known-good checkpoint state upon entering the block."""
        try:
            if not os.path.exists(self.save_path):
                raise FileNotFoundError
                
            with open(self.save_path, 'rb') as f:
                # LOAD DATA
                state = pickle.load(f)
                self.iteration = state['iteration']
                self.external_state = state.get('external_state', {})
                print(f"Resuming from Checkpoint: Iteration {self.iteration}.")
                
                # Advance the iterator to the resume point
                if self.iteration > 0:
                    # If it's a list/range, we can slice it (Fast)
                    if isinstance(self._raw_iterable, (list, range)):
                        self.iterator = iter(self._raw_iterable[self.iteration:])
                    # If it's a generator (like df.iterrows), we must manually consume items (Safe)
                    else:
                        print(f"Fast-forwarding iterator by {self.iteration} steps...")
                        # We re-create the iterator from scratch to be safe
                        self.iterator = iter(self._raw_iterable)
                        # Burn the first N items
                        for _ in range(self.iteration):
                            try:
                                next(self.iterator)
                            except StopIteration:
                                break

        except FileNotFoundError:
            print("Starting from scratch: no checkpoint found.")
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Safe exiting. 
        If an error occurred, we DO NOT save, because the 
        current iteration likely didn't finish processing data.
        """
        if exc_type is not None and exc_type is not StopIteration:
            print(f"CRASH DETECTED at iteration {self.iteration}.")

        return False

    def _save_checkpoint(self):
        """
        Internal method that calls the user's getter and persists the state.
        """
        # USER CALLBACK FOR DATA
        try:
            latest_data_payload = self._state_getter()
            self.external_state = latest_data_payload
        except Exception as e:
            print(f"ERROR: State getter callback failed: {e}")
            return 
            
        # SAVE DATA
        state = {
            'iteration': self.iteration, 
            'external_state': self.external_state
        }
       
        # Atomic save (write to temp then rename) prevents corruption
        temp_path = self.save_path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(state, f)
        os.replace(temp_path, self.save_path)
            
        print(f"Checkpoint saved at iteration {self.iteration}")
