import pickle
import os 
from typing import Any, Iterable, Optional, Callable, Dict

# SIGNATE FOR CALLBACK GETTER FUNCTION
StateGetter = Callable[[], Dict[str, Any]]

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
            self.current_item = next(self.iterator)
            self.iteration += 1
            
            # SAVE IF AT INTERVAL
            if self.iteration % self._interval == 0:
                self._save_checkpoint()
                
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
                if isinstance(self._raw_iterable, (list, range)) and self.iteration > 0:
                    self.iterator = iter(self._raw_iterable[self.iteration:])

        except FileNotFoundError:
            print("Starting from scratch: no checkpoint found.")
            
        # Return self, so the user can access loaded state (trainer.external_state)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Safe exiting (pull and save current state).
        """
        if exc_type is not None and exc_type is not StopIteration:
            print(f"CRASH DETECTED at iteration {self.iteration}.")
            self._save_checkpoint()
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
        with open(self.save_path, 'wb') as f:
            pickle.dump(state, f)
            
        print(f"Checkpoint saved at iteration {self.iteration}")