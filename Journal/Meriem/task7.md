# Project Updates

## Initial Training Script

- Jonas initially wrote the training script.

## Testing and Debugging

- I tested and debugged the training script, making necessary fixes until it worked correctly for the Cartpole environment (see commits on July 18). Some of these changes included:
  - Checking to ensure there are enough samples in the memory before returning a batch, preventing errors when the buffer is still being filled.
  - Setting input based on the environment's observation.
  - Adding a learning rate parameter for the optimizer to customize it.
  - Checking if there are enough mini-batches before proceeding.
  - Correctly evaluating the exploration condition and handling the Q-network input and output properly (converting `self.preprocessed` to a PyTorch tensor, adding a batch dimension, and detaching the output before converting it to a NumPy array for action selection).
  - Changing the extraction from the mini-batches.
  - Saving the network and evaluating it.
  - Handling `None` values for `target_action_values` in the loss function, using appropriate data types for states and actions, and reshaping states to `(batch_size, input_dim)` to ensure compatibility with the network.
  
## Model Evaluation

- The first version of the model evaluation in evaluation.py, including saving and loading networks in Agent.py. (see commits on July 18)
  
## Cartpole Training

- Trained the model with Jonas for Cartpole but initially did not achieve a reward higher than 9.
  
## Additional Changes and Final Version of Training Script (for Cartpole)

Since we didn't achieve higher reward than 9, I had to make some other changes to the training script:
  - The final version of the training script was committed on July 19 (`training.py`).
  - Changed the exploration strategy to an exponential decay for the exploration rate, which gradually reduces over time.
  - Modified the logging.
  - Evaluation the model and saved it every 100 episodes.

## Results

- Collaborated with Jonas to find the best training parameters, achieving a reward range of 500-600 for Cartpole.

## Other Models and Fixes

- Trained the Minesweeper model with Jonas.
- Fixed a minor bug in `videos.py`.
- Introduced a new reward function for Flappy Bird.
- Trained the Flappy Bird model using the updated reward function.
