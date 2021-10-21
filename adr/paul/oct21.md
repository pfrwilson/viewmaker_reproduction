# Issues
- experiments performed with similar hyperparameters as authors for SIMCLR are not achieving the same performance.

# Ideas
- Not entirely clear what the author is doing with the projection head. Is the projection head kept with the encoder in the transfer learning task?
- Our augmentations library (src/utils/SimCLR_data_util.py) did not come from the author's code repository. TODO: dig through repository to see what the author is doing for expert views and whether there is any difference there.
- So far, we have not used weight decay in our loss function. We need to pre-train the model with weight decay. 
- We have not been using data augmentations during supervised pre-training. Data augmentations may help prevent overfitting.

