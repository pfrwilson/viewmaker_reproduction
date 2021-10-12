
### viewmaker.py

#### decisions
I didn't find an `affine` parameter in the [tensorflow docs](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/InstanceNormalization) for InstanceNorm, and the authors originally used this parameter in their computation, as seen [here](https://github.com/alextamkin/viewmaker/blob/d9a7d4b05ac5126fe348c8c5217877ebcff7e2d7/src/models/viewmaker.py#L50) and [here](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html). I also didn't find any mention of this in the [original paper](https://arxiv.org/pdf/1607.08022.pdf) for instance norm so it may not be necessary to acheive the improved performance that it adds.

[This line](https://github.com/alextamkin/viewmaker/blob/d9a7d4b05ac5126fe348c8c5217877ebcff7e2d7/src/models/viewmaker.py#L74) uses `m.weight.data` to initalize all parameters to a normal distribution. However, I can't find much documentation for the weight class (it looks like it's just a matrix of weights [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html), and as far as I know matrices don't have a `data` parameter. Will use `m.weights` for now, and try to rectify if I run into any issues.

#### todos/ concerns/ issues

TODO: need to determine how to initialize biases to 0 in the `zero_init` method. Afaik, the only way to get a layer's weights is using `m.weights`, as documented [here](https://www.tensorflow.org/lattice/api_docs/python/tfl/layers/Linear), which seems to hold both weights and biases. For now, all the weights and biases are being initialized with the normal distribution, but this needs to be rectified to exclude the biases, which have to be initialized to 0. An idea for how to do this can be found [here](https://stackoverflow.com/questions/41821502/zero-initialiser-for-biases-using-get-variable-in-tensorflow).

Not sure if we should be using `tf.keras.layers.Dense` for consistency or `tfl.layers.Linear` in the `zero_init` method ([reference](https://stackoverflow.com/questions/66626700/difference-between-tensorflows-tf-keras-layers-dense-and-pytorchs-torch-nn-lin)).

I can't find a 2D inverse discrete cosine transform II function in tensorflow. I've only been able to find the 1D version [here](https://www.tensorflow.org/api_docs/python/tf/signal/idct).

#### to do tomorrow:
- configure libraries
- check sample inputs, fix any bugs that may come up
- compare outputs of sample inputs to original viewmaker. resolve issues or inconsistencies if any come up.
- start training? 