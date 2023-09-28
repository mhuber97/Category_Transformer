# Category Transformer

This implementation is modified from Andrew Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/model.py) in order to work with tabular data, containing continuous and categorical features.

Define the properties of your data in the TabularConfig class in ``tabular_config.py``:
- output_size: the number of different labels
- n_layer: how many decoder blocks do we need
- n_head: how many attention heads
- n_embd: size of the embedding presentation of all features
- n_features: number of non-categoric features encoded with a label encoder
- classification_weights: in a case of a non-balanced dataset, we can use those weights in order to train the model and weight few-occuring classes higher.
- embedding_config: here we define a list of dicts that captures information about the categorical features in the data. For each of the features we build a separate embedding. Each dict in the list we define `nr_classes` which captures the number of different labels that need to be encoded. Also we need to define `embedding_dimension` in order to define the number of dimensions of the embedding.


You can also use the `train.py` file and modify it to your needs in order to train the model. Replace the dataset with yours and run it!

Enjoy!