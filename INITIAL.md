# Inputs
Inside the `inputs` folder, there are two types of  inputs:
1. Torch modeling script that defines the model architecture, there may a few modeling script to define a full model graph.
2. Torch configuration file that defines the configuration arguments and default values.

# Outputs:
There are expected to be have two types of outputs:

1. MindSpore modeling script that works the same as the Torch modeling script, the number of files and the file names should align with torch modeling script.
2. MindSpore test script that test if mindspore and torch model has the same inputs under the same inputs. 

# Reference

In `examples`, there are examples:
- `configuration_cohere2.py`: the configuration file of the model cohere2. It can be shared by both mindspore and pytorch model.
- `modeling_cohere2_torch.py`: the Torch modeling script that defines the model architecture of cohere2.
- `modeling_cohere2.py`: the MindSpore modeling script that works the same as the Torch modeling script.
- `test_modeling_cohere2.py`: the MindSpore test script for the model cohere2.

The files above can be reference files when converting a new model.