## Run Code Conversion

The goal is to convert the Pytorch python scripts to MindSpore python scripts and write the test script.

Take the following steps and complete the conversion task: 
1. Read the `output` folder. 

If `output` folder is empty, read all input files under `inputs` folder, convert the Pytorch scripts to MindSpore scripts.  

If `output` folder is not empty, they are partially converted MindSpore script. Skip reading `inputs` folder. Start converting the python scripts in `output` folder using diff and always ask for human evaluation.

2. For each converted file:

Syntax validation with Python AST
Import resolution check
Basic MindSpore API compatibility check
Cross-reference with conversion rules

3. Read all files under `examples` folder, take the `cohere2` as a reference of configuration file and the test script.

4. Given the configuration file in `output` folder, write a test script for the converted modeling file. You can always use smaller batch size, vocab_size, num_hidden_layers, hidden_size, and max_position_embeddings than the original configurations in order to initiate a small-sized model for testing.

## Inputs
Inside the `inputs` folder, there are two types of inputs:
1. Torch modeling script that defines the model architecture, there maybe a few modeling script to define a full model graph.
2. Torch configuration file that defines the configuration arguments and default values.

## Outputs:
There are expected to be have two types of outputs:

1. MindSpore modeling script that works the same as the Torch modeling script, the number of files and the file names should align with torch modeling script.
2. MindSpore test script that test if mindspore and torch model has the same inputs under the same inputs. 

If there are available mindspore script under `outputs` folder, regard it as a file partially converted from Pytorch to MindSpore. Take it as a starting file and complete the edits remaining. Use diff and always ask for human evaluation.


# Reference

In `examples`, there are examples about a single model named cohere2:
- `configuration_cohere2.py`: the configuration file of the model cohere2. It can be shared by both mindspore and pytorch model.
- `modeling_cohere2_torch.py`: the Torch modeling script that defines the model architecture of cohere2.
- `modeling_cohere2.py`: the MindSpore modeling script that works the same as the Torch modeling script.
- `test_modeling_cohere2.py`: the MindSpore test script for the model cohere2.


