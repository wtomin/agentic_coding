# Run conversion

## Input files $ARGUMENTS

Generate a complete PRP for vode conversion project.

## Process

1. Codebase Analysis

- Search for similar features/patterns in the codebase
- Identify files to reference in PRP
- Note existing conventions to follow
- Check test patterns in the example test script

2. Analyse code conversion task
- list the apis (functions) to be converted from MindSpore to Torch, and check if they are existent in MindSPore. If not, write equivalent api (function) using MindSpore.
- list the arguments in the input configuration file. Some arguments with integer default values can be lowered to minize the model size to be verified. List the possible argument names and their update values.

## Output
Save as: PRPs/{process-name}.md