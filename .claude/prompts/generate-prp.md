# Create PRP for code conversion project

## TASK FILE $ARGUMENTS

Generate a complete PRP for code conversion project with the instruction in $ARGUMENTS. Ensure context is passed to the AI agent to enable self-validation and iterative refinement. Read the task file to understand what needs to be created, how the examples provided help, and any other considerations.


## Process

1. Codebase Analysis

- Identify files to reference in PRP
- Note existing conventions to follow
- Check test patterns in the example test script

2. Start code conversion task
- List the files be converted from Torch to MindSpore;
- Specify the input file path and the output file path;
- Make backup copies of the original files if necessary;
- Read corresponding examples in `examples` folder;
- Run automated conversion tasks and ask for human review;
- If there are no errors, the converted files are saved in the output directory;

The above process is repeated for each file. You many need a to-do list to keep track of the files to be converted.

3. Review Outputs
- Check the output files, and make sure they are correctly converted.
- If there are errors, fix them and re-run the process.
- If no errors, you should generate a conversion report.


## Output
Save as: PRPs/{process-name}.md