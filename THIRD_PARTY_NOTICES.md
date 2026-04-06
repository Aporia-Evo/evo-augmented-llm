This project vendors a minimal subset of TensorNEAT source code under [src/tensorneat](/mnt/c/Users/joach/NEAT.wsl.projekt/src/tensorneat).

Source:
- https://github.com/EMI-Group/tensorneat

Reason:
- The upstream top-level package imports RL dependencies that are unnecessary for this XOR V1 and significantly increase local setup complexity.

The vendored code is used only for the NEAT/common/genome components required by this repository.

