# GPU-CoDaSHep-Princeton

This repository provides tutorials, demos, and analysis scripts for GPU programming, tailored for the Princeton CoDAS-HEP course. Materials are organized into modules covering CUDA basics, unified memory, streaming, profiling, and neural network demos.

## Repository Structure

- **01/** – Introductory CUDA C notebooks and exercises  
    - `AC_CUDA_C.ipynb`: Main CUDA C tutorial notebook  
    - Subfolders:  
        - `01-hello/` to `09-heat/`: Step-by-step CUDA examples (hello world, parallelism, indices, loops, memory allocation, error handling, vector addition, matrix multiplication, heat conduction)  
        - `edit/`: Editable files for exercises

- **02/** – Unified Memory tutorials  
    - `Unified Memory.ipynb`: Unified memory concepts and demos  
    - Subfolders: Vector addition, device properties, page faults, prefetching, SAXPY example, and editable files

- **03/** – Streaming and Visual Profiling  
    - `Streaming and Visual Profiling.ipynb`: Streaming and profiling notebook  
    - Subfolders: Vector addition, kernel initialization, prefetch checks, stream introduction, manual memory allocation, overlap transfer, n-body simulation, and editable files

- **04/** – Additional CUDA modules  
    - Contains further CUDA exercises and scripts

- **05/** – Advanced CUDA topics and performance analysis  
    - Exercises on optimizing memory access, shared memory usage, and occupancy

- **06/** – Neural Network Demos  
    - Simple neural network implementations and GPU acceleration examples

- **07/** – Analysis Scripts  
    - Data analysis, visualization, and benchmarking GPU code

- **08/** – Notebook Verification Tools  
    - Utilities for extracting code from notebooks and checking notebook integrity  
    - `verify_notebook.py`: Python script for verifying and extracting code from Jupyter notebooks

## Notebooks

Each module contains Jupyter notebooks (`.ipynb`) with explanations, code samples, and exercises. Topics include:

- CUDA programming basics
- Memory management (device, unified memory)
- Parallel algorithms (vector/matrix operations, heat conduction)
- Streaming and concurrency
- Profiling and performance analysis

## Scripts

- [`08/verify_notebook.py`](08/verify_notebook.py):  
    Analyze notebook structure, extract code cells to files, and report statistics on content and coverage.

## Credits

- The heat conduction CPU source code in `01/AC_CUDA_C.ipynb` is credited to [An OpenACC Example Code for a C-based heat conduction code](http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html) from the University of Houston.

## Getting Started

1. Clone the repository.
2. Open the notebooks in Jupyter or VS Code.
3. Follow the step-by-step exercises in each module.
4. Use the verification script to analyze or extract code from notebooks:
     ```sh
     python 08/verify_notebook.py <notebook_path>
     ```

## License

Refer to individual files and notebooks for licensing and attribution information.

---

For more details, explore the subfolders and notebooks in each module.