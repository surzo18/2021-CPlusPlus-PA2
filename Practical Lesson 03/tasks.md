# Parallel Algorithms 2 - Practical Lessons

## LESSON 3
### Prerequisites
* CUDA - memory allocation, page-locked memory
* [download the project template with all additional libraries for further usage](https://vsb.sharepoint.com/sites/PAII/Class%20Materials/Templates/cuda11_2-VS2019.zip)
### Topics and Tasks
* Create a column matrix $M[mRows\times mCols]$ containing the numbers 0 1 2 3 $\dots$
* The data should be well alligned in the page-locked memory.
* The matrix should be filled in CUDA kernel.
* You must use a Pitch CUDA memory with appropriate alignment. 
* 2D grid of 2D blocks of size 8x8 must be used to process data.
* Increment the values of the matrix.
* Finally, copy the matrix to HOST using cudaMemcpy2D function.