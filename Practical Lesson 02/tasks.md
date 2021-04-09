# Parallel Algorithms 2 - Practical Lessons

## LESSON 2
### Prerequisites
* knowledge of C++
* [download the project template with all additional libraries for further usage]()
### Topics and Tasks
* Allocate the HOST memory that will represent two M-dimensional vectors (A, B) and fill them with some values, where M is big enough. However,
 start with a small M to see reasonable outputs during debugging.
* Allocate the DEVICE memory to be able to copy data from HOST.
* Allocate the DEVICE memory to store an output M-dimensional vector C.
* Create a kernel that sums scalar values such that  $C_i = A_i + B_i$, where $i\in <0,m-1>$
* Allocate the HOST memory that will represent 2 M-dimensional vectors $A=[a_0,...,a_{m-1}]$, $B=[b_0,...,b_{m-1}]$ and fill them with some values.
* Allocate the DEVICE memory to be able to copy data from HOST.
* Allocate the DEVICE memory to store output M-dimensional vectors $C=[c_0,\dots, c_{m-1}]$.
* Create a kernel that sums the vectors, such that $C_i = A_i + B_i$.
* THINK ABOUT THE VARIANTS OF YOUR SOLUTION, CONSIDER THE PROS AND CONS.
