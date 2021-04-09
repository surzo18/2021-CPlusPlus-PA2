# Parallel Algorithms 2 - Practical Lessons

## LESSON 4
### Prerequisites
* [download the template](https://vsb.sharepoint.com/sites/PAII/Class%20Materials/Templates/cuda11_2-VS2019.zip)
* CUDA - shared memory
### Topics and Tasks
* Lets have a simple particle system representing a set of positions of $N$ rain drops in the 3D space, where $N>=(1<<20)$.
* Create a suitable data representation of the mentioned set of rain drops.
* Lets have an array of 256 wind power plants that give 256 movement vectors. These movement vectors invoke changes of all rain drops positions in a second.
* Create a kernel that simulates the falling of rain drops.
* Just for sake of simplicity suppose that a single kernel call simulates one second in the simulated world. 