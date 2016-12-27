# USAGE
`Note`: This is for running the code on the refactor branch.

## How to Build
To build this project, we suggest getting CMake along with Hunter (optional). If you operating system does not support this or you decide not to install the Hunter package manager, use Docker or install the libraries needed (mainly `HDF5`). 

Assuming that you checked out the project into `$SRCDIR` do
```{.sh}
cd $SRCDIR
mkdir build
cd build
cmake $SRCDIR [-DCONFIG_USE_HUNTER=OFF] [-DCMAKE_BUILD_TYPE=Debug]
```
By default, the flags set in `CMakeLists.txt`will use Hunter and will not set the Debug mode. If your system already contains the dependencies (like HDF5 and ZLib) you do not need the hunter package manager. Turning off the Hunter tag will speed up the compilation. If you would like to debug the code or perform any profiling, set the builf type tag to 'Debug'.

Once CMake has been run, a `Makefile` is generated so you can then perform `make` to build the project.

## How to Test 
While testing on a local machine, first check if your system's GPU supports CUDA. If the system does not support CUDA then the parallel code cannot be tested. 

2, 10, 100 and 10,000 queries are provided in `data/test2.hdf5`, `data/test10.hdf5`,  `data/test100.hdf5` and `data/testfull.hdf5` in the [`data`][data] folder. Make sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.
Otherwise, run th following command:
```{.sh}
./ece408 ../data/<data-file> ../data/<model-file> <batch size> 
```
Example:
```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
```

### Profiling
Profiling can be performed using `nvprof`. Run the following commands after you build the project:

```yaml
nvprof --cpu-profiling on --export-profile timeline.nvprof -- ./ece408 /src/data/test10.hdf5 /src/data/model.hdf5 10

nvprof --cpu-profiling on --export-profile analysis.nvprof -- ./ece408 /src/data/test10.hdf5 /src/data/model.hdf5 10
```

You could change the input and test datasets. This will output two files `timeline.nvprof` and `analysis.nvprof` which can be viewed using the `nvvp` tool (by performing a `file>import`).

## Most Optimal Version
[https://github.com/kawaiirice/patastula/tree/refactor](https://github.com/kawaiirice/patastula/tree/refactor)

[data]: https://github.com/kawaiirice/patastula/tree/refactor/data
