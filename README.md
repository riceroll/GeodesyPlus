##### Compilation

```bash
git submodule update --init
mkdir build; cd build; cmake ..; make
```

##### Run
```bash
./geodesy ../dat/bear/bear3.graph
```

##### Dependency
- libigl
- eigen3
- ShapeOp	

##### Partial Compilation
compile only libraries without main.cpp
```bash
cmake .. -DMAIN=OFF; make -j8
```