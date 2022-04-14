# GeodesyPlus: Inverse Design Tool for Asymmetrical Self-Rising Surfaces with Color Texture


This is the code for the paper: 

**<a href="https://dl.acm.org/doi/10.1145/3424630.3425420">Inverse Design Tool for Asymmetrical Self-Rising Surfaces with Color Texture</a>**


## Dependency
- libigl
- eigen3
- ShapeOp 
- build-essential
- xorg-dev
- libgl1-mesa-dev


## Build

```bash
git submodule update --init
mkdir build; cd build; cmake ..; make
```

## Run
```bash
./geodesy ../dat/bear/bear.graph
```

