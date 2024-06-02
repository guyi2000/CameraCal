# Camera Cal

## Usage

- `OpenCV` **MUST** be installed (Environment `OpenCV_DIR` must be set before.)
- Just run shell below

### Windows

```shell
cd .
mkdir build && cd build
cmake ..
msbuild CameraCal.sln /p:Configuration=Release -m
```

### Linux

```shell
cd .
mkdir build && cd build
cmake ..
make -j8
```

- Then in `bin` directory, you can found the `exe` programs.
