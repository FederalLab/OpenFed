## Build OpenFed from source

### Build on Linux or macOS

After cloning the repo with

```bash
git clone https://github.com/FederalLab/OpenFed.git
cd OpenFed
```

You can either

- install the lite version

  ```bash
  pip install -e .
  ```

- install the full version

  ```bash
  OPENFED_WITH_THIRD_PARTY=1 pip install -e .
  ```

If you are on macOS, add the following environment variables before the installing command.

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++'
```

e.g.,

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' OPENFED_WITH_THIRD_PARTY=1 pip install -e .
```

### Build on Windows

Building OpenFed on Windows is a familiar with that on Linux.
