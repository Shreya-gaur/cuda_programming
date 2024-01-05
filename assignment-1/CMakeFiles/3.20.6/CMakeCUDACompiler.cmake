set(CMAKE_CUDA_COMPILER "/apps/scholar/cuda-toolkit/11.8.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.8.89")
set(CMAKE_CUDA_DEVICE_LINKER "/apps/scholar/cuda-toolkit/11.8.0/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/apps/scholar/cuda-toolkit/11.8.0/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "9.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/apps/scholar/cuda-toolkit/11.8.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/apps/scholar/cuda-toolkit/11.8.0")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/apps/scholar/cuda-toolkit/11.8.0")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/apps/scholar/cuda-toolkit/11.8.0/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/apps/scholar/cuda-toolkit/11.8.0/targets/x86_64-linux/lib/stubs;/apps/scholar/cuda-toolkit/11.8.0/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/apps/spack/scholar/fall20/apps/zlib/1.2.11-gcc-4.8.5-64vg6e4/include;/apps/spack/scholar/fall20/apps/mpc/1.1.0-gcc-4.8.5-eogmmao/include;/apps/spack/scholar/fall20/apps/mpfr/3.1.6-gcc-4.8.5-nsgsjfy/include;/apps/spack/scholar/fall20/apps/gmp/6.1.2-gcc-4.8.5-zn55wh7/include;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/include/c++/9.3.0;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/include/c++/9.3.0/x86_64-pc-linux-gnu;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/include/c++/9.3.0/backward;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/lib/gcc/x86_64-pc-linux-gnu/9.3.0/include;/usr/local/include;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/include;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/lib/gcc/x86_64-pc-linux-gnu/9.3.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/apps/scholar/cuda-toolkit/11.8.0/targets/x86_64-linux/lib/stubs;/apps/scholar/cuda-toolkit/11.8.0/targets/x86_64-linux/lib;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/lib64;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/lib/gcc/x86_64-pc-linux-gnu/9.3.0;/lib64;/usr/lib64;/apps/spack/scholar/fall20/apps/gcc/9.3.0-gcc-4.8.5-3acpdic/lib;/apps/spack/scholar/fall20/apps/zlib/1.2.11-gcc-4.8.5-64vg6e4/lib;/apps/spack/scholar/fall20/apps/mpc/1.1.0-gcc-4.8.5-eogmmao/lib;/apps/spack/scholar/fall20/apps/mpfr/3.1.6-gcc-4.8.5-nsgsjfy/lib;/apps/spack/scholar/fall20/apps/gmp/6.1.2-gcc-4.8.5-zn55wh7/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/apps/cent7/xalt/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
