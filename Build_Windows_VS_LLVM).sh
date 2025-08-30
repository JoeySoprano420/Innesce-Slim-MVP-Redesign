cmake -S . -B build -G "Visual Studio 17 2022" -DLLVM_DIR="C:\Program Files\LLVM\lib\cmake\llvm"
cmake --build build --config Release
