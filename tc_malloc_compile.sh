export CC=icx
export CXX=icpx
export CFLAGS="-O3 -xHost -qopenmp -ipo -fp-model=fast"
export CXXFLAGS="-O3 -xHost"

#spack load intel-oneapi-compilers@2025.1.1/py4jsac
#./configure --prefix=/usr/local --disable-cpu-profiler --disable-heap-profiler --disable-heap-checker --disable-debugalloc --enable-minimal --with-tcmalloc-pagesize=32
#make -j20
#make install