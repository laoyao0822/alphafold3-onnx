export CC=icx
export CXX=icpx
export CFLAGS="-O3 -xHost -qopenmp -ipo -fp-model=fast"
export CXXFLAGS="-O3 -xHost"

#./configure --prefix=/usr/local --disable-cpu-profiler --disable-heap-profiler --disable-heap-checker --disable-debugalloc --enable-minimal --with-tcmalloc-pagesize=128
#make -j20
#make install