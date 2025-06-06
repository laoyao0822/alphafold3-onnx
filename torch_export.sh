export LD_PRELOAD=/root/tools/spack/opt/spack/linux-icelake/intel-oneapi-compilers-2025.1.1-py4jsacn4cok2a455wfxuqjaomm2pgiq/compiler/latest/lib/libiomp5.so:$LD_PRELOAD
export LD_PRELOAD=/root/ASC25F/tcmalloc/lib/libtcmalloc_minimal.so.4:$LD_PRELOAD

#export ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY="cpu:10240"
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=91860
export KMP_BLOCKTIME=10
export KMP_AFFINITY=granularity=fine,compact,1,0
#export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=10240000000000000000
#export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=134217728

#绑定 分配给线程缓存的字节总数
export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=21474836480

#大于此值的分配会导致堆栈跟踪被转储 到 stderr。
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=53687091200

#export TCMALLOC_PAGE_HEAP_RESERVE_BYTES=2147483648
#
export TCMALLOC_RELEASE_RATE=4

export GLOO_SOCKET_IFNAME=ibp50s0
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
#export TCMALLOC_MMAP_THRESHOLD=1048576  # 1MB