export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=21474836480
#export OMP_PROC_BIND=CLOSE
export OMP_SCHEDULE=STATIC
#大于此值的分配会导致堆栈跟踪被转储 到 stderr。
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=53687091200
#export OMP_PROC_BIND=CLOSE
#export GOMP_CPU_AFFINITY="0-95"
#export GOMP_CPU_AFFINITY="0-95"
#export TCMALLOC_PAGE_HEAP_RESERVE_BYTES=2147483648
#
export TCMALLOC_RELEASE_RATE=5
export LD_PRELOAD=/usr/local/lib/libtcmalloc_minimal.so.4:$LD_PRELOAD
export LD_PRELOAD=/root/ASC25F/AF3/alphafold_ASC25F_conda_env/lib/python3.11/site-packages/openvino/libs/libtbbbind_2_5.so.3:$LD_PRELOAD

export GLOO_SOCKET_IFNAME=ibp50s0

#export LD_PRELOAD=/root/tools/spack/opt/spack/linux-icelake/intel-oneapi-compilers-2025.1.1-py4jsacn4cok2a455wfxuqjaomm2pgiq/compiler/latest/lib/libiomp5.so:$LD_PRELOAD
#export KMP_AFFINITY=granularity=fine,compact
