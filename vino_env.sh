export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=21474836480

#大于此值的分配会导致堆栈跟踪被转储 到 stderr。
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=53687091200

#export TCMALLOC_PAGE_HEAP_RESERVE_BYTES=2147483648
#
export TCMALLOC_RELEASE_RATE=5
export LD_PRELOAD=/usr/local/lib/libtcmalloc_minimal.so.4:$LD_PRELOAD
export LD_PRELOAD=/root/anaconda3/envs/alphafold/lib/python3.11/site-packages/openvino/libs/libtbbbind_2_5.so.3:$LD_PRELOAD