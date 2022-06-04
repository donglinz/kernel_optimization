# GPU Kernel implementation for common neural network operators
Author: Donglin Zhuang

# Benchmark
Softmax on shape FP32[128\*12\*128, 128], which is the softmax in BERT-Base model when batch size=128 and seq-len=128, 83% peak memory throughput on T4.

Layernorm on shape FP32[128\*128, 768], which is the layernorm in BERT-Base model when batch size=128 and seq-len=128, 83% peak memory throughput on T4.

GEMM on shape M,N,K=1024, 1024, 1024, 60% performance relative to cuBlas on SGEMM.