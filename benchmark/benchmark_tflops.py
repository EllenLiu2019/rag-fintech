import torch
import time


def benchmark_gemm(M=16384, N=16384, K=16384, num_iterations=50, dtype=torch.float16):
    """
    测量 GPU 在执行大矩阵乘法时的真实 TFLOPS
    公式: C[M, N] = A[M, K] @ B[K, N]
    """
    device = torch.device("cuda:0")  # 默认在 GPU 0 上测试单卡算力

    # 打印基础硬件信息
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[{gpu_name}] 算力基准测试启动")
    print(f"矩阵维度: M={M}, N={N}, K={K}")
    print(f"数据精度: {dtype}")

    # 1. 在显存中分配随机矩阵
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # 2. Warm-up (预热阶段)
    # 预热极其重要，用于唤醒 GPU 从闲置低频状态进入最高 Boost 频率，并完成 CUDA 上下文初始化
    print("正在执行 Warm-up 预热...")
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()  # 阻塞 CPU，确保 GPU 端上的预热指令全部执行完毕

    # 3. 正式压测阶段
    print(f"开始连续执行 {num_iterations} 次矩阵乘法...")
    start_time = time.time()

    for _ in range(num_iterations):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()  # 必须再次同步，否则 CPU 计时器会提前结束
    end_time = time.time()

    # 4. 统计与计算 TFLOPS
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations

    # 浮点运算量计算:
    # 结果矩阵 C 中有 M * N 个元素。
    # 计算每一个元素需要 K 次乘法和 K 次加法（如果使用 FMA 指令则视为 2 次操作）。
    # 单次矩阵乘法的总 FLOPs = 2 * M * N * K
    flops_per_iteration = 2.0 * M * N * K

    # 转换为 TFLOPS (Tera = 10^12)
    tflops = (flops_per_iteration / avg_time_per_iter) / (10**12)

    print("-" * 40)
    print(f"单次 GEMM 平均耗时: {avg_time_per_iter * 1000:.2f} 毫秒")
    print(f"实测物理算力: {tflops:.2f} TFLOPS")
    print("-" * 40)


if __name__ == "__main__":
    # 确保当前环境能够调用 CUDA
    assert torch.cuda.is_available(), "未检测到 CUDA 环境"

    # 默认使用 FP16 精度进行测试，这与 LLM 推理的计算量级完全对齐
    benchmark_gemm(M=16384, N=16384, K=16384, num_iterations=100)

# [NVIDIA H20] 算力基准测试启动
# 矩阵维度: M=16384, N=16384, K=16384
# 数据精度: torch.float16
# 正在执行 Warm-up 预热...
# 开始连续执行 100 次矩阵乘法...
# ----------------------------------------
# 单次 GEMM 平均耗时: 63.12 毫秒
# 实测物理算力: 139.35 TFLOPS
# ----------------------------------------
