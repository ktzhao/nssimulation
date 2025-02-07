# test_all.jl
#
# 本文件对 GridModule、DomainDecomposition、Communication 和 AdvancedParallel 模块进行单元测试与性能分析，
# 并根据硬件环境自动检测 GPU 可用性，若 GPU 不可用则跳过 GPU 相关测试。
#
# 运行命令示例：
#   julia --project test_all.jl

using Test
using Distributed
using BenchmarkTools
using LinearAlgebra
using Dates

# 如果 worker 数量不足，则添加（测试分布式部分）
if nprocs() < 2
    addprocs(2)
end

# 包含各模块文件
include("GridModule.jl")
include("DomainDecomposition.jl")
include("Communication.jl")
include("AdvancedParallel.jl")

# 导入本地模块（使用相对引用）
using .GridModule
using .DomainDecomposition
using .Communication
using .AdvancedParallel

##########################################################################
# 顶层全局变量定义（使用 const 声明）
##########################################################################
const limits = Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0))
const spacing = Dict(:x => 0.1, :y => 0.1, :z => 0.1)
const bc = Dict(:xlow => (:Dirichlet, 0.0), :xhigh => :Neumann)
const adaptive_params = Dict(:x => (0.4, 0.6, 0.01, 0.1))
const stretch_funcs = Dict{Symbol, Function}()  # 暂不使用
const custom_bc = Dict{Symbol, Function}()      # 暂不使用

# 创建全局网格，并预先初始化物理场字段（在 grid.physical_fields 中定义所需的键）
const grid = create_grid(coordinate_system = :cartesian, limits = limits, spacing = spacing,
                         bc = bc, adaptive_params = adaptive_params, stretch_funcs = stretch_funcs,
                         custom_bc = custom_bc)
grid.physical_fields[:density] = fill(0.0, 10)
grid.physical_fields[:pressure] = fill(0.0, 10)

##########################################################################
# 1. GridModule 测试
##########################################################################
@testset "GridModule Tests" begin
    @test grid.coordinate_system == :cartesian
    @test haskey(grid.coordinates, :x)
    @test length(grid.coordinates[:x]) ≥ 1

    # 测试边界条件函数：对一维场变量应用边界条件
    dummy_field = [1.0, 2.0, 3.0, 4.0]
    new_field = apply_boundary_conditions(copy(dummy_field), :xlow, grid)
    @test new_field[1] == 0.0  # xlow 应采用 Dirichlet 0.0

    # 测试物理场接口：初始化和更新
    field_data = Dict(:density => fill(1.0, 10), :pressure => fill(2.0, 10))
    println("field_data 类型：", typeof(field_data))  # 应输出 Dict{Symbol, Vector{Float64}}
    init_physical_fields!(grid, field_data)
    @test haskey(grid.physical_fields, :density)
    update_physical_field!(grid, :density, fill(3.0, 10))
    @test grid.physical_fields[:density][1] == 3.0

    # 测试配置读取（TOML 示例，使用内置字符串模拟文件内容）
    toml_str = """
    [simulation]
    title = "Test Simulation"
    dt = 0.01
    """
    open("test_config.toml", "w") do f
        write(f, toml_str)
    end
    config = read_config("test_config.toml")
    @test config[:simulation][:title] == "Test Simulation"
end

##########################################################################
# 2. DomainDecomposition 测试
##########################################################################
@testset "DomainDecomposition Tests" begin
    # 定义局部变量（不使用 const，因为在局部作用域中不支持 const 声明）
    overlap = (1, 1, 1)
    weights = Dict(:x => [1.0, 1.0], :y => [1.0, 1.0], :z => [1.0, 1.0])
    proc_dims = (2, 1, 1)  # 强制 x 方向分为 2 块
    domains = decompose_grid_3d(grid; overlap = overlap, proc_dims = proc_dims, weights = weights)
    @test length(domains) == prod(proc_dims)

    # 测试生成局部网格
    local_grid = get_local_grid(domains[1])
    @test local_grid.dims[:x] ≥ 1

    # 测试多级域分解：粗分解和细分解
    fine_domains = multi_level_decompose(grid, (2, 1, 1), (2, 1, 1);
                                         overlap = overlap, coarse_weights = weights, fine_weights = weights)
    @test all(d -> d.level == 2, fine_domains)
end

##########################################################################
# 3. Communication 测试
##########################################################################
@testset "Communication Tests" begin
    # 构造测试用的局部 3D 数组（含 ghost 区域）
    nx, ny, nz = 10, 10, 10
    ghost = (1, 1, 1)
    function create_dummy_field(val)
        return fill(val, nx + 2 * ghost[1], ny + 2 * ghost[2], nz + 2 * ghost[3])
    end
    # 创建两个物理场：density 和 pressure
    local_fields = Dict(:density => create_dummy_field(1.0), :pressure => create_dummy_field(2.0))
    # 将 local_fields 设置为全局变量（模拟各 worker 上的情况）
    Main.local_fields = local_fields

    # 调用 ghost_exchange_3d_batch!（此处仅为示例，测试环境中可能不进行实际远程通信）
    updated_fields = ghost_exchange_3d_batch!(local_fields, domains[1]; timeout = 2.0)
    @test isa(updated_fields, Dict)
    stats = comm_performance_report()
    @test haskey(stats, :total_time)
end

##########################################################################
# 4. AdvancedParallel 测试
##########################################################################
@testset "AdvancedParallel Tests" begin
    # 自动检测 GPU 可用性，若不可用则跳过 GPU 相关测试
    gpu_available = false
    try
        gpu_available = CUDA.has_cuda()
    catch e
        @warn "检测 GPU 状态失败: $e"
        gpu_available = false
    end

    if gpu_available
        test_data = rand(1000)
        gpu_future = async_gpu_exchange(test_data, 0)
        gpu_result = fetch(gpu_future)
        @test gpu_result !== nothing
    else
        @info "GPU 不可用，跳过 GPU 相关测试。"
    end

    # 测试混合通信：异步 GPU 与 CPU 混合任务
    test_data2 = rand(1000)
    mixed_future = async_mixed_communication(test_data2, 0)
    mixed_result = fetch(mixed_future)
    @test isa(mixed_result, Tuple)

    # 测试全局同步与异步工作流
    sync_channel = AdvancedParallel.create_sync_channel()
    test_data3 = rand(1000)
    tuning_result = parallel_workflow_with_tuning(test_data3, 0, sync_channel)
    @test tuning_result !== nothing
    channel_result = take!(sync_channel)
    @test channel_result == tuning_result

    # 获取并检查性能报告
    perf_report = AdvancedParallel.get_parallel_performance_report()
    @test haskey(perf_report, :total_time)
end

##########################################################################
# 5. 性能分析
##########################################################################
@testset "Performance Analysis" begin
    println("Benchmarking Grid creation...")
    @btime create_grid(coordinate_system=:cartesian, limits=$limits, spacing=$spacing, bc=$bc, adaptive_params=$adaptive_params);

    println("Benchmarking Domain Decomposition...")
    @btime decompose_grid_3d($grid; overlap=$(overlap), proc_dims=$(proc_dims), weights=$(weights));

    println("Benchmarking Communication ghost_exchange_3d_batch!")
    @btime ghost_exchange_3d_batch!($local_fields, $(domains[1]); timeout=2.0);

    if gpu_available
        println("Benchmarking AdvancedParallel advanced_async_communication...")
        test_data4 = rand(1000)
        @btime AdvancedParallel.advanced_async_communication($test_data4, 0);
    else
        println("GPU not available; skipping AdvancedParallel GPU benchmark.")
    end
end

println("All tests completed successfully.")
