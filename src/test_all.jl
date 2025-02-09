# test_all.jl
#
# 本文件对 GridModule、DomainDecomposition、Communication 和 AdvancedParallel 模块进行单元测试与性能分析，
# 并根据硬件环境自动检测 GPU 可用性，若 GPU 不可用则跳过 GPU 相关测试。
#
# 运行命令示例：
#   julia --project test_all.jl

using Distributed
using Test
using BenchmarkTools
using LinearAlgebra
using Dates
using CUDA  # 加载 CUDA 模块

# 如果 worker 数量不足，则添加

if nprocs() < 2
    addprocs(2)
end

# 主进程加载 GridModule.jl
include("GridModule.jl")
using .GridModule

# 在所有 worker 上设置工作目录并加载 GridModule.jl
@everywhere begin
    cd(@__DIR__)
    include("GridModule.jl")
    using Main.GridModule
end

# 之后再加载其它模块，例如 DomainDecomposition、Communication、AdvancedParallel
include("DomainDecomposition.jl")
include("Communication.jl")
include("AdvancedParallel.jl")

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

# 显式使用模块前缀调用函数，确保调用的是 Main.GridModule.create_grid
const grid = GridModule.create_grid(
    coordinate_system = :cartesian,
    limits = limits,
    spacing = spacing,
    bc = bc,
    adaptive_params = adaptive_params,
    stretch_funcs = stretch_funcs,
    custom_bc = custom_bc
)
grid.physical_fields[:density] = fill(0.0, 10)
grid.physical_fields[:pressure] = fill(0.0, 10)

##########################################################################
# 定义用于性能分析的全局变量
##########################################################################
const pd_overlap = (1, 1, 1)
const pd_weights = Dict(:x => [1.0, 1.0], :y => [1.0, 1.0], :z => [1.0, 1.0])
const pd_proc_dims = (2, 1, 1)
const pd_domains = DomainDecomposition.decompose_grid_3d(
    grid; overlap = pd_overlap, proc_dims = pd_proc_dims, weights = pd_weights
)

##########################################################################
# 1. GridModule 测试
##########################################################################
@testset "GridModule Tests" begin
    @test grid.coordinate_system == :cartesian
    @test haskey(grid.coordinates, :x)
    @test length(grid.coordinates[:x]) ≥ 1

    dummy_field = [1.0, 2.0, 3.0, 4.0]
    new_field = GridModule.apply_boundary_conditions(copy(dummy_field), :xlow, grid)
    @test new_field[1] == 0.0

    field_data = Dict(:density => fill(1.0, 10), :pressure => fill(2.0, 10))
    println("field_data 类型：", typeof(field_data))
    GridModule.init_physical_fields!(grid, field_data)
    @test haskey(grid.physical_fields, :density)
    GridModule.update_physical_field!(grid, :density, fill(3.0, 10))
    @test grid.physical_fields[:density][1] == 3.0

    toml_str = """
    [simulation]
    title = "Test Simulation"
    dt = 0.01
    """
    open("test_config.toml", "w") do f
        write(f, toml_str)
    end
    config = GridModule.read_config("test_config.toml")
    @test config[:simulation][:title] == "Test Simulation"
end

##########################################################################
# 2. DomainDecomposition 测试
##########################################################################
@testset "DomainDecomposition Tests" begin
    overlap = (1, 1, 1)
    weights = Dict(:x => [1.0, 1.0], :y => [1.0, 1.0], :z => [1.0, 1.0])
    proc_dims = (2, 1, 1)
    domains = DomainDecomposition.decompose_grid_3d(
        grid; overlap = overlap, proc_dims = proc_dims, weights = weights
    )
    @test length(domains) == prod(proc_dims)

    local_grid = DomainDecomposition.get_local_grid(domains[1])
    @test local_grid.dims[:x] ≥ 1

    fine_domains = DomainDecomposition.multi_level_decompose(
        grid, (2, 1, 1), (2, 1, 1); overlap = overlap,
        coarse_weights = weights, fine_weights = weights
    )
    @test all(d -> d.level == 2, fine_domains)
end

##########################################################################
# 3. Communication 测试
##########################################################################
@testset "Communication Tests" begin
    overlap = (1, 1, 1)
    weights = Dict(:x => [1.0, 1.0], :y => [1.0, 1.0], :z => [1.0, 1.0])
    proc_dims = (2, 1, 1)
    domains = DomainDecomposition.decompose_grid_3d(
        grid; overlap = overlap, proc_dims = proc_dims, weights = weights
    )

    nx, ny, nz = 10, 10, 10
    ghost = (1, 1, 1)
    function create_dummy_field(val)
        return fill(val, nx + 2 * ghost[1], ny + 2 * ghost[2], nz + 2 * ghost[3])
    end

    local_fields = Dict(
        :density => create_dummy_field(1.0),
        :pressure => create_dummy_field(2.0)
    )
    @eval Main begin
        local_fields = $local_fields
    end

    updated_fields = Communication.ghost_exchange_3d_batch!(local_fields, domains[1]; timeout = 2.0)
    @test isa(updated_fields, Dict)
    stats = Communication.comm_performance_report()
    @test haskey(stats, :total_time)
end

##########################################################################
# 4. AdvancedParallel 测试
##########################################################################
@testset "AdvancedParallel Tests" begin
    global gpu_available = false
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

        test_data2 = rand(1000)
        mixed_future = async_mixed_communication(test_data2, 0)
        mixed_result = fetch(mixed_future)
        @test isa(mixed_result, Tuple)
    else
        @info "GPU 不可用，跳过 GPU 相关测试。"
    end

    sync_channel = AdvancedParallel.create_sync_channel()
    test_data3 = rand(1000)
    tuning_result = AdvancedParallel.parallel_workflow_with_tuning(test_data3, 0, sync_channel)
    @test tuning_result !== nothing
    channel_result = take!(sync_channel)
    @test channel_result == tuning_result

    perf_report = AdvancedParallel.get_parallel_performance_report()
    @test haskey(perf_report, :total_time)
end

##########################################################################
# 5. 性能分析
##########################################################################
@testset "Performance Analysis" begin
    println("Benchmarking Grid creation...")
    @btime GridModule.create_grid(coordinate_system=:cartesian, limits=$limits, spacing=$spacing, bc=$bc, adaptive_params=$adaptive_params);

    println("Benchmarking Domain Decomposition...")
    @btime DomainDecomposition.decompose_grid_3d($grid; overlap=$(pd_overlap), proc_dims=$(pd_proc_dims), weights=$(pd_weights));

    println("Benchmarking Communication ghost_exchange_3d_batch!")
    @btime Communication.ghost_exchange_3d_batch!($local_fields, $(pd_domains[1]); timeout=2.0);

    if gpu_available
        println("Benchmarking AdvancedParallel advanced_async_communication...")
        test_data4 = rand(1000)
        @btime AdvancedParallel.advanced_async_communication($test_data4, 0);
    else
        println("GPU not available; skipping AdvancedParallel GPU benchmark.")
    end
end

println("All tests completed successfully.")
