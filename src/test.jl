using Test
using LinearAlgebra

include("IOManager.jl")
include("EOSModule.jl")

include("GridModule.jl")

include("TOVSolver.jl")

include("GRMHDModule.jl")

include("MagneticFieldEvolution.jl")

include("ParallelComputationModule.jl")
include("DomainDecomposition.jl")


using Main.IOManager
using Main.EOSModule
using Main.GridModule

using Main.TOVSolver

using Main.GRMHDModule

using Main.MagneticFieldEvolution
using Main.ParallelComputationModule
using Main.DomainDecomposition
using CUDA

# --------------------------
# GridModule 单元测试
# --------------------------

@testset "GridModule Tests" begin
    grid = create_grid(coordinate_system=:cartesian, 
                       limits=Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0)),
                       spacing=Dict(:x => 0.1, :y => 0.1, :z => 0.1))

    # 检查网格创建是否正常
    @test grid.dims[:x] == 11
    @test grid.dims[:y] == 11
    @test grid.dims[:z] == 11
    @test grid.coordinate_system == :cartesian
end

# --------------------------
# EOSModule 单元测试
# --------------------------

@testset "EOSModule Tests" begin
    eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)
    rho = 1.0
    T = 1.0e6
    P = pressure(eos, rho, T)
    density_val = density(eos, P, T)

    # 测试压力和密度计算
    @test isapprox(P, 1.0e6)
    @test isapprox(density_val, 1.0)

    # 测试温度更新
    new_temperature = update_temperature(P, eos, rho, T)
    @test isapprox(new_temperature, T - eos.cooling_rate * T^2 + eos.heat_source(T))
end

# --------------------------
# GRMHDModule 单元测试
# --------------------------

@testset "GRMHDModule Tests" begin
    eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)
    grid = create_grid(coordinate_system=:cartesian,
                       limits=Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0)),
                       spacing=Dict(:x => 0.1, :y => 0.1, :z => 0.1))
    
    # 测试演化
    @testset "evolve_grmhd" begin
        dt = 0.01
        evolve_grmhd(grid, eos, dt)
        # 这里只测试是否能够无错误执行
        @test true
    end

    # 测试磁场计算
    B = compute_magnetic_field(grid, eos, 1.0, 1.0e6, 1.0)
    @test isapprox(B, eos.magnetic_strength_factor * 1.0^0.5 * 1.0e6^0.25)

    # 测试电流密度计算
    J = compute_current_density(grid, eos)
    @test isapprox(J[1], eos.current_density_factor * 1.0e6^2)
end

# --------------------------
# TOVSolver 单元测试
# --------------------------

@testset "TOVSolver Tests" begin
    eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)
    
    # 测试TOV方程求解
    mass, pressure, density, temperature, r = solve_tov(1.0e6, eos=eos)
    
    @test length(mass) > 0
    @test length(pressure) > 0
    @test length(density) > 0
    @test length(temperature) > 0
end

# --------------------------
# DomainDecomposition 单元测试
# --------------------------

@testset "DomainDecomposition Tests" begin
    grid = create_grid(coordinate_system=:cartesian,
                       limits=Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0)),
                       spacing=Dict(:x => 0.1, :y => 0.1, :z => 0.1))

    # 测试3D域分解
    domains = decompose_grid_3d(grid; overlap=(1,1,1), proc_dims=(2,2,2))
    @test length(domains) > 0
end

# --------------------------
# I/O管理模块测试
# --------------------------

@testset "IOManager Tests" begin
    state = Dict(:density => rand(100), :temperature => rand(100))

    # 测试并行数据保存
    filename = "test_data.h5"
    @testset "Save and Load Checkpoint" begin
        save_checkpoint(filename, state)
        loaded_state = load_checkpoint(filename)
        
        # 检查保存和加载的数据是否一致
        @test loaded_state[:density] ≈ state[:density]
        @test loaded_state[:temperature] ≈ state[:temperature]
    end
end

# --------------------------
# ParallelComputationModule 测试
# --------------------------

@testset "ParallelComputationModule Tests" begin
    grid = create_grid(coordinate_system=:cartesian,
                       limits=Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0)),
                       spacing=Dict(:x => 0.1, :y => 0.1, :z => 0.1))

    # 测试GPU加速更新
    gpu_accelerated_update(grid)
    @test true  # 只测试是否能无错误执行

    # 测试并行计算
    parallel_update_domain(grid, 4)
    @test true  # 只测试是否能无错误执行
end

# --------------------------
# 集成测试：AMR与GRMHD集成
# --------------------------

@testset "AMR + GRMHD Integration" begin
    eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)
    grid = create_grid(coordinate_system=:cartesian,
                       limits=Dict(:x => (0.0, 1.0), :y => (0.0, 1.0), :z => (0.0, 1.0)),
                       spacing=Dict(:x => 0.1, :y => 0.1, :z => 0.1))

    # 创建自适应网格
    amr = AdaptiveMeshRefinement(
        grid_size=10,
        max_refinement_level=5,
        min_refinement_level=2,
        refinement_threshold=0.1,
        spacing=1.0,
        coordinates=Dict(:x => LinRange(0, 20, 100)),
        physical_fields=Dict(:temperature => zeros(100), :pressure => zeros(100)),
        current_refinement_level=1
    )

    # 进行网格细化
    refine_grid!(amr, :pressure, eos)
    @test amr.grid_size > 10  # 检查网格是否细化

    # 演化GRMHD
    evolve_grmhd(grid, eos, 0.01)
    @test true  # 只测试是否能无错误执行
end

# --------------------------
# 集成测试：TOV与EOS模块的协同工作
# --------------------------

@testset "TOV + EOS Integration" begin
    eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)
    
    # 求解TOV方程
    mass, pressure, density, temperature, r = solve_tov(1.0e6, eos=eos)

    # 测试EOS影响下的物理量
    @test length(mass) > 0
    @test length(pressure) > 0
end

