# main.jl
#
# 这是一个TOV模拟程序，结合了网格生成、TOV求解、并行计算和自适应网格（AMR）技术。
# 此程序假定已加载所有依赖模块（如 TOVSolver.jl, GridModule.jl, DomainDecomposition.jl, AMRModule.jl, Communication.jl, AdvancedParallel.jl）

include("IOManager.jl")
include("EOSModule.jl")
include("GridModule.jl")
include("TOVSolver.jl")
include("Communication.jl")
include("DomainDecomposition.jl")
include("AMRModule.jl")
include("AdvancedParallel.jl")

using .EOSModule
using .IOManager
using .GridModule
using .TOVSolver

using .DomainDecomposition
using .AMRModule
using .Communication
using .AdvancedParallel
using LinearAlgebra
using Distributed

# 全局网格设置
function create_global_grid()
    # 设置全局网格的空间范围和分辨率
    r_min = 1.0
    r_max = 20.0
    num_points = 100
    grid_spacing = (r_max - r_min) / (num_points - 1)
    
    # 初始化网格
    r = LinRange(r_min, r_max, num_points)
    dx = repeat([grid_spacing], num_points)
    
    # 创建全局网格对象
    global_grid = GridModule.create_grid(coordinate_system = :cartesian, 
                                          limits = Dict(:x => (r_min, r_max)),
                                          spacing = Dict(:x => grid_spacing), 
                                          bc = Dict(:x => :Dirichlet))
    return global_grid
end

# TOV模拟设置
function run_tov_simulation(global_grid, Pc, T)
    # 创建TOVSolver并求解TOV方程
    eos = EOSModule.finite_temp_eos(1.0, 2.0, 0.0, 0.1)  # 选择适当的EOS模型
    tov_solution = TOVSolver.solve_tov(Pc; K=1.0, gamma=2.0, T=T, eos=eos, r_end=20.0, tol=1e-8, solver=:Rosenbrock23)
    
    # 输出结果，例如质量-半径曲线
    mass_radius_curve, effective_radius, surface_pressure = TOVSolver.compute_observables(tov_solution)
    println("Effective Radius: ", effective_radius)
    println("Surface Pressure: ", surface_pressure)
    
    # 返回TOV解的结果
    return tov_solution
end

# 网格分解与负载均衡
function decompose_grid(global_grid)
    # 使用3D域分解
    overlap = (1, 1, 1)
    proc_dims = (2, 2, 2)  # 假设2x2x2的进程网格布局
    
    # 进行网格分解
    domains = DomainDecomposition.decompose_grid_3d(global_grid; overlap=overlap, proc_dims=proc_dims)
    
    # 返回分解后的子域
    return domains
end

# AMR自适应网格处理
function adapt_grid(global_grid, tov_solution)
    # 基于TOV解进行网格细化
    physical_gradient = abs.(diff(tov_solution.P))
    amr_grid = AMRModule.create_initial_grid(1.0, 20.0, 100, 1.5, 0.1)
    
    # 调整网格
    amr_grid = AMRModule.adapt_grid(amr_grid, physical_gradient)
    
    return amr_grid
end

# 主函数，组合所有模块

    # 初始化全局网格
global_grid = create_global_grid()
    
    # 设置初始中心压力和温度
Pc = 1.0e15  # 初始中心压力，单位可以根据实际需要进行设置
T = 1.0e6    # 初始温度，根据需要调整
    
    # 求解TOV方程
tov_solution = run_tov_simulation(global_grid, Pc, T)
    
    # 进行网格分解
domains = decompose_grid(global_grid)
    
    # 使用自适应网格
amr_grid = adapt_grid(global_grid, tov_solution)
    
    # 进行并行计算，假设已经启用了Distributed模块
@everywhere begin
     # 在每个子域上执行并行计算，调用相应的求解器
    println("Running parallel computation on process ", myid())
end
    
    # 处理并行计算的结果
comm_performance = Communication.comm_performance_report()
println("Communication performance: ", comm_performance)
end

# 初始化并行环境（如果需要）
#if !isdistributed()
    #addprocs(4)  # 假设我们启动4个进程进行计算
#end

# 执行主模拟
#main()
