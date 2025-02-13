using Random
using LinearAlgebra
include("EOSModule.jl")
using Main.EOSModule  # 确保导入了 EOSModule

# --------------------------
# 网格生成：创建1D网格，假设恒星半径范围为[0, r_end]
# --------------------------

function create_grid_data(r_end::Float64, num_points::Int)
    r = collect(LinRange(0.0, r_end, num_points))  # 将 LinRange 转换为 Vector{Float64}
    dx = r[2] - r[1]
    return r, dx
end

# --------------------------
# 物理场生成：根据简单模型生成初始物理场数据
# --------------------------

function generate_physical_fields(r::Vector{Float64}, eos::FiniteTempEOS)
    num_points = length(r)
    
    # 假设恒星温度在表面到中心逐渐升高
    temperature = [eos.T0 * (1.0 + r[i]/maximum(r)) for i in 1:num_points]
    
    # 假设恒星密度从表面到中心逐渐增加，使用一个简单的多项式关系
    # 使用eos.density函数来计算密度
    density_vals = [density(eos, pressure(eos, 1.0e6, temperature[i]), temperature[i]) * (1.0 - r[i]/maximum(r)) + 1.0e-6 for i in 1:num_points]
    
    # 使用eos.pressure函数计算压力
    pressure_vals = [pressure(eos, density_vals[i], temperature[i]) for i in 1:num_points]
    
    return temperature, density_vals, pressure_vals
end

# --------------------------
# EOS设置：定义有限温度模型
# --------------------------

eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)  # 假设热源为零

# --------------------------
# 模拟参数：设置网格和模拟条件
# --------------------------

r_end = 20.0  # 恒星半径最大值 (单位: km)
num_points = 100  # 网格点数
r, dx = create_grid_data(r_end, num_points)

# --------------------------
# 生成初始物理场数据
# --------------------------

temperature, density_vals, pressure_vals = generate_physical_fields(r, eos)

# --------------------------
# 输出模拟输入数据
# --------------------------

println("模拟输入数据:")
println("网格数据 (r): ", r)
println("网格间距 (dx): ", dx)
println("温度数据 (T): ", temperature)
println("密度数据 (ρ): ", density_vals)
println("压力数据 (P): ", pressure_vals)


# --------------------------
# 生成自适应网格细化（AMR）相关数据
# --------------------------

function generate_amr_data(r::Vector{Float64}, temperature::Vector{Float64}, pressure::Vector{Float64})
    # 这里假设根据温度和压力来决定细化区域
    refinement_threshold = 0.1
    max_refinement_level = 5
    min_refinement_level = 1

    # 根据温度和压力的变化判断细化区域
    gradient_temperature = diff(temperature) ./ diff(r)
    gradient_pressure = diff(pressure) ./ diff(r)
    
    # 计算需要细化的区域（基于温度和压力梯度）
    refinement_level = [min_refinement_level + Int(abs(gradient_temperature[i]) > refinement_threshold) +
                        Int(abs(gradient_pressure[i]) > refinement_threshold) for i in 1:length(gradient_temperature)]
    
    return refinement_level
end

# 生成AMR数据
refinement_level = generate_amr_data(r, temperature, pressure_vals)
println("自适应网格细化级别：", refinement_level)

# --------------------------
# 设置EOS与初始条件
# --------------------------

eos = FiniteTempEOS(2.0, 1.0, 1.0e6, 1.0e-3, t -> 0.0)  # 假设没有热源

# 输出最终模拟输入数据
println("初始EOS参数：")
println("gamma = ", eos.gamma)
println("K = ", eos.K)
println("T0 = ", eos.T0)
