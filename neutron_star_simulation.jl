################################################################################
# 中子星内部结构与磁场耦合模拟 —— Julia 完整代码示例
#
# 本代码分为以下几部分：
# 1. TOV 方程求解（中子星结构求解）
# 2. 纯偶极磁场分布计算
# 3. 耦合模型：在 TOV 方程中加入简单的磁场修正项
#
# 依赖包：DifferentialEquations.jl, Plots.jl, LinearAlgebra
################################################################################

using DifferentialEquations
using Plots
using LinearAlgebra

################################################################################
# 1. TOV 方程求解 —— 中子星内部结构
################################################################################

# 为简化计算，采用单位 G = c = 1
# 采用多项式 EOS: P = K * ρ^γ，其中 ρ 表示原始密度，此处我们近似用能量密度 ε = ρ
const K = 100.0     # 多项式常数（单位根据具体情况选择）
const gamma = 2.0   # 多项式指数

# EOS 函数：给定 ρ，计算压力 P
function eos(ρ)
    return K * ρ^gamma
end

# 由于 TOV 方程通常以压力 P 为主要变量，
# 我们采用 EOS 的反函数：ρ(P) = (P/K)^(1/γ)
ρ_of_P(P) = (P / K)^(1 / gamma)

# TOV 方程组：
# dm/dr = 4π r^2 ε, 其中 ε = ρ(P)
# dP/dr = - ((ε + P) (m + 4π r^3 P)) / (r (r - 2m))
function TOV!(du, u, r, params)
    m, P = u
    # 为避免 r=0 时分母为 0，使用近似处理
    if r < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ  # 简单近似：ε = ρ
    dm_dr = 4π * r^2 * ε
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2 * m))
    du[1] = dm_dr
    du[2] = dP_dr
end

# 求解 TOV 方程，给定中心压力 P_c
function solve_TOV(P_c)
    r0 = 1e-6               # 初始半径（避免 r=0）
    m0 = 0.0                # 中心处质量为 0
    u0 = [m0, P_c]          # 初始条件向量
    # 定义积分区间，例如到 r = 20（单位长度），并设定终止条件：当 P 降到极小值时停止积分
    prob = ODEProblem(TOV!, u0, (r0, 20.0), nothing)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10,
                stop_condition = (u, t, integrator) -> u[2] < 1e-8)
    return sol
end

# 根据中心压力 P_c 计算中子星的半径 R 和质量 M
function compute_star(P_c)
    sol = solve_TOV(P_c)
    R = sol.t[end]           # 积分终止时的半径，即中子星半径
    M = sol.u[end][1]        # 对应的总质量
    return R, M
end

# 生成质量-半径曲线（Mass-Radius Curve）
central_pressures = 10 .^ range(0, stop=2, length=50)
Rs = Float64[]
Ms = Float64[]
for P_c in central_pressures
    R, M = compute_star(P_c)
    push!(Rs, R)
    push!(Ms, M)
end

# 绘制质量-半径曲线
plt1 = plot(Rs, Ms, xlabel="半径 R (单位长度)", ylabel="质量 M (单位质量)",
            title="中子星质量-半径曲线", legend=false)
savefig(plt1, "mass_radius_curve.png")
println("TOV 模型求解完成，已生成质量-半径曲线图：mass_radius_curve.png")

################################################################################
# 2. 纯偶极磁场分布计算
################################################################################

# 假设中子星表面磁场为纯偶极场，其磁流函数定义为：
# Ψ(r,θ) = (B0 * r^2 / 2) * sin^2θ
function psi_dipole(r, θ, B0)
    return (B0 * r^2 / 2) * sin(θ)^2
end

# 根据磁流函数，计算磁场的极坐标分量
# 对于纯偶极场，其解析公式为：
# B_r = 2 * B0 * (R_star/r)^3 * cosθ
# B_θ = B0 * (R_star/r)^3 * sinθ
function dipole_field(r, θ, B0; R_star=10.0)
    B_r = 2 * B0 * (R_star / r)^3 * cos(θ)
    B_θ = B0 * (R_star / r)^3 * sin(θ)
    return sqrt(B_r^2 + B_θ^2)
end

# 生成二维数据示例：计算 r ∈ [1,20] 与 θ ∈ [0,π] 范围内的磁场强度
r_vals = range(1.0, stop=20.0, length=100)
θ_vals = range(0, stop=π, length=100)
B0 = 1e12  # 设定表面磁场强度（单位：高斯）
B_field = zeros(length(r_vals), length(θ_vals))
for (i, r) in enumerate(r_vals)
    for (j, θ) in enumerate(θ_vals)
        B_field[i, j] = dipole_field(r, θ, B0)
    end
end

# 绘制磁场分布热图
plt2 = heatmap(θ_vals, r_vals, B_field,
               xlabel="θ (弧度)", ylabel="r (单位长度)",
               title="纯偶极场磁场强度分布", colorbar_title="B (高斯)")
savefig(plt2, "dipole_field_heatmap.png")
println("纯偶极场磁场分布计算完成，已生成热图：dipole_field_heatmap.png")

################################################################################
# 3. 耦合模型 —— TOV 方程中加入磁场修正项
################################################################################

# 假设磁场修正项为 ΔB(r) = β * B(r, θ=π/2)
# 其中 β 为耦合系数
function magnetic_correction(r, β, B0; R_star=10.0)
    # 取赤道 θ = π/2
    return β * dipole_field(r, π/2, B0; R_star=R_star)
end

# 定义耦合 TOV 方程：
# 在原 TOV 方程 dP/dr 基础上增加磁场修正项 ΔB(r)
function TOV_coupled!(du, u, r, params)
    β, B0 = params
    m, P = u
    if r < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ
    dm_dr = 4π * r^2 * ε
    ΔB = magnetic_correction(r, β, B0)
    # 修正 dP/dr：原公式加上磁场修正项
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2 * m)) + ΔB
    du[1] = dm_dr
    du[2] = dP_dr
end

# 求解耦合模型的 TOV 方程
function solve_TOV_coupled(P_c, β, B0)
    r0 = 1e-6
    m0 = 0.0
    u0 = [m0, P_c]
    params = (β, B0)
    prob = ODEProblem(TOV_coupled!, u0, (r0, 20.0), params)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10,
                stop_condition=(u, t, integrator) -> u[2] < 1e-8)
    return sol
end

# 以给定中心压力 P_c_sample，比较耦合模型与无磁场修正模型的结果
β = 0.01       # 耦合系数
P_c_sample = 1e2  # 示例中心压力
sol_coupled = solve_TOV_coupled(P_c_sample, β, B0)
R_coupled = sol_coupled.t[end]
M_coupled = sol_coupled.u[end][1]
println("耦合模型下：中子星半径 R = $(R_coupled), 质量 M = $(M_coupled)")

# 对比无磁场与耦合模型下的压力分布
sol_noB = solve_TOV(P_c_sample)
plt3 = plot(sol_noB.t, getindex.(sol_noB.u, 2), label="无磁场修正", xlabel="r (单位长度)", ylabel="压力 P")
plot!(sol_coupled.t, getindex.(sol_coupled.u, 2), label="耦合模型 (β=$(β))")
title!("压力分布对比")
savefig(plt3, "pressure_profile_comparison.png")
println("已生成压力分布对比图：pressure_profile_comparison.png")

################################################################################
# 4. 代码说明与总结
################################################################################

# 本代码示例实现了：

#   1. 通过 DifferentialEquations.jl 求解 TOV 方程得到中子星质量-半径曲线；
#   2. 利用解析公式计算纯偶极磁场分布，并绘制磁场强度热图；
#   3. 将磁场修正项简单引入 TOV 方程，构建耦合模型，并比较耦合前后的压力分布变化。
#
# 代码中的 EOS、磁场模型和耦合方法为示例，实际研究中可根据需要引入更复杂的 EOS 和磁场演化模型，
# 并采用全三维 GRMHD 数值方法进一步提高模拟精度。
#
# 您可以将此代码作为基础，扩展更多模块（如多极场、磁场时间演化、自适应网格等）来构建一个
# 完整的中子星内部结构与磁场耦合数值模拟平台。

println("所有模拟与绘图任务均已完成，请检查生成的图像文件。")