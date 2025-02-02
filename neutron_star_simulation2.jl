扩展版中子星内部结构与磁场耦合模拟 —— Julia 完整代码示例
#
# 主要功能：
# 1. 利用 TOV 方程求解中子星静态结构（基于多项式 EOS）
# 2. 计算纯偶极磁场以及引入多极修正的磁场分布
#    （多极项利用二阶勒让德多项式 P₂(x)=0.5*(3x^2-1)）
# 3. 模拟磁场时间演化（采用简化的扩散方程：∂B/∂t = η ∂²B/∂r²，
#    以展示 Ohmic 衰减效应，使用自适应时间步长求解）
# 4. 分别绘制中子星结构、磁场分布以及磁场随时间演化的结果
#
# 依赖包：DifferentialEquations.jl, Plots.jl, LinearAlgebra
################################################################################

using DifferentialEquations
using Plots
using LinearAlgebra

################################################################################
# 模块1：TOV 模型求解 —— 中子星内部结构
################################################################################

module TOVModule

export eos, ρ_of_P, TOV!, solve_TOV, compute_star

# 为简化计算，采用单位 G = c = 1
# 采用多项式 EOS: P = K * ρ^γ，其中 ρ 作为能量密度的近似
const K = 100.0     # EOS 多项式常数（单位需保证一致）
const gamma = 2.0   # 多项式指数

# EOS 函数：给定能量密度 ρ，计算压力 P
function eos(ρ)
    return K * ρ^gamma
end


ρ_of_P(P) = (P / K)^(1 / gamma)

# TOV 方程组
# u = [m, P]，其中 m(r) 为半径 r 内质量，P(r) 为压力
function TOV!(du, u, r, params)
    m, P = u
    # 为避免 r=0 时分母为零，采用近似处理
    if r < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ  # 近似认为能量密度 ε = ρ
    dm_dr = 4π * r^2 * ε
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2 * m))
    du[1] = dm_dr
    du[2] = dP_dr
end

# 求解 TOV 方程，输入中心压力 P_c
function solve_TOV(P_c)
    r0 = 1e-6                # 初始半径，避免 r=0 问题
    m0 = 0.0                 # 中心质量为 0
    u0 = [m0, P_c]
    # 设置积分区间，采用自适应步长和终止条件：当压力 P 降至 1e-8 以下时终止
    prob = ODEProblem(TOV!, u0, (r0, 20.0), nothing)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10,
                stop_condition = (u, t, integrator) -> u[2] < 1e-8)
    return sol
end

# 根据中心压力 P_c 计算中子星的半径 R 和总质量 M
function compute_star(P_c)
    sol = solve_TOV(P_c)
    R = sol.t[end]
    M = sol.u[end][1]
    return R, M
end

end  # module TOVModule

################################################################################
# 模块2：磁场模块 —— 多极场及磁场分布计算
################################################################################

module MagneticFieldModule

export psi_dipole, psi_multipole, dipole_field, multipole_field

# 引入 TOVModule 中的常量和函数（如有需要，可通过using ..TOVModule，但此处独立定义）
# 纯偶极磁场的磁流函数
function psi_dipole(r, θ, B0)
    return (B0 * r^2 / 2) * sin(θ)^2
end

# 定义二阶勒让德多项式 P₂(x) = 0.5*(3x^2 - 1)
function P2(x)
    return 0.5 * (3 * x^2 - 1)
end

# 多极修正：引入 P₂(cosθ) 项作为修正因子
# 定义多极磁流函数：Ψ(r,θ) = (B0 * r^2/2) * sin^2θ * [1 + α * P₂(cosθ)]
function psi_multipole(r, θ, B0, α)
    return psi_dipole(r, θ, B0) * (1 + α * P2(cos(θ)))
end

# 根据磁流函数计算磁场大小（仅计算 poloidal 分量）
# 简化计算：利用数值梯度近似
function multipole_field(r, θ, B0, α; dr=1e-3, dθ=1e-3)
    # 计算 Ψ 在 (r,θ) 点的数值偏导
    ψ = psi_multipole(r, θ, B0, α)
    ψ_r_plus = psi_multipole(r + dr, θ, B0, α)
    ψ_θ_plus = psi_multipole(r, θ + dθ, B0, α)
    dψ_dr = (ψ_r_plus - ψ) / dr
    dψ_dθ = (ψ_θ_plus - ψ) / dθ
    # poloidal 磁场分量 B_p ≈ |∇Ψ| / (r sinθ)
    denom = r * sin(θ)
    if denom < 1e-6
        return 0.0
    end
    B_p = sqrt(dψ_dr^2 + (dψ_dθ / r)^2) / denom
    return B_p
end

# 对于纯偶极场，可以直接调用 dipole_field（解析表达式）
function dipole_field(r, θ, B0; R_star=10.0)
    B_r = 2 * B0 * (R_star / r)^3 * cos(θ)
    B_θ = B0 * (R_star / r)^3 * sin(θ)
    return sqrt(B_r^2 + B_θ^2)
end

end  # module MagneticFieldModule

################################################################################
# 模块3：磁场时间演化 —— 简化的磁场扩散模型
################################################################################

module MagneticEvolutionModule

using DifferentialEquations
using LinearAlgebra
export magnetic_diffusion!

# 为简化起见，采用一维径向磁场扩散模型
# 模型：∂B/∂t = η * ∂²B/∂r²
# 此处 B = B(r, t) 表示沿径向分量的磁场，η 为磁扩散系数

# 定义 PDE 的空间离散化函数（方法：有限差分）
# 将空间 [r_min, r_max] 分成 N 个点
function magnetic_diffusion!(dB, B, p, t)
    η, r_min, r_max, N = p.η, p.r_min, p.r_max, p.N
    # 计算空间步长
    dr = (r_max - r_min) / (N - 1)
    # 使用中心差分计算第二阶导数
    for i in 2:(N-1)
        dB[i] = η * (B[i+1] - 2*B[i] + B[i-1]) / dr^2
    end
    # 边界条件：固定边界磁场（Dirichlet 边界条件）
    dB[1] = 0.0
    dB[N] = 0.0
end

# 设置参数类型
struct DiffusionParams
    η::Float64
    r_min::Float64
    r_max::Float64
    N::Int
end

end  # module MagneticEvolutionModule

################################################################################
# 主程序部分：调用上述模块实现 TOV 求解、多极磁场计算与磁场时间演化
################################################################################

# 引入各模块
using .TOVModule
using .MagneticFieldModule
using .MagneticEvolutionModule

# ------------------------------
# Part 1: TOV 模型求解，生成质量—半径曲线
# ------------------------------
println("正在求解 TOV 模型，生成质量—半径曲线……")
central_pressures = 10 .^ range(0, stop=2, length=30)
Rs = Float64[]
Ms = Float64[]
for P_c in central_pressures
    R, M = compute_star(P_c)
    push!(Rs, R)
    push!(Ms, M)
end
plt1 = plot(Rs, Ms, xlabel="Radius R (arbitrary units)", ylabel="Mass M (arbitrary units)",
            title="Mass-Radius Curve (TOV Model)", legend=false)
savefig(plt1, "mass_radius_curve_extended.png")
println("质量—半径曲线已保存：mass_radius_curve_extended.png")

# ------------------------------
# Part 2: 多极场磁场分布计算
# ------------------------------
println("正在计算多极场磁场分布……")
# 设置磁场参数
B0 = 1e12            # 表面偶极场强度 (Gauss)
α = 0.5              # 多极修正系数
# 生成二维数据：r 范围 [1, 20]；θ 范围 [0, π]
r_vals = range(1.0, stop=20.0, length=100)
θ_vals = range(0, stop=π, length=100)
B_field_multipole = zeros(length(r_vals), length(θ_vals))
for (i, r) in enumerate(r_vals)
    for (j, θ) in enumerate(θ_vals)
        B_field_multipole[i, j] = multipole_field(r, θ, B0, α)
    end
end
plt2 = heatmap(θ_vals, r_vals, B_field_multipole,
               xlabel="θ (radians)", ylabel="r (arbitrary units)",
               title="Multipole Field Distribution", colorbar_title="B (Gauss)")
savefig(plt2, "multipole_field_heatmap.png")
println("多极场磁场分布热图已保存：multipole_field_heatmap.png")

# ------------------------------
# Part 3: 磁场时间演化模拟（1D 磁扩散模型）
# ------------------------------
println("正在进行磁场时间演化模拟（1D 磁扩散模型）……")
# 设定空间区间和网格数量
r_min = 1.0
r_max = 20.0
N = 200
dr = (r_max - r_min) / (N - 1)
# 定义空间网格
r_grid = range(r_min, r_max, length=N)
# 初始条件：采用多极场在赤道（θ = π/2）的磁场分布作为初始 B(r)
B0_initial = [dipole_field(r, π/2, B0) for r in r_grid]
# 设置磁扩散参数
η = 1e-4    # 磁扩散系数，单位与模型一致
diff_p = MagneticEvolutionModule.DiffusionParams(η, r_min, r_max, N)
# 定义时间区间：0 到 t_max（单位时间）
t_max = 50.0
tspan = (0.0, t_max)
# 定义 ODEProblem：采用方法的线性化空间离散化（方法：method of lines）
prob_diff = ODEProblem(MagneticEvolutionModule.magnetic_diffusion!, B0_initial, tspan, diff_p)
# 使用自适应求解器求解磁场扩散 PDE
sol_diff = solve(prob_diff, Rosenbrock23(), reltol=1e-8, abstol=1e-10)
# 绘制不同时间下磁场分布曲线
plt3 = plot(r_grid, sol_diff.u[end], xlabel="r (arbitrary units)", ylabel="Magnetic Field B (Gauss)",
            title="Magnetic Field Diffusion at t = $(t_max)",
            label="t = $(t_max)")
for t in [0.0, t_max/4, t_max/2, 3*t_max/4, t_max]
    sol_t = sol_diff(t)
    plot!(r_grid, sol_t, label="t = $(round(t, digits=2))")
end
savefig(plt3, "magnetic_diffusion_profiles.png")
println("磁场扩散模拟图已保存：magnetic_diffusion_profiles.png")

################################################################################
# Part 4: 耦合模型 —— TOV 方程中引入磁场修正项
################################################################################

println("正在求解耦合模型（TOV 方程中加入磁场修正项）……")
# 定义一个简单的磁场修正项，基于赤道 (θ = π/2) 的磁场值
function magnetic_correction(r, β, B0; R_star=10.0)
    # 取赤道时 θ = π/2
    return β * dipole_field(r, π/2, B0; R_star=R_star)
end

# 修改 TOV 方程：加入磁场修正项 ΔB(r)
function TOV_coupled!(du, u, r, params)
    β, B0 = params
    m, P = u
    if r < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = TOVModule.ρ_of_P(P)
    ε = ρ
    dm_dr = 4π * r^2 * ε
    ΔB = magnetic_correction(r, β, B0)
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2 * m)) + ΔB
    du[1] = dm_dr
    du[2] = dP_dr
end

# 求解耦合模型 TOV 方程的函数
function solve_TOV_coupled(P_c, β, B0)
    r0 = 1e-6
    m0 = 0.0
    u0 = [m0, P_c]
    params = (β, B0)
    prob = ODEProblem(TOV_coupled!, u0, (r0, 20.0), params)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10,
                stop_condition = (u, t, integrator) -> u[2] < 1e-8)
    return sol
end

# 设置参数并求解耦合模型
β = 0.01
P_c_sample = 1e2
sol_coupled = solve_TOV_coupled(P_c_sample, β, B0)
R_coupled = sol_coupled.t[end]
M_coupled = sol_coupled.u[end][1]
println("耦合模型下：Radius R = $(R_coupled), Mass M = $(M_coupled)")

# 绘制无磁场与耦合模型下的压力分布对比
sol_noB = TOVModule.solve_TOV(P_c_sample)
plt4 = plot(sol_noB.t, getindex.(sol_noB.u, 2), label="Without Magnetic Correction",
            xlabel="r (arbitrary units)", ylabel="Pressure P",
            title="Pressure Profile Comparison")
plot!(sol_coupled.t, getindex.(sol_coupled.u, 2), label="Coupled Model (β=$(β))")
savefig(plt4, "pressure_profile_comparison_extended.png")
println("耦合模型压力分布对比图已保存：pressure_profile_comparison_extended.png")

################################################################################
# 模块结束：总结与提示
################################################################################

println("所有模拟与绘图任务已完成，请检查生成的图像文件：")
println("1. Mass-Radius Curve: mass_radius_curve_extended.png")
println("2. Multipole Field Heatmap: multipole_field_heatmap.png")
println("3. Magnetic Diffusion Profiles: magnetic_diffusion_profiles.png")
println("4. Pressure Profile Comparison: pressure_profile_comparison_extended.png")
println("本代码示例扩展了多极场、磁场时间演化及耦合 TOV 模型，供后续进一步改进和扩展。")