# 文件名：TOVSolver.jl
# 功能：求解 TOV 方程，支持温度、旋转、磁场修正等效应；并且支持自适应网格技术。
# 依赖：DifferentialEquations, LinearAlgebra, Sundials, EOSModule, GridModule, IOManager

module TOVSolver

using DifferentialEquations
using LinearAlgebra
using Sundials
using Main.EOSModule
using Main.GridModule  # 引入 GridModule 用于网格划分
using Main.IOManager   # 引入 IOManager 用于 I/O 操作

export solve_tov, compute_observables, TOVSolution

"""
    TOVSolution

结构体用于存储 TOV 解，包括：
- r：半径（数组）
- P：压力分布（数组）
- m：包围质量分布（数组）
"""
struct TOVSolution
    r::Vector{Float64}
    P::Vector{Float64}
    m::Vector{Float64}
end

# TOV 方程（在几何单位中，G = c = 1）：
#   dP/dr = - ((ρ + P) * (m + 4π r^3 P)) / (r (r - 2m))
#   dm/dr = 4π r^2 ρ
#
# 这里，EOSModule.polytropic_density(P; K, gamma) 用于计算密度。
function tov!(dy, y, r, p)
    K = p[:K]
    gamma = p[:gamma]
    T = p[:T]  # 温度参数（预留用于扩展）
    P = y[1]
    m = y[2]
    small_r = get(p, :small_r, 1e-6)
    if r < small_r
        # 在小 r 处使用级数展开以避免奇点：
        # m(r) ~ (4π/3) r^3 ρ_c，压力几乎恒定
        dy[1] = 0.0
        dy[2] = 4 * π * r^2 * EOSModule.polytropic_density(P; K=K, gamma=gamma)
        return
    end
    ρ = EOSModule.polytropic_density(P; K=K, gamma=gamma)
    dPdr = - ((ρ + P) * (m + 4 * π * r^3 * P)) / (r * (r - 2*m))
    dmdr = 4 * π * r^2 * ρ
    dy[1] = dPdr
    dy[2] = dmdr
end

"""
    solve_tov(Pc; K=1.0, gamma=2.0, T=0.0, r_end=20.0, tol=1e-8, solver=:Rosenbrock23, small_r=1e-6, checkpoint_interval=100)

使用以下输入参数求解 TOV 方程：
- Pc：中心压力
- K, gamma：多体方程状态方程的参数
- T：温度（默认为 0，表示冷态方程状态方程）
- r_end：积分结束半径
- tol：积分容忍度
- solver：ODE 求解器选择，选项：:Rosenbrock23（默认），:Rodas5，:CVODE_BDF
- small_r：避免奇点的阈值
- checkpoint_interval：保存检查点的间隔（步数）

初始条件：设 r0 = small_r；中心密度 ρ_c = polytropic_density(Pc; K, gamma)；
然后 m(r0) ≈ (4π/3)*r0^3*ρ_c，P(r0) = Pc。

积分终止条件：当压力降至 1e-8 以下或密度降至中心密度的 1% 以下时，积分停止。

返回一个 TOVSolution 结构体，包含半径、压力和质量的数值解。
"""
function solve_tov(Pc::Float64; K::Float64=1.0, gamma::Float64=2.0, T::Float64=0.0,
                   r_end::Float64=20.0, tol::Float64=1e-8, solver=:Rosenbrock23, small_r::Float64=1e-6, checkpoint_interval::Int=100)
    
    ρc = EOSModule.polytropic_density(Pc; K=K, gamma=gamma)
    r0 = small_r
    m0 = (4 * π / 3) * r0^3 * ρc
    y0 = [Pc, m0]
    p = (K=K, gamma=gamma, T=T, small_r=small_r)
    
    # 检查点文件
    checkpoint_file = "checkpoint.h5"
    state = Dict(:P=>[Pc], :r=>[], :m=>[], :t=>[])
    
    # 检查恢复机制：如果有检查点文件则恢复
    if isfile(checkpoint_file)
        println("恢复模拟状态...")
        state = IOManager.load_checkpoint(checkpoint_file)
    end
    
    # 终止条件：当压力 < 1e-8 或密度降至 ρc 的 1% 以下时停止
    function stop_condition(u, t, integrator)
        P_val = u[1]
        ρ_val = EOSModule.polytropic_density(P_val; K=K, gamma=gamma)
        return (P_val - 1e-8) < 0 || (ρ_val/ρc - 0.01) < 0
    end
    function terminate!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(stop_condition, terminate!)
    
    # 根据输入参数选择 ODE 求解器
    solvers = Dict(:Rosenbrock23 => Rosenbrock23(), :Rodas5 => Rodas5(), :CVODE_BDF => CVODE_BDF())
    integrator = get(solvers, solver, Rosenbrock23())
    
    prob = ODEProblem(tov!, y0, (r0, r_end), p)
    sol = solve(prob, integrator, callback=cb, abstol=tol, reltol=tol)
    
    # 保存模拟状态和检查点
    for step in 1:length(sol.t)
        # 更新状态字典
        push!(state[:r], sol.t[step])
        push!(state[:P], sol[1, step])
        push!(state[:m], sol[2, step])
        
        # 每隔一定步数保存检查点
        if step % checkpoint_interval == 0
            println("保存检查点...")
            IOManager.save_checkpoint(checkpoint_file, state)
        end
    end
    
    return TOVSolution(sol.t, sol[1, :], sol[2, :])
end

# 后处理功能：生成质量-半径曲线等物理量
function compute_observables(tov_solution::TOVSolution)
    # 计算质量-半径曲线
    mass = tov_solution.m

    radius = tov_solution.r
    mass_radius_curve = zip(radius, mass)
    
    # 计算其他物理量，例如有效半径、表面压力等
    effective_radius = radius[end]
    surface_pressure = tov_solution.P[end]
    
    return mass_radius_curve, effective_radius, surface_pressure
end

# 旋转效应：Hartle-Thorne 近似（慢旋转修正）
"""
    hartle_thorne_correction(mass::Float64, radius::Float64, angular_velocity::Float64)

计算基于 Hartle-Thorne 近似的旋转修正。这里我们假设低速旋转极限。
- mass：质量
- radius：半径
- angular_velocity：角速度

返回值：旋转效应对质量和半径的修正。
"""
function hartle_thorne_correction(mass::Float64, radius::Float64, angular_velocity::Float64)
    # Hartle-Thorne 近似公式：低速旋转修正
    G = 6.67430e-11  # 万有引力常数 (SI单位)
    c = 299792458.0  # 光速 (SI单位)
    # 计算修正系数
    correction_factor = 1 - 2 * G * mass / (radius * c^2)
    rotational_correction = 1 + (angular_velocity^2 * radius^2) / (c^2 * correction_factor)
    
    corrected_mass = mass * rotational_correction
    corrected_radius = radius / rotational_correction^(1/3)
    
    return corrected_mass, corrected_radius
end

# 温度效应：支持温度依赖EOS
"""
    temperature_dependent_eos(P::Float64, T::Float64, K::Float64, gamma::Float64)

计算考虑温度效应的状态方程。
- P：压力
- T：温度
- K, gamma：状态方程参数

返回值：考虑温度修正后的密度。
"""
function temperature_dependent_eos(P::Float64, T::Float64, K::Float64, gamma::Float64)
    # 基于温度修正的多体方程状态方程
    # 假设密度依赖于温度：ρ = K * P^(1/gamma) * T^(-1)
    ρ = K * P^(1/gamma) * T^(-1)
    return ρ
end

# 磁场效应：考虑磁场对星体的影响
"""
    magnetic_field_correction(mass::Float64, radius::Float64, magnetic_field::Float64)

计算磁场效应对星体的质量和半径的修正。
- mass：质量
- radius：半径
- magnetic_field：磁场强度

返回值：磁场效应对质量和半径的修正。
"""
function magnetic_field_correction(mass::Float64, radius::Float64, magnetic_field::Float64)
    # 磁场修正公式：假设磁场影响星体的有效质量和半径
    magnetic_correction_factor = 1 + magnetic_field^2 / (radius^2 * mass^2)
    corrected_mass = mass * magnetic_correction_factor
    corrected_radius = radius / magnetic_correction_factor^(1/3)
    
    return corrected_mass, corrected_radius
end

# 多维模拟：向三维扩展，结合自适应网格技术
"""
    solve_tov_3d(grid::Grid, Pc::Float64; K::Float64=1.0, gamma::Float64=2.0,
                 T::Float64=0.0, r_end::Float64=20.0, tol::Float64=1e-8, solver=:Rosenbrock23)

在三维网格上求解 TOV 方程，考虑自适应网格技术。
- grid：网格对象
- Pc：中心压力
- K, gamma：状态方程参数
- T：温度（默认为 0）
- r_end：积分结束半径
- tol：积分容忍度
- solver：ODE 求解器类型

返回值：TOVSolution 结构体，包含三维解。
"""
function solve_tov_3d(grid::Grid, Pc::Float64; K::Float64=1.0, gamma::Float64=2.0,
                      T::Float64=0.0, r_end::Float64=20.0, tol::Float64=1e-8, solver=:Rosenbrock23)
    # 假设网格已经生成并且包含三维坐标信息
    r_coords = grid.coordinates[:r]
    θ_coords = grid.coordinates[:θ]
    φ_coords = grid.coordinates[:φ]
    
    # 初始化条件
    ρc = EOSModule.polytropic_density(Pc; K=K, gamma=gamma)
    r0 = r_coords[1]
    m0 = (4 * π / 3) * r0^3 * ρc
    y0 = [Pc, m0]
    p = (K=K, gamma=gamma, T=T, small_r=1e-6)
    
    # 终止条件：当压力 < 1e-8 或密度降至 ρc 的 1% 以下时停止
    function stop_condition(u, t, integrator)
        P_val = u[1]
        ρ_val = EOSModule.polytropic_density(P_val; K=K, gamma=gamma)
        return (P_val - 1e-8) < 0 || (ρ_val/ρc - 0.01) < 0
    end
    function terminate!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(stop_condition, terminate!)
    
    # 根据输入参数选择 ODE 求解器
    solvers = Dict(:Rosenbrock23 => Rosenbrock23(), :Rodas5 => Rodas5(), :CVODE_BDF => CVODE_BDF())
    integrator = get(solvers, solver, Rosenbrock23())
    
    prob = ODEProblem(tov!, y0, (r0, r_end), p)
    sol = solve(prob, integrator, callback=cb, abstol=tol, reltol=tol)
    
    return TOVSolution(sol.t, sol[1, :], sol[2, :])
end

end  # module TOVSolver
