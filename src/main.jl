# main.jl
#
# 这个主程序调用 EOSModule 和 TOVSolver 模块进行 TOV 星模型的求解。
# 采用多项式状态方程（也可以根据需求修改为其他 EOS）作为 EOS，
# 使用 DifferentialEquations.jl 求解 TOV 方程，
# 最后输出半径、压力和质量分布，并绘制压力分布图。
#
# 使用前请确保安装以下包：
#   - DifferentialEquations.jl
#   - Plots.jl
#
# 运行命令（在终端中）：
#   julia --project main.jl

using DifferentialEquations
using Plots

# 加载 EOS 模块和 TOV 求解模块

include("GridModule.jl")
include("EOSModule.jl")
include("TOVSolver.jl")

using .GridModule
using .EOSModule
using .TOVSolver
# 设置 TOV 模型参数
Pc = 1e-3          # 中心压力（可根据物理意义调整单位和数值）
K = 1.0            # 多项式 EOS 常数
gamma = 2.0        # 多项式指数
T = 0.0            # 温度（0 表示不考虑温度效应）
B = 0.0            # 磁场强度（0 表示不考虑磁场效应）
omega = 0.0        # 旋转角速度（0 表示不考虑旋转效应）
r_end = 10.0       # 积分终止半径
tol = 1e-8         # 积分容差
small_r = 1e-6     # 避免奇点的最小半径

println("开始求解 TOV 方程……")
solution = solve_tov(Pc; K=K, gamma=gamma, T=T, B=B, omega=omega, r_end=r_end, tol=tol, small_r=small_r)

# 输出求解结果
println("\nTOV 模型求解结果：")
println("半径 (r) 数组：")
# println(solution.r)  # 可取消注释查看完整半径数组
println("压力 (P) 数组：")
# println(solution.P)  # 可取消注释查看完整压力数组
println("质量 (m) 数组：")
# println(solution.m)  # 可取消注释查看完整质量数组
println("最终星体半径： $(maximum(solution.r))")
println("最终星体质量： $(maximum(solution.m))")

# 计算并输出观测量
observables = compute_observables(solution, K, gamma; T=T)
println("\n星体的观测量：")
println("星体半径 (R)： $(observables[:R])")
println("星体质量 (M)： $(observables[:M])")
println("星体惯性矩 (I)： $(observables[:I])")
println("紧凑度 (Compactness)： $(observables[:compactness])")
println("引力红移 (Redshift)： $(observables[:redshift])")
println("惯性比 (Inertia Ratio)： $(observables[:inertia_ratio])")

# 绘制压力分布图
plot(solution.r, solution.P,
     xlabel="Radius", ylabel="Pressure",
     title="TOV Pressure Profile", lw=2)
savefig("tov_pressure_profile.png")
println("压力分布图已保存为 tov_pressure_profile.png")

# 可选：如果需要计算旋转效应的修正，可以如下调用：
rotation_corrections = compute_rotation_corrections(solution, omega)
println("\n旋转效应修正：")
println("质量修正 (ΔM)： $(rotation_corrections[:delta_mass])")
println("半径修正 (ΔR)： $(rotation_corrections[:delta_radius])")
