
# main.jl
#
# 这个主程序调用 EOSModule 和 TOVSolver 模块进行 TOV 星模型的求解。
# 采用简单的多项式状态方程作为 EOS，
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
include("EOSModule.jl")
include("TOVSolver.jl")

using .EOSModule
using .TOVSolver

# 设置 TOV 模型参数
Pc = 1e-3          # 中心压力（可根据物理意义调整单位和数值）
K = 1.0            # 多项式 EOS 常数
gamma = 2.0        # 多项式指数
r_end = 10.0       # 积分终止半径
tol = 1e-8         # 积分容差

println("开始求解 TOV 方程……")
solution = solve_tov(Pc; K=K, gamma=gamma, r_end=r_end, tol=tol)

# 输出求解结果
println("\nTOV 模型求解结果：")
println("半径 (r) 数组：")
#println(solution.r)
println("压力 (P) 数组：")
#println(solution.P)
println("质量 (m) 数组：")
#println(solution.m)
println("最终星体半径： $(maximum(solution.r))")
println("最终星体质量： $(maximum(solution.m))")

# 绘制压力分布图
plot(solution.r, solution.P,
     xlabel="Radius", ylabel="Pressure",
     title="TOV Pressure Profile", lw=2)
savefig("tov_pressure_profile.png")
println("压力分布图已保存为 tov_pressure_profile.png")
