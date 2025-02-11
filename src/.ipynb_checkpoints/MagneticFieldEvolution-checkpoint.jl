# 文件名：MagneticFieldModule.jl
# 功能：定义纯偶极磁场模型，计算磁流函数 Ψ(r,θ)，从磁流函数计算磁场分量（B_r, B_θ）及其模值；
#       引入多极修正（如二阶勒让德多项式修正），模拟磁场演化过程，支持一维、二维和三维磁扩散模型。
# 依赖：LinearAlgebra, SpecialFunctions

module MagneticFieldModule

using LinearAlgebra
using SpecialFunctions

# 计算纯偶极磁场的磁流函数 Ψ(r, θ)
# 输入：r - 径向坐标，θ - 极角坐标
# 输出：磁流函数 Ψ(r, θ)
function magnetic_potential(r, θ)
    return r^2 * sin(θ)
end

# 从磁流函数 Ψ(r, θ) 计算磁场分量 B_r 和 B_θ
# 输入：r - 径向坐标，θ - 极角坐标
# 输出：磁场分量 B_r 和 B_θ
function magnetic_field_components(r, θ)
    Ψ = magnetic_potential(r, θ)
    B_r = -diff(Ψ, r)
    B_θ = diff(Ψ, θ) / r
    return B_r, B_θ
end

# 计算磁场的模值 B
# 输入：r - 径向坐标，θ - 极角坐标
# 输出：磁场的模值 B
function magnetic_field_magnitude(r, θ)
    B_r, B_θ = magnetic_field_components(r, θ)
    return sqrt(B_r^2 + B_θ^2)
end

# 引入多极修正（如二阶勒让德多项式修正）以模拟更复杂的磁场
# 输入：r - 径向坐标，θ - 极角坐标，l - 多极阶数
# 输出：修正后的磁流函数 Ψ(r, θ)
function multipole_correction(r, θ, l)
    P_l = legendre(l, cos(θ))
    return r^2 * P_l
end

# 从修正后的磁流函数 Ψ(r, θ) 计算磁场分量 B_r 和 B_θ
# 输入：r - 径向坐标，θ - 极角坐标，l - 多极阶数
# 输出：修正后的磁场分量 B_r 和 B_θ
function multipole_magnetic_field_components(r, θ, l)
    Ψ = multipole_correction(r, θ, l)
    B_r = -diff(Ψ, r)
    B_θ = diff(Ψ, θ) / r
    return B_r, B_θ
end

# 计算修正后的磁场的模值 B
# 输入：r - 径向坐标，θ - 极角坐标，l - 多极阶数
# 输出：修正后的磁场的模值 B
function multipole_magnetic_field_magnitude(r, θ, l)
    B_r, B_θ = multipole_magnetic_field_components(r, θ, l)
    return sqrt(B_r^2 + B_θ^2)
end

# 模拟磁场演化过程（适用于 1D、2D、3D 磁扩散模型）
# 输入：r - 径向坐标，θ - 极角坐标，l - 多极阶数，时间步长 dt，扩散系数 D
# 输出：演化后的磁场分量 B_r 和 B_θ
function magnetic_field_evolution(r, θ, l, dt, D)
    B_r, B_θ = multipole_magnetic_field_components(r, θ, l)
    # 计算磁场的时间导数（简化模型）
    dB_r = -D * diff(B_r, r) * dt
    dB_θ = -D * diff(B_θ, θ) * dt
    return B_r + dB_r, B_θ + dB_θ
end

end # module MagneticFieldModule
