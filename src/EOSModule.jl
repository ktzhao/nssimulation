# EOSModule.jl
# 本模块定义 EOS 相关函数，支持多项式 EOS、理想气体 EOS、有限温度 EOS（简单修正模型）、查表 EOS，
# 以及压强和密度的计算。依赖：Interpolations。

module EOSModule

export EquationOfState, TemperatureDependentEOS, PolytropicEOS, IdealGasEOS, FiniteTempEOS, TabulatedEOS,
       polytropic_eos, ideal_gas_eos, finite_temp_eos, tabulated_eos, pressure, density, polytropic_density, add_eos, remove_eos

using Interpolations

abstract type EquationOfState end

# 多项式 EOS（不含温度依赖）
struct PolytropicEOS <: EquationOfState
    K::Float64
    gamma::Float64
end

# 理想气体 EOS
struct IdealGasEOS <: EquationOfState
    R::Float64
end

# 温度依赖 EOS 抽象类型
abstract type TemperatureDependentEOS <: EquationOfState end

# 有限温度 EOS，简单修正模型：P = K * ρ^γ * (1 + beta*(T - T0))
struct FiniteTempEOS <: TemperatureDependentEOS
    K::Float64
    gamma::Float64
    T0::Float64
    beta::Float64
end

# 查表 EOS，通过二维插值实现 P = f(ρ, T) 和 ρ = g(P, T)
struct TabulatedEOS <: EquationOfState
    itp_P::Interpolations.GriddedInterpolation{Float64,2,Float64,Gridded{Linear},Tuple{Vector{Float64},Vector{Float64}}}
    itp_rho::Interpolations.GriddedInterpolation{Float64,2,Float64,Gridded{Linear},Tuple{Vector{Float64},Vector{Float64}}}
end

const user_defined_eos = Dict{String, EquationOfState}()

function polytropic_eos(K::Float64, gamma::Float64)
    return PolytropicEOS(K, gamma)
end

function ideal_gas_eos(R::Float64)
    return IdealGasEOS(R)
end

function finite_temp_eos(K::Float64, gamma::Float64, T0::Float64, beta::Float64)
    return FiniteTempEOS(K, gamma, T0, beta)
end

function tabulated_eos(rho_vals::Vector{Float64}, T_vals::Vector{Float64},
                       P_matrix::Matrix{Float64}, density_matrix::Matrix{Float64})
    itp_P = interpolate((rho_vals, T_vals), P_matrix, Gridded(Linear()))
    itp_rho = interpolate((P_matrix[:,1], T_vals), density_matrix, Gridded(Linear()))
    return TabulatedEOS(itp_P, itp_rho)
end

function pressure(eos::EquationOfState, rho::Float64, T::Float64=0.0)
    if isa(eos, PolytropicEOS)
        return eos.K * rho^eos.gamma
    elseif isa(eos, IdealGasEOS)
        return rho * eos.R * T
    elseif isa(eos, FiniteTempEOS)
        return eos.K * rho^eos.gamma * (1 + eos.beta*(T - eos.T0))
    elseif isa(eos, TabulatedEOS)
        return eos.itp_P(rho, T)
    else
        error("Unknown EOS type")
    end
end

function density(eos::EquationOfState, P::Float64, T::Float64=0.0)
    if isa(eos, PolytropicEOS)
        return (P / eos.K)^(1 / eos.gamma)
    elseif isa(eos, IdealGasEOS)
        return P / (eos.R * T)
    elseif isa(eos, FiniteTempEOS)
        return (P / (eos.K * (1 + eos.beta*(T - eos.T0))))^(1 / eos.gamma)
    elseif isa(eos, TabulatedEOS)
        return eos.itp_rho(P, T)
    else
        error("Unknown EOS type")
    end
end

function polytropic_density(P::Real; K::Real=1.0, gamma::Real=2.0)
    P_adj = P < 0 ? 0.0 : float(P)
    return (P_adj / K)^(1 / gamma)
end

function add_eos(name::String, eos::EquationOfState)
    user_defined_eos[name] = eos
end

function remove_eos(name::String)
    if haskey(user_defined_eos, name)
        delete!(user_defined_eos, name)
    else
        error("EOS name not found")
    end
end

end  # module EOSModule
