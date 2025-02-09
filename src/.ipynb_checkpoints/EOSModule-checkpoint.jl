module EOSModule

export EquationOfState, polytropic_eos, ideal_gas_eos, pressure, density, polytropic_density, add_eos, remove_eos

using Interpolations

abstract type EquationOfState end

struct PolytropicEOS <: EquationOfState
    K::Float64
    gamma::Float64
end

struct IdealGasEOS <: EquationOfState
    R::Float64
end

const user_defined_eos = Dict{String, EquationOfState}()

function polytropic_eos(K::Float64, gamma::Float64)
    return PolytropicEOS(K, gamma)
end

function ideal_gas_eos(R::Float64)
    return IdealGasEOS(R)
end

function pressure(eos::EquationOfState, rho::Float64, T::Float64=0.0)
    if isa(eos, PolytropicEOS)
        return eos.K * rho^eos.gamma
    elseif isa(eos, IdealGasEOS)
        return rho * eos.R * T
    else
        error("Unknown EOS type")
    end
end

function density(eos::EquationOfState, P::Float64, T::Float64=0.0)
    if isa(eos, PolytropicEOS)
        return (P / eos.K)^(1 / eos.gamma)
    elseif isa(eos, IdealGasEOS)
        return P / (eos.R * T)
    else
        error("Unknown EOS type")
    end
end

function polytropic_density(P::Number; K::Number=1.0, gamma::Number=2.0)
    # Ensure P is non-negative (if small negative due to numerical error, set to zero)
    P_adj = real(P) < 0 ? zero(P) : P
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
