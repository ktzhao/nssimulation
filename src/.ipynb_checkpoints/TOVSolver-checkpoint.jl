module TOVSolver

using DifferentialEquations
using LinearAlgebra
using Sundials
using Main.EOSModule

export solve_tov, compute_observables, TOVSolution

struct TOVSolution
    r::Vector{Float64}
    P::Vector{Float64}
    m::Vector{Float64}
end

# ODE function signature: (du, u, p, t) where t is the independent variable.
function tov!(dy, y, p, r)
    K = p[:K]
    gamma = p[:gamma]
    T = p[:T]
    P = y[1]
    m = y[2]
    small_r = get(p, :small_r, 1e-6)
    if r < small_r
        dy[1] = 0.0
        dy[2] = 4 * π * r^2 * EOSModule.polytropic_density(P; K=K, gamma=gamma)
        return
    end
    ρ = EOSModule.polytropic_density(P; K=K, gamma=gamma)
    dPdr = - ((ρ + P) * (m + 4 * π * r^3 * P)) / (r * (r - 2 * m))
    dmdr = 4 * π * r^2 * ρ
    dy[1] = dPdr
    dy[2] = dmdr
end

function solve_tov(Pc::Float64; K::Float64=1.0, gamma::Float64=2.0, T::Float64=0.0,
                   r_end::Float64=20.0, tol::Float64=1e-8, solver=:Rosenbrock23, small_r::Float64=1e-6)
    ρc = EOSModule.polytropic_density(Pc; K=K, gamma=gamma)
    r0 = small_r
    m0 = (4 * π / 3) * r0^3 * ρc
    y0 = [Pc, m0]
    p = (K=K, gamma=gamma, T=T, small_r=small_r)
    function stop_condition(u, t, integrator)
        P_val = u[1]
        ρ_val = EOSModule.polytropic_density(P_val; K=K, gamma=gamma)
        return (P_val - 1e-8) < 0 || (ρ_val/ρc - 0.01) < 0
    end
    cb = ContinuousCallback(stop_condition, (integrator) -> terminate!(integrator))
    solvers = Dict(:Rosenbrock23 => Rosenbrock23(), :Rodas5 => Rodas5(), :CVODE_BDF => CVODE_BDF())
    integrator = get(solvers, solver, Rosenbrock23())
    prob = ODEProblem(tov!, y0, (r0, r_end), p)
    sol = solve(prob, integrator, callback=cb, abstol=tol, reltol=tol)
    return TOVSolution(sol.t, sol[1, :], sol[2, :])
end

function compute_inertia(sol::TOVSolution, K::Float64, gamma::Float64)
    r = sol.r
    P = sol.P
    m = sol.m
    n = length(r)
    f = zeros(n)
    for i in 1:n
        ρ = EOSModule.polytropic_density(P[i]; K=K, gamma=gamma)
        if r[i] <= 2 * m[i] || r[i] == 0.0
            f[i] = 0.0
        else
            f[i] = (8 * π / 3) * r[i]^4 * (ρ + P[i]) / sqrt(1 - 2 * m[i] / r[i])
        end
    end
    I = 0.0
    for i in 1:(n - 1)
        dr = r[i+1] - r[i]
        I += 0.5 * (f[i] + f[i+1]) * dr
    end
    return I
end

function compute_observables(sol::TOVSolution, K::Float64, gamma::Float64)
    R = sol.r[end]
    M = sol.m[end]
    I = compute_inertia(sol, K, gamma)
    compactness = M / R
    redshift = (1 - 2 * M / R)^(-0.5) - 1
    inertia_ratio = I / (M * R^2)
    return Dict(:R => R, :M => M, :I => I, :compactness => compactness, :redshift => redshift, :inertia_ratio => inertia_ratio)
end

end  # module TOVSolver
