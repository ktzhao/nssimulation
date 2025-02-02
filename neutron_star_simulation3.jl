################################################################################
# Complete Neutron Star Structure and Magnetic Field Coupling Simulation
#
# This code extends the basic TOV simulation by including:
# 1. Multipole magnetic field: introduces a quadrupole correction using the 
#    second Legendre polynomial.
# 2. Magnetic field time evolution: a simple 1D magnetic diffusion model
#    (Ohmic decay) is implemented using a finite difference method and 
#    DifferentialEquations.jl with adaptive time stepping.
# 3. Coupled iterative solution: an iterative loop that solves the TOV 
#    equations with a magnetic correction term and updates the correction 
#    until convergence.
# 4. Basic adaptive time stepping for the PDE.
#
# Dependencies: DifferentialEquations.jl, Plots.jl, LinearAlgebra
################################################################################

using DifferentialEquations
using Plots
using LinearAlgebra

################################################################################
# Module 1: TOV Model for Neutron Star Structure
################################################################################
module TOVModule

export eos, ρ_of_P, TOV!, solve_TOV, compute_star

# In natural units: G = c = 1.
# Using a polytropic EOS: P = K * ρ^γ, where ρ approximates the energy density.
const K = 100.0     # Polytropic constant (in consistent units)
const gamma = 2.0   # Polytropic exponent

# EOS function: given density ρ, return pressure P.
function eos(ρ)
    return K * ρ^gamma
end

# Inverse EOS: given pressure P, return density ρ.
ρ_of_P(P) = (P / K)^(1 / gamma)

# TOV equations:
#   dm/dr = 4π r^2 ε   (with ε ≈ ρ(P))
#   dP/dr = - ((ε + P) (m + 4π r^3 P)) / (r (r - 2m))
function TOV!(du, u, r, params)
    m, P = u
    if r < 1e-6  # Avoid singularity at r=0
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ  # Simple approximation: energy density equals ρ
    dm_dr = 4π * r^2 * ε
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2*m))
    du[1] = dm_dr
    du[2] = dP_dr
end

# Solve the TOV equations for a given central pressure P_c.
function solve_TOV(P_c)
    r0 = 1e-6                # Start at a small radius to avoid r=0 singularity.
    m0 = 0.0
    u0 = [m0, P_c]
    prob = ODEProblem(TOV!, u0, (r0, 20.0), nothing)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10,
                stop_condition = (u, t, integrator) -> u[2] < 1e-8)
    return sol
end

# Compute star radius and mass for a given central pressure.
function compute_star(P_c)
    sol = solve_TOV(P_c)
    R = sol.t[end]
    M = sol.u[end][1]
    return R, M
end

end  # module TOVModule

################################################################################
# Module 2: Magnetic Field Module (Multipole Field)
################################################################################
module MagneticFieldModule

export psi_dipole, psi_multipole, dipole_field, multipole_field

# Pure dipole magnetic flux function:
function psi_dipole(r, θ, B0)
    return (B0 * r^2 / 2) * sin(θ)^2
end

# Second Legendre polynomial: P₂(x) = 0.5*(3x^2 - 1)
function P2(x)
    return 0.5 * (3 * x^2 - 1)
end

# Multipole magnetic flux function: adds a quadrupole correction with coefficient α.
# Ψ(r,θ) = (B0 * r^2/2) * sin^2θ * (1 + α * P₂(cosθ))
function psi_multipole(r, θ, B0, α)
    return psi_dipole(r, θ, B0) * (1 + α * P2(cos(θ)))
end

# Compute magnetic field strength from the multipole flux function via numerical differentiation.
function multipole_field(r, θ, B0, α; dr=1e-3, dθ=1e-3)
    ψ = psi_multipole(r, θ, B0, α)
    ψ_r_plus = psi_multipole(r + dr, θ, B0, α)
    ψ_θ_plus = psi_multipole(r, θ + dθ, B0, α)
    dψ_dr = (ψ_r_plus - ψ) / dr
    dψ_dθ = (ψ_θ_plus - ψ) / dθ
    denom = r * sin(θ)
    if denom < 1e-6
        return 0.0
    end
    B_p = sqrt(dψ_dr^2 + (dψ_dθ / r)^2) / denom
    return B_p
end

# Analytical dipole field: used for pure dipole comparisons.
function dipole_field(r, θ, B0; R_star=10.0)
    B_r = 2 * B0 * (R_star / r)^3 * cos(θ)
    B_θ = B0 * (R_star / r)^3 * sin(θ)
    return sqrt(B_r^2 + B_θ^2)
end

end  # module MagneticFieldModule

################################################################################
# Module 3: Magnetic Field Evolution Module (1D Diffusion Model)
################################################################################
module MagneticEvolutionModule

using DifferentialEquations
using LinearAlgebra
export magnetic_diffusion!, DiffusionParams

# This module implements a 1D magnetic diffusion model as a simplified version of Ohmic decay.
# The PDE: ∂B/∂t = η ∂²B/∂r²

# Define a structure to hold diffusion parameters.
struct DiffusionParams
    η::Float64     # Magnetic diffusivity
    r_min::Float64
    r_max::Float64
    N::Int         # Number of spatial grid points
end

# Magnetic diffusion PDE using finite differences.
function magnetic_diffusion!(dB, B, p, t)
    η, r_min, r_max, N = p.η, p.r_min, p.r_max, p.N
    dr = (r_max - r_min) / (N - 1)
    # Central finite difference for interior points.
    for i in 2:(N-1)
        dB[i] = η * (B[i+1] - 2*B[i] + B[i-1]) / dr^2
    end
    # Dirichlet boundary conditions: fixed magnetic field at boundaries.
    dB[1] = 0.0
    dB[N] = 0.0
end

export DiffusionParams, magnetic_diffusion!

end  # module MagneticEvolutionModule

################################################################################
# Module 4: Coupled TOV and Magnetic Correction Iteration
################################################################################
module CoupledModelModule

using DifferentialEquations
using TOVModule
using MagneticFieldModule

export solve_TOV_coupled, coupled_TOV_iteration

# Define a simple magnetic correction term.
# Here we assume the correction ΔB(r) = β * B_dipole(r, θ=π/2)
function magnetic_correction(r, β, B0; R_star=10.0)
    return β * dipole_field(r, π/2, B0; R_star=R_star)
end

# Coupled TOV equations: TOV equations with an added magnetic correction term in dP/dr.
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
    dP_dr = -((ε + P) * (m + 4π * r^3 * P)) / (r * (r - 2*m)) + ΔB
    du[1] = dm_dr
    du[2] = dP_dr
end

# Solve the coupled TOV equations with given central pressure P_c, magnetic field strength B0, and coupling coefficient β.
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

# Coupled iteration: iteratively solve the coupled TOV equations, updating the coupling parameter if needed.
# Here我们简单采用迭代判断星体半径和质量是否收敛。
function coupled_TOV_iteration(P_c, B0, β_init, tol=1e-4, max_iter=20)
    β = β_init
    prev_R, prev_M = 0.0, 0.0
    sol_final = nothing
    for iter in 1:max_iter
        sol = solve_TOV_coupled(P_c, β, B0)
        R = sol.t[end]
        M = sol.u[end][1]
        println("Iteration $iter: β = $β, R = $R, M = $M")
        if iter > 1 && abs(R - prev_R) < tol && abs(M - prev_M) < tol
            println("Convergence reached at iteration $iter.")
            sol_final = sol
            break
        end
        prev_R, prev_M = R, M
        # 根据实际情况更新 β，例如这里简单地微调 β（真实物理更新可更复杂）
        β *= 1.01
    end
    return sol_final, β
end

end  # module CoupledModelModule

################################################################################
# Main Program: Running the Complete Simulation
################################################################################

using .TOVModule
using .MagneticFieldModule
using .MagneticEvolutionModule
using .CoupledModelModule

# ------------------------------
# Part 1: TOV Model - Mass-Radius Curve Generation
# ------------------------------
println("=== Running TOV Model: Generating Mass-Radius Curve ===")
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
println("Mass-Radius curve saved as: mass_radius_curve_extended.png")

# ------------------------------
# Part 2: Multipole Magnetic Field Distribution
# ------------------------------
println("=== Calculating Multipole Magnetic Field Distribution ===")
B0 = 1e12            # Surface dipole field strength (Gauss)
α = 0.5              # Multipole correction coefficient
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
println("Multipole field heatmap saved as: multipole_field_heatmap.png")

# ------------------------------
# Part 3: Magnetic Field Time Evolution (1D Diffusion Model)
# ------------------------------
println("=== Running Magnetic Field Time Evolution (1D Diffusion Model) ===")
# Spatial discretization parameters for magnetic evolution
r_min = 1.0
r_max = 20.0
N = 200
r_grid = range(r_min, r_max, length=N)
# Initial magnetic field: use the dipole field at equator (θ = π/2)
B0_initial = [dipole_field(r, π/2, B0) for r in r_grid]
# Diffusion parameters
η = 1e-4  # Magnetic diffusivity (arbitrary units)
diff_params = MagneticEvolutionModule.DiffusionParams(η, r_min, r_max, N)
tspan = (0.0, 50.0)
prob_diff = ODEProblem(MagneticEvolutionModule.magnetic_diffusion!, B0_initial, tspan, diff_params)
sol_diff = solve(prob_diff, Rosenbrock23(), reltol=1e-8, abstol=1e-10)
plt3 = plot(r_grid, sol_diff.u[end], xlabel="r (arbitrary units)", ylabel="Magnetic Field B (Gauss)",
            title="Magnetic Field Profile at t = $(tspan[2])")
for t in [0.0, tspan[2]/4, tspan[2]/2, 3*tspan[2]/4, tspan[2]]
    sol_t = sol_diff(t)
    plot!(r_grid, sol_t, label="t = $(round(t, digits=2))")
end
savefig(plt3, "magnetic_diffusion_profiles.png")
println("Magnetic field diffusion profile saved as: magnetic_diffusion_profiles.png")

# ------------------------------
# Part 4: Coupled Model - TOV with Magnetic Correction
# ------------------------------
println("=== Running Coupled Model: TOV Equations with Magnetic Correction ===")
# Set parameters: central pressure P_c_sample, initial coupling coefficient β_init, and field strength B0.
P_c_sample = 1e2
β_init = 0.01
sol_coupled, β_final = coupled_TOV_iteration(P_c_sample, B0, β_init, tol=1e-4, max_iter=20)
if sol_coupled !== nothing
    R_coupled = sol_coupled.t[end]
    M_coupled = sol_coupled.u[end][1]
    println("Coupled model converged: Final β = $(β_final), Radius R = $(R_coupled), Mass M = $(M_coupled)")
else
    println("Coupled model did not converge within the maximum iterations.")
end
# Compare pressure profiles between the pure TOV and the coupled model.
sol_noB = TOVModule.solve_TOV(P_c_sample)
plt4 = plot(sol_noB.t, getindex.(sol_noB.u, 2), label="Without Magnetic Correction",
            xlabel="r (arbitrary units)", ylabel="Pressure P", title="Pressure Profile Comparison")
plot!(sol_coupled.t, getindex.(sol_coupled.u, 2), label="Coupled Model (with Magnetic Correction)")
savefig(plt4, "pressure_profile_comparison_extended.png")
println("Pressure profile comparison saved as: pressure_profile_comparison_extended.png")

################################################################################
# End of Simulation
################################################################################
println("All simulation and plotting tasks completed. Please check the generated image files.")
