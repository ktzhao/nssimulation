################################################################################
# Advanced Neutron Star Structure and Magnetic Field Coupling Simulation
#
# This code demonstrates a framework for simulating the internal structure of a
# neutron star (via the TOV equations) coupled with magnetic field evolution.
#
# Features included:
# 1. TOV Equation Solver with a complex, piecewise polytropic EOS.
# 2. Dipole Magnetic Field Distribution Calculation.
# 3. Coupled Model: Incorporates a magnetic correction term in the TOV equation.
#
# Dependencies: DifferentialEquations.jl, Plots.jl, LinearAlgebra, StaticArrays
################################################################################

using DifferentialEquations
using Plots
using LinearAlgebra
using StaticArrays

################################################################################
# 1. TOV Equation Solver with a Complex EOS
################################################################################

# We use natural units (G = c = 1).
# We implement a piecewise polytropic EOS:
# For P < P_transition, use P = K1 * ρ^γ1; for P ≥ P_transition, use P = K2 * ρ^γ2.
const K1 = 100.0      # Polytropic constant for low density
const γ1 = 2.0        # Polytropic exponent for low density
const K2 = 500.0      # Polytropic constant for high density
const γ2 = 2.5        # Polytropic exponent for high density
const P_transition = 1e2  # Transition pressure

# EOS function: Given density ρ, return pressure P.
function eos(ρ)
    if ρ < 1e-3
        return K1 * ρ^γ1
    else
        return K2 * ρ^γ2
    end
end

# Inverse EOS: Given pressure P, compute density ρ.
# To avoid complex values when P is slightly negative, we use max(P, 0.0)
function ρ_of_P(P)
    P_eff = max(P, 0.0)
    if P_eff < P_transition
        return (P_eff / K1)^(1 / γ1)
    else
        return (P_eff / K2)^(1 / γ2)
    end
end

# TOV equation in in-place form:
#   dm/dr = 4π r^2 ε, where ε ≈ ρ(P)
#   dP/dr = - ((ε + P) (m + 4π r^3 P)) / (r (r - 2m))
# Function signature: f(du, u, p, t)
function TOV!(du, u, p, t)
    m, P = u
    # t represents the radial coordinate r.
    if t < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ  # Approximate energy density as ρ
    dm_dr = 4π * t^2 * ε
    dP_dr = -((ε + P) * (m + 4π * t^3 * P)) / (t * (t - 2 * m))
    du[1] = dm_dr
    du[2] = dP_dr
end

# Solve TOV equations for a given central pressure P_c.
function solve_TOV(P_c)
    r0 = 1e-3      # Starting radius; increased to avoid extreme gradients near r=0.
    m0 = 0.0
    u0 = [m0, P_c]
    # Create ODEProblem without passing extra parameters.
    prob = ODEProblem(TOV!, u0, (r0, 20.0))
    
    # Define a continuous callback to terminate integration when pressure P < 1e-8.
    condition(u, t, integrator) = u[2] - 1e-8
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)
    
    # Use a stiff solver and relaxed tolerances; increase maxiters to allow more iterations.
    sol = solve(prob, Rosenbrock23(), reltol=1e-5, abstol=1e-7, callback=cb, maxiters=1e7)
    return sol
end

# Compute the star's radius and mass given a central pressure P_c.
function compute_star(P_c)
    sol = solve_TOV(P_c)
    R = sol.t[end]           # Final time value as the star's radius.
    M = sol.u[end][1]        # Total mass.
    return R, M
end

# Generate a mass-radius curve.
central_pressures = 10 .^ range(0, stop=2, length=50)
Rs = Float64[]
Ms = Float64[]
for P_c in central_pressures
    R, M = compute_star(P_c)
    push!(Rs, R)
    push!(Ms, M)
end

# Plot the mass-radius curve.
plt1 = plot(Rs, Ms, xlabel="Radius R (arbitrary units)", ylabel="Mass M (arbitrary units)",
            title="Neutron Star Mass-Radius Curve", legend=false)
savefig(plt1, "mass_radius_curve.png")
println("TOV model solved; mass-radius curve saved as: mass_radius_curve.png")

################################################################################
# 2. Dipole Magnetic Field Distribution Calculation
################################################################################

# Assume a pure dipole magnetic flux function:
#   Ψ(r, θ) = (B0 * r^2 / 2) * sin^2(θ)
function psi_dipole(r, θ, B0)
    return (B0 * r^2 / 2) * sin(θ)^2
end

# Compute the magnetic field magnitude from the dipole flux function.
# Analytical formulas for a pure dipole:
#   B_r = 2 * B0 * (R_star/r)^3 * cos(θ)
#   B_θ = B0 * (R_star/r)^3 * sin(θ)
function dipole_field(r, θ, B0; R_star=10.0)
    B_r = 2 * B0 * (R_star / r)^3 * cos(θ)
    B_θ = B0 * (R_star / r)^3 * sin(θ)
    return sqrt(B_r^2 + B_θ^2)
end

# Generate 2D data: compute magnetic field strength for r ∈ [1,20] and θ ∈ [0,π].
r_vals = range(1.0, stop=20.0, length=100)
θ_vals = range(0, stop=π, length=100)
B0_val = 1e12  # Surface magnetic field strength (Gauss)
B_field = zeros(length(r_vals), length(θ_vals))
for (i, r) in enumerate(r_vals)
    for (j, θ) in enumerate(θ_vals)
        B_field[i, j] = dipole_field(r, θ, B0_val)
    end
end

# Plot the dipole magnetic field heatmap.
plt2 = heatmap(θ_vals, r_vals, B_field,
               xlabel="θ (radians)", ylabel="r (arbitrary units)",
               title="Pure Dipole Magnetic Field Distribution", colorbar_title="B (Gauss)")
savefig(plt2, "dipole_field_heatmap.png")
println("Dipole magnetic field distribution computed; heatmap saved as: dipole_field_heatmap.png")

################################################################################
# 3. Coupled Model: TOV Equation with Magnetic Correction
################################################################################

# Magnetic correction: assume ΔB(r) = β * B(r, θ = π/2).
function magnetic_correction(r, β, B0; R_star=10.0)
    return β * dipole_field(r, π/2, B0; R_star=R_star)
end

# Coupled TOV equation: add magnetic correction ΔB(r) to the dP/dr equation.
function TOV_coupled!(du, u, p, t)
    β, B0 = p  # p contains the parameters (β, B0)
    m, P = u
    if t < 1e-6
        du[1] = 0.0
        du[2] = 0.0
        return
    end
    ρ = ρ_of_P(P)
    ε = ρ
    dm_dr = 4π * t^2 * ε
    ΔB = magnetic_correction(t, β, B0)
    dP_dr = -((ε + P) * (m + 4π * t^3 * P)) / (t * (t - 2 * m)) + ΔB
    du[1] = dm_dr
    du[2] = dP_dr
end

# Solve the coupled TOV equations with a given central pressure, coupling parameter β, and B0.
function solve_TOV_coupled(P_c, β, B0)
    r0 = 1e-3
    m0 = 0.0
    u0 = [m0, P_c]
    params = (β, B0)
    prob = ODEProblem(TOV_coupled!, u0, (r0, 20.0), params)
    
    # Use a ContinuousCallback to terminate integration when pressure < 1e-8.
    condition(u, t, integrator) = u[2] - 1e-8
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)
    
    sol = solve(prob, Rosenbrock23(), reltol=1e-5, abstol=1e-7, callback=cb, maxiters=1e7)
    return sol
end

# Compare the coupled model with the pure TOV model using a sample central pressure.
β_val = 0.01        # Coupling coefficient
P_c_sample = 1e2    # Sample central pressure
sol_coupled = solve_TOV_coupled(P_c_sample, β_val, B0_val)
R_coupled = sol_coupled.t[end]
M_coupled = sol_coupled.u[end][1]
println("Coupled model: Neutron star radius R = $(R_coupled), Mass M = $(M_coupled)")

# Compare pressure profiles between the pure TOV and coupled models.
sol_noB = solve_TOV(P_c_sample)
plt3 = plot(sol_noB.t, getindex.(sol_noB.u, 2), label="No Magnetic Correction", xlabel="r (arbitrary units)", ylabel="Pressure P")
plot!(sol_coupled.t, getindex.(sol_coupled.u, 2), label="Coupled Model (β=$(β_val))")
title!("Pressure Profile Comparison")
savefig(plt3, "pressure_profile_comparison.png")
println("Pressure profile comparison plot saved as: pressure_profile_comparison.png")

################################################################################
# 4. Code Summary
################################################################################

println("All simulation and plotting tasks completed. Please check the generated image files.")
