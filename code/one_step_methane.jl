using ModelingToolkit, DifferentialEquations, Plots

E = 2e4            # activation energy
β = 0.0            # temperature exponent
A = 1.1e10         # pre-exponential factor
R = 8.3145         # gas constant 
T = 1e3            # temperature in Kelvin

@variables t CH₄(t) O₂(t) CO₂(t) H₂O(t) r(t)
@parameters k, n_CH₄, n_O₂
D = Differential(t)

u₀ = [CH₄ => 1.0,  # initial value CH₄
      O₂  => 2.0,  # initial value O₂
      CO₂ => 0.0,  # initial value CO₂
      H₂O => 0.0]  # initial value H₂O

tspan = (0.0,      # start time t₀
         2e-8)     # end time tₙ

p = [k => A * T^β * exp(-E / R / T), # rate coefficient
     n_CH₄ => 1.0, # reaction exponent CH₄
     n_O₂ => 0.5]  # reaction exponent O₂

@named fol_separate = ODESystem([ r ~ CH₄^n_CH₄ * O₂^n_O₂,
                                  D(CH₄) ~ -k * r,
                                  D(O₂)  ~ -2k * r,
                                  D(CO₂) ~ k * r,
                                  D(H₂O) ~ 2k * r ])

prob = ODEProblem(structural_simplify(fol_separate), u₀, tspan, p)
sol = solve(prob)
plot(sol, linewidth = 2, xlims = (0., 2.1e-8),
     title = "One-step mechanism of methane combustion", 
     xlabel = "time [s]", 
     ylabel = "concentration [mol/L]")
savefig("res/one_step_mechanism_methane.svg")