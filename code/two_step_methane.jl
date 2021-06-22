using ModelingToolkit, DifferentialEquations, Plots

E₁ = 3.55e4        # activation energy
E₂ = 1.2e4
β₁ = 0.0           # temperature exponent
β₂ = 0.8
A₁ = 4.9e9         # pre-exponential factor
A₂ = 2e8
R = 8.3145         # gas constant 
T = 1e3            # temperature in Kelvin

@variables t CH₄(t) O₂(t) CO(t) CO₂(t) H₂O(t) r₁(t) r₂(t)
@parameters k₁, k₂, n_CH₄, n_CO, n_O₂_₁, n_O₂_₂
D = Differential(t)
u₀ = [CH₄ => 1.0,  # initial value CH₄
      O₂  => 2.0,  # initial value O₂
      CO  => 0.0,  # initial value CO
      CO₂ => 0.0,  # initial value CO₂
      H₂O => 0.0]  # initial value H₂O
tspan = (0.0,      # start time t₀
         1e-7)     # end time tₙ
p = [k₁ => A₁ * T^β₁ * exp(-E₁ / R / T), # rate coefficient 1
     k₂ => A₂ * T^β₂ * exp(-E₂ / R / T), # rate coefficient 2
     n_CH₄  => 0.5,  # reaction exponent CH₄
     n_CO   => 1.0,  # reaction exponent CO
     n_O₂_₁ => 0.65, # reaction exponent O₂ 1
     n_O₂_₂ => 0.5]  # reaction exponent O₂ 2
@named fol_separate = ODESystem([ r₁ ~ CH₄^n_CH₄ * O₂^n_O₂_₁,
                                  r₂ ~ CO^n_CO * O₂^n_O₂_₂,
                                  D(CH₄) ~ -k₁ * r₁,
                                  D(O₂)  ~ -1.5k₁ * r₁ - 0.5k₂ * r₂,
                                  D(CO)  ~ k₁ * r₁ - k₂ * r₂,
                                  D(CO₂) ~ k₂ * r₂,
                                  D(H₂O) ~ 2k₁ * r₁ ])
prob = ODEProblem(structural_simplify(fol_separate), u₀, tspan, p)
sol = solve(prob)
plot(sol, linewidth = 2, xlims = (0., 1.04e-7),
     title="Two-step mechanism of methane combustion", 
     xlabel="time [s]", 
     ylabel="concentration [mol/L]")
savefig("res/two_step_mechanism_methane.svg")

plot(sol, linewidth = 2, xlims = (0., 1.04e-7), vars = (0, 3),
     title="CO in two-step mechanism of methane combustion", 
     xlabel="time [s]", 
     ylabel="concentration [mol/L]")
savefig("res/two_step_mechanism_methane_CO.svg")