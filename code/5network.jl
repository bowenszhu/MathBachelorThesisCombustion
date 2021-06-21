using SimpleWeightedGraphs, LightGraphs, NetworkDynamics, 
      DifferentialEquations, Plots

E = 2e4            # activation energy
β = 0.             # temperature exponent
A = 1.1e10         # pre-exponential factor
R = 8.3145         # gas constant 
T = 1e3            # temperature in Kelvin
k = A * T^β * exp(-E / R / T) # rate coefficient
n_CH₄ = 1.         # reaction order CH₄
n_O₂  = 0.5        # reaction order O₂

config = "recirculation"
# config = "by-pass"

# adjacency matrix
G = [0. 1. 0. 0. 0.  # main flux and recirculation
     0. 0. 1. 0. 0.
     0. 0. 0. 1. 1.
     0. 0. 0. 0. 0.
     0. 1. 0. 0. 0.]
# G = [0. 1. 0. 0. 0.  # main flux and by-pass
#      0. 0. 1. 0. 1.
#      0. 0. 0. 1. 0.
#      0. 0. 0. 0. 0.
#      0. 0. 1. 0. 0.]

g_weighted = SimpleWeightedDiGraph(G - G')
edge_weights = getfield.(collect(edges(g_weighted)), :weight)
g_directed = SimpleDiGraph(g_weighted)

@inline Base.@propagate_inbounds function chemistry!(du, u, edges, p, t)
    if u[1] < 0.
        u[1] = 0.
    end
    if u[2] < 0.
        u[2] = 0.
    end
    rate = k * u[1]^n_CH₄ * u[2]^n_O₂
    du[1] = -rate    # dCH₄
    du[2] = -2rate   # dO₂
    du[3] = rate     # dCO₂
    du[4] = 2rate    # dH₂O
    for e in edges
        du .+= e
    end
    nothing
end

@inline Base.@propagate_inbounds function transport!(e, u_s, u_d, p, t)
    e .= 0.
    if p > 0.
        for i in 1:4
            if u_s[i] > u_d[i]
                e[i] = p * (u_s[i] - u_d[i])
            end
        end
    elseif p < 0.
        for i in 1:4
            if u_d[i] > u_s[i]
                e[i] = p * (u_d[i] - u_s[i])
            end
        end
    end
    nothing
end

node = ODEVertex(f! = chemistry!, dim = 4, sym=[:CH₄, :_O₂, :CO₂, :H₂O])
edge = StaticEdge(f! = transport!, dim = 4, coupling = :directed)
network! = network_dynamics(node, edge, g_directed)

p = (nothing, 5e8 * edge_weights)
u₀ = zeros(4 * 5)
u₀[1] = 1.        # initial value CH₄ in node 1
u₀[4*2-2] = 1.    # initial value O₂  in node 2
u₀[4*4-2] = 1.    # initial value O₂  in node 4
tspan = (0.,      # start time t₀
         2e-8)    # end time tₙ
prob = ODEProblem(network!, u₀, tspan, p)
sol = solve(prob, Rodas5(autodiff=false))

plot(sol, linewidth = 1, titlefont = 11,
     xlims = (0., 2.1e-8), ylims = (0., 1.),
     title = "5-node 5-edge network with main flux and $(config)", 
     ylabel = "concentration [mol/L]",
     xlabel = "time [s]")
savefig("res/$(config).svg")

for n in 1:5
    plot(sol, linewidth = 2,
         titlefont = 11, legendfont = 12,
         xlims = (0., 2.1e-8), ylims = (0., 1.),
         vars = syms_containing(network!, string(n)),
         title = "node $(n) in 5-node 5-edge network "
               * "with main flux and $(config)", 
         ylabel = "concentration [mol/L]",
         xlabel = "time [s]")
    savefig("res/$(config)$(n).svg")
end

for s in ["CH₄", "_O₂", "CO₂", "H₂O"]
    plot(sol, linewidth = 2,
         titlefont = 11, legendfont = 12,
         xlims = (0., 2.1e-8), ylims = (0., 1.),
         vars = idx_containing(network!, s),
         title = "$(s) in 5-node 5-edge network "
               * "with main flux and $(config)", 
         ylabel = "concentration [mol/L]",
         xlabel = "time [s]")
    savefig("res/$(config)$(s).svg")
end
