using SimpleWeightedGraphs, LightGraphs, NetworkDynamics, 
      DifferentialEquations, Plots, SparseArrays, BenchmarkTools

E = 2e4            # activation energy
β = 0.             # temperature exponent
A = 1.1e10         # pre-exponential factor
R = 8.3145         # gas constant 
T = 1e3            # temperature in Kelvin
k = A * T^β * exp(-E / R / T) # rate coefficient
σ = 5e8            # constant for edges
n_CH₄ = 1.         # reaction order CH₄
n_O₂  = 0.5        # reaction order O₂
tspan = (0.,       # start time t₀
         2e-8)     # end time tₙ
n_species = 4

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
    if p > 0.     # forward  direction of directed edge
        for i in 1:4
            if u_s[i] > u_d[i]
                e[i] = p * (u_s[i] - u_d[i])
            end
        end
    elseif p < 0. # backward direction of directed edge
        for i in 1:4
            if u_d[i] > u_s[i]
                e[i] = p * (u_d[i] - u_s[i])
            end
        end
    end
    nothing
end

node = ODEVertex(f! = chemistry!, dim = n_species, 
                 sym=[:CH₄, :_O₂, :CO₂, :H₂O])
edge = StaticEdge(f! = transport!, dim = n_species, 
                  coupling = :directed)

function measure(n_node, solver)
    println("\n$(n_node) nodes\n$(solver)")
    # adjacency matrix
    G = spdiagm(n_node, n_node, 1 => ones(n_node-1))
    g_weighted = SimpleWeightedDiGraph(G - G')
    edge_weights = getfield.(collect(edges(g_weighted)), :weight)
    g_directed = SimpleDiGraph(g_weighted)
    network! = network_dynamics(node, edge, g_directed)

    p = (nothing, σ * edge_weights)
    u₀ = zeros(4n_node)
    init = collect(range(1., 0., length=n_node))
    for i in 1:n_node
        u₀[4i-3] = init[i]  # initial value CH₄ in node i
        u₀[4i-2] = 2init[i] # initial value O₂  in node i
    end
    prob = ODEProblem(network!, u₀, tspan, p)
    @btime solve($prob, $solver, save_everystep=false)
    nothing
end

for n_node in [2^i for i in 1:10]
    for solver in [Rodas5(autodiff=false), TRBDF2(autodiff=false),
                   TRBDF2(autodiff=false, linsolve=LinSolveGMRES())]
        measure(n_node, solver)
    end
end