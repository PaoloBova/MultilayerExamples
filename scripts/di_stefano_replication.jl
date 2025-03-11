
# Next, we should use Agents.jl to simulate the change in strategies played
# in a prisoner's dilemma over time.

using Agents
using Random
using Graphs
using MultilayerGraphs
using DrWatson

using DataFrames
using Plots
using KernelDensity

using LinearAlgebra
using Distributions
using Test
include("abm_utils.jl")
using .Utilities

using ProgressMeter

# First, let's define the Agent types.

# Define the agent type
mutable struct Player <: AbstractAgent
    id::Int                       # Unique identifier for each agent
    node::MultilayerGraphs.Node                 # Unique identifier for its node
    strategy::Int              # Strategy (Cooperate, 1, or Defect, 0)
end

function simple_graph_to_layer(g, layer_id)
    "Convert a simple graph created using Graphs.jl to a layer of a
    multilayer network to be used by MultilayerGraphs.jl

    Note: This function will have to be updated if there are breaking changes
    to ME, MV, or layer_simplegraph.

    Note: Assumes the vertextype is INT64 and the weighttype is FLOAT64."

    vertex_list = Graphs.vertices(g)
    edge_list = Graphs.edges(g)
    node_list = [MultilayerGraphs.Node("node_$i") for i in vertex_list]
    mv_list = MultilayerGraphs.MV.(node_list)
    # Note that the src and dst of an Edge in Graphs.jl will always be integers
    # and by construction will always be a member of the vertex_list.
    edges_to_me(e) = MultilayerGraphs.ME(mv_list[e.src], mv_list[e.dst])
    me_list = edges_to_me.(edge_list)
    layer = MultilayerGraphs.layer_simplegraph(layer_id, mv_list, me_list)
    return layer
end

"""
Derive the communicability matrix, G, for the multiplex network in the model.
"""
function derive_G(layers, delta, W)
    M = length(layers)
    N = size(delta)[1]
    # Below we retrieve a sparse representation of each layer's adjacency matrix.
    adj_lists = [layer.graph.fadjlist for layer in layers]
    # We define homophily as 1/(1+delta)
    h = @. 1 / (1 + delta)
    # Define Z = h * adj_matrix (elementwise-product)
    masks = [Utilities.create_mask_matrix(adj, N) for adj in adj_lists]
    Z = cat([h[:, :, i] * masks[i] for i in 1:M]...; dims=3)
    # We need the matrix G = Z_L + C_LL
    G = zeros(N * M, N * M)
    for i in 1:M
        for j in 1:M
            if i == j
                G_ij = Z[:, :, i]
            end
            if i != j
                G_ij = W[i, j] * Diagonal(ones(N))
            end
            # Place X_ij in the appropriate block position in the final matrix X
            G[((i - 1) * N + 1):(i * N), ((j - 1) * N + 1):(j * N)] = G_ij
        end
    end
    return exp(G)
end

function get_positive_real_eigenvector(matrix, tol=1e-10)
    # Step 1: Compute the eigen decomposition
    eigen_decomp = eigen(matrix)

    # Step 2: Find indices of eigenvalues that are approximately real
    real_eigenvalue_indices = findall(x -> abs(imag(x)) < tol,
                                      eigen_decomp.values)

    # Check if there are no approximately real eigenvalues
    if isempty(real_eigenvalue_indices)
        error("No approximately real eigenvalues found")
    end

    # Step 3: Find the index of the largest real eigenvalue by magnitude
    real_eigenvalues = eigen_decomp.values[real_eigenvalue_indices]
    _, index_in_real = findmax(abs.(real_eigenvalues))
    index = real_eigenvalue_indices[index_in_real]

    # Step 4: Get the corresponding eigenvector
    principal_eigenvector = eigen_decomp.vectors[:, index]

    # Step 5: Ensure the eigenvector is real and positive
    positive_eigenvector = abs.(real.(principal_eigenvector))

    # Step 6: Normalize the eigenvector
    normalized_eigenvector = positive_eigenvector / sum(positive_eigenvector)

    # Eigenvector should sum to 1
    @test isapprox(sum(normalized_eigenvector), 1.0)

    return normalized_eigenvector
end

function derive_multilayer_centrality(layers, delta, W)
    M = length(layers)
    N = size(delta)[1]
    # Below we retrieve a sparse representation of each layer's adjacency matrix.
    adj_lists = [layer.graph.fadjlist for layer in layers]
    # We define homophily as 1/(1+delta)
    h = @. 1 / (1 + delta)
    # Define Z = h * adj_matrix (elementwise-product)
    masks = [Utilities.create_mask_matrix(adj, N) for adj in adj_lists]
    Z = cat([h[:, :, i] * masks[i] for i in 1:M]...; dims=3)
    # We next want to calcuate Zx, which is the Khatri-Rho product of the
    # matrices, Z, and the influence matrix,W.
    Zx = Utilities.khatri_rao_block(W, Z)

    # TODO: Can we make use of the sparse representation of Z when computing Zx.
    # Can we make use of the sparse representation of Zx when finding eigenvectors?

    o = get_positive_real_eigenvector(Zx)
    O = reshape(o, N, M)
    L = sum(O; dims=2)[:, 1]
    @test isapprox(sum(L), 1)

    # We now have a measure of the global eigenvector centrality of each node.
    # (I wonder how this compares to other methods of computing this).

    # TODO: See if we can already compute this efficienctly using
    # MultilayerGraphs.jl if we add weights to all of our edges as suggested by W
    # and Z.

    return L
end

"""
This function creates our Agent Based Model where players on a multiplex
network play a prisoner's dilemma with their neighbours each round. Later, we
will define the agent and environment steps. For now, all we do is create
a multiplex network of a given number of nodes and layers. We then add an
agent for each node. Each agent must be associated uniquely with one of the
nodes of the multplex network. They are initialised with a strategy which
is represented either by the number 1 or 0: 1 for cooperate, 0 for defect.

"""
function pd_multiplex_model(; # Specify types for parameters
                            model_id=1,
                            n_players=1000,
                            layer_ids=[:layer1, :layer2, :layer3],
                            b=10,
                            c=1,
                            beta=1,
                            delta_dist=Distributions.TruncatedNormal(0, 1, -5,
                                                                     -5),
                            W_dist=Distributions.TruncatedNormal(3.5, 2, 0, 5),
                            eta_max=1,
                            eta_min=0.1,
                            critical_mass_threshold=500,
                            seed=6998)
    # @time begin
    graphs = [Graphs.SimpleGraphs.barabasi_albert(n_players, 2)
              for _ in layer_ids]
    layers = simple_graph_to_layer.(graphs, layer_ids)
    node_ids = [node.id for node in nodes(layers[1])] # All layers have the same nodes.
    # end
    # @time begin
    # multiplex = MultilayerGraphs.MultilayerGraph(layers)

    mv_lookup = Dict() # Specify key-value types
    player_payoffs_initial = Dict() # Specify key-value types
    for layer in layers
        mvs = MultilayerGraphs.mv_vertices(layer)
        layer_node_ids = MultilayerGraphs.id.(MultilayerGraphs.node.(mvs))
        mv_lookup[layer.name] = Dict(zip(layer_node_ids, mvs))
        player_payoffs_initial[layer.name] = Dict(zip(layer_node_ids,
                                                      zeros(length(layer_node_ids))))
    end
    # end

    # @time begin
    M = length(layers)
    # The similarity matrix delta determines the homophily for each layer.
    # delta = rand(delta_dist, (n_players, n_players, 1))
    score = rand(delta_dist, n_players)
    delta = zeros(n_players, n_players)
    for i in 1:n_players
        for j in 1:n_players
            delta[i, j] = abs(score[j] - score[i])
        end
    end
    delta = cat([delta for _ in 1:M]...; dims=3)
    # The influence matrix W captures the influence of each layer on the other
    # layers. The influence of each layer on itself is 0.
    W = rand(W_dist, (M, M))
    for i in 1:M
        W[i, i] = 0
    end
    # We use delta and W to derive a matrix containing the communicability
    # of each node with each other node in every layer of the network.
    G = derive_G(layers, delta, W)
    L = derive_multilayer_centrality(layers, delta, W)
    # end

    # Our measure of centrality is a good proxy for how critical each agent
    # is to the multilayer network. We may be interested in how many of these
    # agents we need to be cooperative to encourage the rest of the network
    # to converge to full cooperation.
    # We want the set of nodes which have the highest centrality to be part
    # of our critical mass node set.

    critical_agent_ids = sortperm(L; rev=true)[1:critical_mass_threshold]
    critical_agent_node_ids = node_ids[critical_agent_ids]

    # @time begin
    model = Agents.ABM(Player,
                       nothing; # No space needed as graph passed in as a property
                       properties=Dict(:model_id => model_id,
                                       # :multilayer_network => multiplex,
                                       :layers => layers,
                                       :communicability_matrix => G,
                                       :centrality_measure => L,
                                       :critical_mass_threshold => critical_mass_threshold,
                                       :critical_agent_ids => critical_agent_ids,
                                       :critical_agent_node_ids => critical_agent_node_ids,
                                       :similarity_matrix => delta,
                                       :influence_matrix => W,
                                       :b => b,
                                       :c => c,
                                       :beta => beta,
                                       :eta_max => eta_max,
                                       :eta_min => eta_min,
                                       :mv_lookup => mv_lookup,
                                       :player_lookup => Dict(),
                                       :player_payoffs => player_payoffs_initial),
                       rng=Random.MersenneTwister(seed))
    # All layers have the same nodes
    # for node in nodes(multiplex)
    for node in nodes(layers[1])
        # Each player is a node of the multiplex.
        # Only the most central agents are initialised to be cooperative.
        initial_strategy = node.id in critical_agent_node_ids ? 1 : 0
        add_agent!(model, node, initial_strategy)
    end
    # Construct a lookup from node_id to agent_id
    agent_ids = getproperty.(Agents.allagents(model), :id)
    agent_node_ids = MultilayerGraphs.id.(getproperty.(Agents.allagents(model),
                                                       :node))
    player_lookup = Dict(zip(agent_node_ids, agent_ids))
    model.player_lookup = player_lookup

    return model
    # end
end

"""
In each round, all players play a prisoner's dilemma with each of their
neighbours, earning a given payoff. We achieve this by iterating through all
of the edges within each layer of the multiplex. 
"""
function model_step!(model) # Specify the type of model
    # multiplex = model.multilayer_network
    player_lookup = model.player_lookup # Specify the type of player_lookup
    b = model.b
    c = model.c
    # layers = [getproperty(multiplex, l.name) for l in multiplex.layers]
    layers = model.layers

    player_payoffs = Dict() # Specify key-value types
    # @time begin
    for layer in layers
        mvs = MultilayerGraphs.mv_vertices(layer)
        layer_node_ids = MultilayerGraphs.id.(MultilayerGraphs.node.(mvs))
        player_payoffs[layer.name] = Dict(zip(layer_node_ids,
                                              zeros(length(layer_node_ids))))
        # @time begin
        for edge in Graphs.edges(layer)

            # Retrieve the relevant players
            src_id = MultilayerGraphs.id(MultilayerGraphs.node(Graphs.src(edge)))
            dst_id = MultilayerGraphs.id(MultilayerGraphs.node(Graphs.dst(edge)))
            src_player = getindex(model, player_lookup[src_id])
            dst_player = getindex(model, player_lookup[dst_id])

            # Compute payoffs to each player given their strategies. Record
            # these behaviours in the payoffs property of the model.
            if (src_player.strategy == 1) & (dst_player.strategy == 1)
                src_player_payoff = b - c
                dst_player_payoff = b - c
            elseif (src_player.strategy == 1) &
                   (dst_player.strategy == 0)
                src_player_payoff = -c
                dst_player_payoff = b
            elseif (src_player.strategy == 0) &
                   (dst_player.strategy == 1)
                src_player_payoff = b
                dst_player_payoff = -c
            elseif (src_player.strategy == 0) &
                   (dst_player.strategy == 0)
                src_player_payoff = 0
                dst_player_payoff = 0
            end

            player_payoffs[layer.name][src_player.node.id] = get(player_payoffs[layer.name],
                                                                 src_player.node.id,
                                                                 0) +
                                                             src_player_payoff
            player_payoffs[layer.name][dst_player.node.id] = get(player_payoffs[layer.name],
                                                                 dst_player.node.id,
                                                                 0) +
                                                             dst_player_payoff
        end
        # end
    end
    # end
    model.player_payoffs = player_payoffs
    return model
end

# TODO: Should I restrict learning to take place with a counterpart from another layer?
"""
The agent step involves an agent considering whether to adopt a different
strategy than they currently do. All agents go through this step.
"""
function agent_step!(player::Player, model) # Specify the type of model
    # multiplex = model.multilayer_network
    player_lookup = model.player_lookup # Specify the type of player_lookup
    mv_lookup = model.mv_lookup # Specify the type of mv_lookup
    player_payoffs = model.player_payoffs # Specify the type of player_payoffs
    beta = model.beta
    delta = model.similarity_matrix
    # layers = [getproperty(multiplex, l.name) for l in multiplex.layers]
    layers = model.layers
    M = length(layers)
    N = length(Agents.allagents(model))
    learner_layer_index = rand(1:M)
    learner_layer = layers[learner_layer_index]
    role_model_layer_index = rand(1:M)
    role_model_layer = layers[role_model_layer_index]

    # We want to focus on the elements of our communicability matrix which
    # capture the interactions between the learner and role_model layers.
    G = model.communicability_matrix
    G_ab = G[((learner_layer_index - 1) * N + 1):(learner_layer_index * N),
             ((role_model_layer_index - 1) * N + 1):(role_model_layer_index * N)]
    # We set the min and max values of eta as follows.
    eta_max = model.eta_max
    eta_min = model.eta_min

    # Find a neighbour on the role_model_layer to be the player's role model
    player_mv = mv_lookup[role_model_layer.name][player.node.id]
    role_model_candidates_mvs = MultilayerGraphs.mv_neighbors(role_model_layer,
                                                              player_mv)

    if !isempty(role_model_candidates_mvs)

        # We need the ratio of the sum of all values in G for neighbouring 
        # nodes in the role_model layer which share the same strategy as the
        # player over the sum of all values in G among their neighbours.
        candidate_indices = [player_lookup[candidate.node.id]
                             for candidate in role_model_candidates_mvs]
        G_ab_neighbours = G_ab[player_lookup[player.node.id], candidate_indices]
        candidate_indices_shared = filter(id -> getindex(model, id).strategy ==
                                                player.strategy,
                                          candidate_indices)
        G_ab_shared = G_ab[player_lookup[player.node.id],
                           candidate_indices_shared]
        G_num = sum(G_ab_shared)
        G_den = sum(G_ab_neighbours)
        G_den = (G_den == 0) ? 1 : G_den
        G_ratio = G_num / G_den
        eta_x = 1 - (eta_max - eta_min) * G_ratio
        role_model_mv = rand(role_model_candidates_mvs)
        role_model = getindex(model, player_lookup[role_model_mv.node.id])
        # Player adopts their role_model's behaviour with chance given by the Fermi function
        player_payoff = player_payoffs[learner_layer.name][player.node.id]
        role_model_payoff = player_payoffs[role_model_layer.name][role_model.node.id]
        Px = player_payoff
        Py = role_model_payoff
        delta_xy = delta[player.id, role_model.id, learner_layer_index]
        adoption_likelihood = eta_x *
                              (1 + exp(beta * (Px - Py) / delta_xy))^(-1)

        if rand() < adoption_likelihood
            player.strategy = role_model.strategy
        end
    end
    return model
end

function process_results_data(results_data)
    # Outer join model data to agent data
    results_df = [DataFrames.innerjoin(df_pair[1], df_pair[2]; on=:step)
                  for df_pair in results_data]
    return vcat(results_df...)
end

function group_data_by_strategy(results_df)
    df = results_df
    grouped_df = DataFrames.groupby(df, [:step, :strategy])
    plot_df = DataFrames.combine(grouped_df, nrow => :strategy_count)
    plot_df[!, :strategy_count] = plot_df[!, :strategy_count] / n_replications

    # # Create a DataFrame with all combinations of group and strategy values
    all_strategies = 0:1
    all_combinations = unique([(g, s)
                               for g in unique(df.step), s in all_strategies])
    full_df = DataFrame(; step=[x[1] for x in all_combinations],
                        strategy=[x[2] for x in all_combinations])

    # # Perform an outer join to combine the data and fill missing values with 0
    plot_df = outerjoin(full_df, plot_df; on=[:step, :strategy])
    plot_df[!, :strategy_count] .= coalesce.(plot_df[!, :strategy_count], 0)
    return plot_df
end

function group_data_by_strategy_and_model_id(results_df)
    df = results_df
    group_by_columns = [:step, :strategy, :model_id]
    grouped_df = DataFrames.groupby(df, group_by_columns)
    plot_df = DataFrames.combine(grouped_df, nrow => :strategy_count)
    plot_df[!, :strategy_count] = plot_df[!, :strategy_count]

    # # Create a DataFrame with all combinations of group and strategy values
    all_strategies = 0:1
    all_combinations = unique([(g, s, m)
                               for g in unique(df.step),
                                   s in all_strategies,
                                   m in unique(df.model_id)])
    full_df = DataFrame(; step=[x[1] for x in all_combinations],
                        strategy=[x[2] for x in all_combinations],
                        model_id=[x[3] for x in all_combinations])

    # # Perform an outer join to combine the data and fill missing values with 0
    plot_df = outerjoin(full_df, plot_df; on=group_by_columns)
    plot_df[!, :strategy_count] .= coalesce.(plot_df[!, :strategy_count], 0)
    return plot_df
end

"""
A convenience function for running the desired model_fn and run_fn on
each set of parameters in dict_list. This is meant to be used in
conjunction with DrWatson.jl and Agents.jl but may work for a
wider range of uses. If you wish to save data during the simulation,
I recommend adding this to your run_fn.
"""
function run_simulations(configs)
    models = ProgressMeter.@showprogress "Computing..." [c["build_abm_fn"](c)
                                                         for c in configs]
    results = ProgressMeter.@showprogress "Computing..." [m["run_fn"](m)
                                                          for m in models]
    return results
end

ProgressMeter.@showprogress "Computing..." for i in 1:50
    sleep(0.1)
end

function build_abm(d::Dict{String,Any})
    @unpack influence_matrix, n_players, model_id = d
    model = pd_multiplex_model(; model_id=model_id, n_players=n_players,
                               b=8,
                               c=0.5,
                               beta=0.05,
                               critical_mass_threshold=(0.25 * n_players
                                                        |> floor
                                                        |> Int),
                               W_dist=influence_matrix)
    return Dict(d..., "abm_model" => model)
end

"""
A convenience function for running an ABM created using Agents.jl. 

Note: Agents.jl uses symbols to retrieve model data, but
when the data is collected, the columns of the dataframes are strings. This is
why we expect adata_cols and mdata_cols to have a different type than the keys
of d.
"""
function run_abm!(d::Dict{String,Any};
                  adata_cols::Vector{Symbol}=[:strategy],
                  mdata_cols::Vector{Symbol}=[:model_id],
                  cols_to_store::Union{Nothing,Vector{String}}=nothing)
    @unpack abm_model, n_steps = d

    agents_df, model_df = Agents.run!(abm_model,
                                      agent_step!,
                                      model_step!,
                                      n_steps;
                                      adata=adata_cols,
                                      mdata=mdata_cols)
    # For any data only in d and not in the abm_model, each row of model_df
    # will have the same value.
    for (k, v) in d
        # By default, we write all data in d to model_df, but if cols_to_store
        # is provided, we only add in those values.
        if isnothing(cols_to_store) || k in cols_to_store
            # We don't want to overwrite data already in model_df
            if !(k in DataFrames.names(model_df))
                model_df[!, k] .= [v]
            end
        end
    end
    return Dict(d..., "simulation_results" => [agents_df, model_df])
end

function replicate_configs(configs, n_replications)
    return [Dict(d..., "model_id" => m_id, "replication_id" => r_id)
            for (m_id, d) in enumerate(configs)
            for r_id in 1:n_replications]
end

# Run the simulation and collect results
all_params = Dict("n_players" => 10,
                  "n_steps" => 20,
                  "build_abm_fn" => build_abm,
                  "run_fn" => run_abm!,
                  "influence_matrix" => [Distributions.TruncatedNormal(0.45,
                                                                       100, 0.3,
                                                                       0.6),
                                         Distributions.TruncatedNormal(1, 1,
                                                                       0.5,
                                                                       0.5)])
configs = all_params |> DrWatson.dict_list |> (x -> replicate_configs(x, 40));
length(configs)
results = configs |> run_simulations;

dfs = [r["simulation_results"] for r in results]

df_example = results[1]["simulation_results"][2]
DataFrames.describe(df_example)

# Process results
results_df = process_results_data(dfs)
names(results_df)

# Save results
base_dir = DrWatson.projectdir("/home/ethos/git/abm-regulate-ai")
i = 3
DrWatson.datadir(base_dir,
                 "data",
                 "simulations",
                 "sim_$(i).jld2")
DrWatson.@tagsave(DrWatson.datadir(base_dir,
                                   "data/simulations",
                                   "sim_$(i).jld2"),
                  results_df |> eachcol |> pairs |> Dict)
DrWatson.wload(DrWatson.datadir(base_dir,
                                "data/simulations",
                                "sim_$(i).jld2"))

# # Plot, animate, and save results

# # Test figure
# anim = Plots.@animate for i in 1:n_steps
#     plot_df = results_df |> group_data_by_strategy
#     temp_df = DataFrames.filter(row -> (row.step<=i) & (row.strategy==1), plot_df)
#     temp_df[!, :influence_strength] .=  1
#     colorscale = Dict(zip(w_values, [:Red, :Blue]))
#     fillcolors = [colorscale[i] for i in temp_df[!, :influence_strength]]
#     Plots.plot(
#         temp_df[!,:step],
#         temp_df[!,:strategy_count],
#         fillcolor = fillcolors,
#         seriestype = :scatter,
#         xlims = [0, n_replications],
#         ylims = (0, n_players),
#         legend = false,
#         xlabel = "Time step",
#         ylabel = "Cooperators",
#         title =  "Multiplex Prisoner's Dilemma dynamics - Step $i")
# end

# Plots.gif(anim, "multiplex_pd_scatter.gif", fps = 10)

# # Figure 4 replication
# anim = Plots.@animate for i in 1:n_steps
#     plot_df = results_df |> group_data_by_strategy
#     temp_df = DataFrames.filter(row -> (row.step<=i) & (row.strategy==1), plot_df)
#     temp_df[!, :influence_strength] .=  1
#     w_values = 1:2
#     colorscale = Dict(zip(w_values, [:Red, :Blue]))
#     fillcolors = [colorscale[i] for i in temp_df[!, :influence_strength]]
#     Plots.plot(
#         temp_df[!,:step],
#         temp_df[!,:strategy_count],
#         fillcolor = fillcolors,
#         seriestype = :scatter,
#         xlims = [0, n_replications],
#         ylims = (0, n_players),
#         legend = false,
#         xlabel = "Time step",
#         ylabel = "Cooperators",
#         title =  "Multiplex Prisoner's Dilemma dynamics - Step $i")
# end

# Plots.gif(anim, "fig4_replication_distefano_homophily_and_critical_mass.gif", fps = 10)

# using Interpolations
# using StatsBase
# using ColorSchemes

# # Figure 5 replication
# anim = Plots.@animate for i in 1:n_steps
#     plot_df = results_df |> group_data_by_strategy_and_model_id
#     temp_df = DataFrames.filter(row -> (row.step<=i) & (row.strategy==1), plot_df)

#     x = temp_df[!,:step]
#     y = temp_df[!,:strategy_count]
#     kde_estimation = KernelDensity.kde((x, y))
#     interp_object = Interpolations.interpolate(kde_estimation.density, Interpolations.BSpline(Interpolations.Linear()))
#     i_f(x, y) = interp_object[x, y]

#     x_grid = kde_estimation.x
#     y_grid = kde_estimation.y

#     get_index(val, grid) = findfirst(x -> x ≥ val, grid)

#     density_values = [i_f(get_index(x[i], x_grid), get_index(y[i], y_grid)) for i in 1:length(x)]

#     # Normalize the density values to [0, 1]
#     density_values_normalized = (density_values .- minimum(density_values)) ./ (maximum(density_values) - minimum(density_values))

#     # Convert normalized density values to colors using a colormap
#     colors = get(ColorSchemes.colorschemes[:viridis], density_values_normalized)

#     temp_df[!, :influence_strength] .=  1
#     w_values = 1:2
#     colorscale = Dict(zip(w_values, [:Red, :Blue]))
#     fillcolors = [colorscale[i] for i in temp_df[!, :influence_strength]]
#     Plots.plot(x, y,
#         color = colors,
#         fillcolor = fillcolors,
#         seriestype = :scatter,
#         xlims = [0, n_replications],
#         ylims = (0, n_players),
#         legend = false,
#         xlabel = "Time step",
#         ylabel = "Cooperators",
#         title =  "Multiplex Prisoner's Dilemma dynamics - Step $i")
# end

# Plots.gif(anim, "fig5a_replication_distefano_homophily_and_critical_mass.gif", fps = 10)

# # Figure 6 Replication

# using GraphPlot
# using Compose
# using Cairo
# using Fontconfig

# begin
#     model_id = 1
#     layer_index = 1
#     model_graph = models[model_id].layers[layer_index].graph
#     layout_data = spring_layout(model_graph, C=1, MAXITER=1000)
#     fixed_layout(_) = layout_data
#     anim = Plots.@animate for i in 1:n_steps
#         plot_df = results_df
#         temp_df = DataFrames.filter(row -> (row.step<=i), plot_df)
#         temp_df = DataFrames.filter(row -> (row.model_id==model_id), temp_df)
#         all_strategies = 0:1
#         colorscale = Dict(zip(all_strategies, [:Red, :Blue]))
#         strategies_df = temp_df
#         strategies_dict = Dict(zip(strategies_df.id, strategies_df.strategy))
#         node_colors = [strategies_dict[i] == 1 ? colorant"blue" : colorant"red" for i in vertices(model_graph)]
#         gp = gplot(model_graph, nodefillc=node_colors, layout=fixed_layout)
#         draw(PNG("frame_$i.png"), gp)
#     end
#     # In your terminal, make sure ffmpeg is installed and run:
#     # ffmpeg -framerate 5 -i frame_%d.png -c:v gif network_animation.gif
# end
