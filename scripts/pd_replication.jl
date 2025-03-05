
# Next, we should use Agents.jl to simulate the change in strategies played
# in a prisoner's dilemma over time.

using Agents
using Random
using Graphs
using MultilayerGraphs

# First, let's define the Agent types.

# Define the agent type
mutable struct player <: AbstractAgent
    id::Int                       # Unique identifier for each agent
    node::Node                 # Unique identifier for its node
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
    layer = MultilayerGraphs.layer_simplegraph(layer_id, mv_list, me_list, )
    return layer
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
function pd_multiplex_model(;
    n_players = 1000,
    layer_ids = [:layer1, :layer2, :layer3],
    b = 3,
    c = 1,
    beta = 1,
    eta = 1,
    delta = 1,
    seed = 6998,
)
    graphs = [Graphs.SimpleGraphs.static_scale_free(n_players, 50, 2)
              for _ in layer_ids]
    layers = simple_graph_to_layer.(graphs, layer_ids)
    multiplex = MultilayerGraphs.MultilayerGraph(layers)

    mv_lookup = Dict()
    player_payoffs_initial = Dict()
    for layer in layers
        mvs = layer |> MultilayerGraphs.mv_vertices
        layer_node_ids = MultilayerGraphs.id.(MultilayerGraphs.node.(mvs))
        mv_lookup[layer.name] = Dict(zip(layer_node_ids, mvs))
        player_payoffs_initial[layer.name] = Dict(zip(layer_node_ids, zeros(length(layer_node_ids))))
    end

    model = ABM(
        player,
        nothing; # No space needed as graph passed in as a property
        agent_step! = agent_step!,
        model_step! = model_step!,
        properties = Dict(
            :multilayer_network => multiplex,
            :b => b,
            :c => c,
            :beta => beta,
            :eta => eta,
            :delta => delta,
            :mv_lookup => mv_lookup,
            :player_lookup => Dict(),
            :player_payoffs => player_payoffs_initial,
        ),
        rng = Random.MersenneTwister(seed)
    )

    for node in nodes(multiplex)
        # Each player is a node of the multiplex.
        initial_strategy = rand(0:1)
        add_agent!(model, node, initial_strategy)
    end
    # Construct a lookup from node_id to agent_id
    agent_ids = getproperty.(Agents.allagents(model), :id)
    agent_node_ids = MultilayerGraphs.id.(getproperty.(Agents.allagents(model), :node))
    player_lookup = Dict(zip(agent_node_ids, agent_ids))
    model.player_lookup = player_lookup

    return model
end

"""
In each round, all players play a prisoner's dilemma with each of their
neighbours, earning a given payoff. We achieve this by iterating through all
of the edges within each layer of the multiplex. 
"""
function model_step!(model)
    multiplex = model.multilayer_network
    player_lookup = model.player_lookup
    b = model.b
    c = model.c
    layers = [getproperty(multiplex, l.name) for l in multiplex.layers]

    player_payoffs = Dict()
    for layer in layers
        mvs = layer |> MultilayerGraphs.mv_vertices
        layer_node_ids = MultilayerGraphs.id.(MultilayerGraphs.node.(mvs))
        player_payoffs[layer.name] = Dict(zip(layer_node_ids, zeros(length(layer_node_ids))))
        for edge in Graphs.edges(layer)

            # Retrieve the relevant players
            src_id = edge |> Graphs.src |> MultilayerGraphs.node |> MultilayerGraphs.id
            dst_id = edge |> Graphs.dst |> MultilayerGraphs.node |> MultilayerGraphs.id
            src_player = getindex(model, player_lookup[src_id])
            dst_player = getindex(model, player_lookup[dst_id])

            # Compute payoffs to each player given their strategies. Record
            # these behaviours in the payoffs property of the model.
            if (src_player.strategy == 1) & (dst_player.strategy == 1)
                src_player_payoff = b - c
                dst_player_payoff = b - c
            elseif (src_player.strategy == 1) & (dst_player.strategy == 0)
                src_player_payoff = -c
                dst_player_payoff = b
            elseif (src_player.strategy == 0) & (dst_player.strategy == 1)
                src_player_payoff = b
                dst_player_payoff = -c
            elseif (src_player.strategy == 0) & (dst_player.strategy == 0)
                src_player_payoff = 0
                dst_player_payoff = 0
            end
            
            player_payoffs[layer.name][src_player.node.id] = get(player_payoffs[layer.name], src_player.node.id, 0) + src_player_payoff
            player_payoffs[layer.name][dst_player.node.id] = get(player_payoffs[layer.name], dst_player.node.id, 0) + dst_player_payoff
        end
    end
    model.player_payoffs = player_payoffs
    return model
end

"""
The agent step involves an agent considering whether to adopt a different
strategy than they currently do. All agents go through this step.
"""
function agent_step!(player, model)
    multiplex = model.multilayer_network
    player_lookup = model.player_lookup
    mv_lookup = model.mv_lookup
    player_payoffs = model.player_payoffs
    beta = model.beta
    eta = model.eta
    delta = model.delta
    layers = [getproperty(multiplex, l.name) for l in multiplex.layers]
    learner_layer = rand(layers)
    role_model_layer = rand(layers)

    # Find a neighbour on the role_model_layer to be the player's role model
    player_mv = mv_lookup[role_model_layer.name][player.node.id]
    role_model_candidates = MultilayerGraphs.mv_neighbors(role_model_layer, player_mv)

    if !isempty(role_model_candidates)
        role_model_mv = rand(role_model_candidates)
        role_model = getindex(model, player_lookup[role_model_mv.node.id])
        # Player adopts their role_model's behaviour with chance given by the Fermi function
        player_payoff = player_payoffs[learner_layer.name][player.node.id]
        role_model_payoff = player_payoffs[role_model_layer.name][role_model.node.id]
        Px = player_payoff
        Py = role_model_payoff
        w = eta * (1 + exp(beta * (Px - Py) / delta))^(-1)

        if w > rand()
            player.strategy = role_model.strategy
        end
    end

    return model
end


model = pd_multiplex_model(n_players=20)


# Run the simulation for a number of steps
results_df = run!(model, 1; adata=[:strategy], mdata=[:b])
