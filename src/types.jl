using Molly

mutable struct ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, RI, B, NF, L, F, E} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    replicas::RS
    box_size::B
    neighbor_finder::NF
    replica_loggers::L
    force_units::F
    energy_units::E
end

mutable struct IndividualReplica{C, V}
    index::Int
    coords::C
    velocities::V
end

function ReplicaSystem(;
    atoms,
    atoms_data=[],
    pairwise_inters=(),
    specific_inter_lists=(),
    general_inters=(),
    coords,
    velocities=zero(coords) * u"ps^-1",
    n_replicas,
    box_size,
    neighbor_finder=NoNeighborFinder(),
    loggers=Dict(),
    force_units=u"kJ * mol^-1 * nm^-1",
    energy_units=u"kJ * mol^-1",
    gpu_diff_safe=isa(coords, CuArray)
    )

    D = length(box_size)
    G = gpu_diff_safe
    T = typeof(ustrip(first(box_size)))
    A = typeof(atoms)
    AD = typeof(atoms_data)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    B = typeof(box_size)
    NF = typeof(neighbor_finder)
    F = typeof(force_units)
    E = typeof(energy_units)

    replicas = [IndividualReplica(i, coords, velocities) for i=1:n_replicas]
    RS = typeof(replica_list)

    replica_loggers = Dict(["$i", copy(loggers)] for i=1:n_replicas)
    RL = typeof(replica_loggers)

    return ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, B, NF, L, F, E}(
            atoms, atoms_data, pairwise_inters, specific_inter_lists,
            general_inters, replicas, box_size, neighbor_finder,
            replica_loggers, force_units, energy_units)
end