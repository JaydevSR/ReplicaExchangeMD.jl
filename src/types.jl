export ReplicaSystem

mutable struct ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, B, NF, L, F, E} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    n_replicas::Integer
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
    C = typeof(coords)
    V = typeof(velocities)

    replicas = Dict([i, IndividualReplica{C, V}(i, coords, velocities)] for i=1:n_replicas)
    RS = typeof(replicas)

    replica_loggers = Dict([i, copy(loggers)] for i=1:n_replicas)
    L = typeof(replica_loggers)

    return ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, B, NF, L, F, E}(
            atoms, atoms_data, pairwise_inters, specific_inter_lists,
            general_inters, n_replicas, replicas, box_size, neighbor_finder,
            replica_loggers, force_units, energy_units)
end

is_gpu_diff_safe(::ReplicaSystem{D, G}) where {D, G} = G

float_type(::ReplicaSystem{D, G, T}) where {D, G, T} = T

AtomsBase.species_type(s::ReplicaSystem) = eltype(s.atoms)

Base.getindex(s::ReplicaSystem, i::Integer) = AtomView(s, i)
Base.length(s::ReplicaSystem) = length(s.atoms)

AtomsBase.position(s::ReplicaSystem, ri::Integer) = s.replicas["$ri"].coords
AtomsBase.position(s::ReplicaSystem, ri::Integer, i::Integer) = s.replicas["$ri"].coords[i]
AtomsBase.position(s::ReplicaSystem) = error("Replica index required.")

AtomsBase.velocity(s::ReplicaSystem, ri::Integer) = s.replicas["$ri"].velocities
AtomsBase.velocity(s::ReplicaSystem, ri::Integer, i::Integer) = s.replicas["$ri"].velocities[i]
AtomsBase.velocity(s::ReplicaSystem) = error("Replica index required.")

AtomsBase.atomic_mass(s::ReplicaSystem, i::Integer) = mass(s.atoms[i])
AtomsBase.atomic_symbol(s::ReplicaSystem, i::Integer) = Symbol(s.atoms_data[i].element)
AtomsBase.atomic_number(s::ReplicaSystem, i::Integer) = missing

AtomsBase.boundary_conditions(::ReplicaSystem{3}) = SVector(Periodic(), Periodic(), Periodic())
AtomsBase.boundary_conditions(::ReplicaSystem{2}) = SVector(Periodic(), Periodic())

edges_to_box(bs::SVector{3}, z) = SVector{3}([
    SVector(bs[1], z    , z    ),
    SVector(z    , bs[2], z    ),
    SVector(z    , z    , bs[3]),
])
edges_to_box(bs::SVector{2}, z) = SVector{2}([
    SVector(bs[1], z    ),
    SVector(z    , bs[2]),
])

function AtomsBase.bounding_box(s::ReplicaSystem)
    bs = s.box_size
    z = zero(bs[1])
    bb = edges_to_box(bs, z)
    return unit(z) == NoUnits ? (bb)u"nm" : bb # Assume nm without other information
end

function Base.show(io::IO, s::ReplicaSystem)
    print(io, "ReplicaSystem containing ",  s.n_replicas, " replicas with ", length(s), " atoms, box size ", s.box_size)
end