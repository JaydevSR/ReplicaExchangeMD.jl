export ReplicaSystem

# A system to be simulated using replica exchage
mutable struct ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, B, NF, F, E, K} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    n_replicas::Integer
    replicas::RS
    boundary::B
    neighbor_finder::NF
    force_units::F
    energy_units::E
    k::K
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
    boundary,
    neighbor_finder=NoNeighborFinder(),
    loggers=(),
    force_units=u"kJ * mol^-1 * nm^-1",
    energy_units=u"kJ * mol^-1",
    k=Unitful.k,
    gpu_diff_safe=isa(coords, CuArray)
    )

    D = n_dimensions(boundary)
    G = gpu_diff_safe
    T = float_type(boundary)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    C = typeof(coords)
    V = typeof(velocities)
    B = typeof(boundary)
    NF = typeof(neighbor_finder)
    L = typeof(loggers)
    F = typeof(force_units)
    E = typeof(energy_units)

    if energy_units == NoUnits
        if unit(k) == NoUnits
            # Use user-supplied unitless Boltzmann constant
            k_converted = T(k)
        else
            # Otherwise assume energy units are (u* nm^2 * ps^-2)
            k_converted = T(ustrip(u"u * nm^2 * ps^-2 * K^-1", k))
        end
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k_converted = T(uconvert(energy_units * u"mol * K^-1", k))
    else
        k_converted = T(uconvert(energy_units * u"K^-1", k))
    end
    
    K = typeof(k_converted)

    # FLAGS
    # Copy of neighbor_finder to support CellListNeighborFinder which stores data 
    replicas = Dict([i, System{D, G, T, A, AD, PI, SI, GI, C, V, B, NF, L, F, E, K}(
            atoms, atoms_data, pairwise_inters, specific_inter_lists,
            general_inters, copy(coords), copy(velocities), boundary, neighbor_finder,
            copy(loggers), force_units, energy_units, k_converted)] for i=1:n_replicas)
    
    RS = typeof(replicas)

    return ReplicaSystem{D, G, T, A, AD, PI, SI, GI, RS, B, NF, F, E, K}(
            atoms, atoms_data, pairwise_inters, specific_inter_lists,
            general_inters, n_replicas, replicas, boundary, neighbor_finder,
            force_units, energy_units, k_converted)
end

is_gpu_diff_safe(::ReplicaSystem{D, G}) where {D, G} = G

float_type(::ReplicaSystem{D, G, T}) where {D, G, T} = T

AtomsBase.species_type(s::ReplicaSystem) = eltype(s.atoms)

Base.getindex(s::ReplicaSystem, i::Integer) = AtomView(s, i)
Base.length(s::ReplicaSystem) = length(s.atoms)

AtomsBase.position(s::ReplicaSystem) = s.replicas[1].coords
AtomsBase.position(s::ReplicaSystem, i::Integer) = s.replicas[1].coords[i]

AtomsBase.velocity(s::ReplicaSystem) = s.replicas[1].velocities
AtomsBase.velocity(s::ReplicaSystem, i::Integer) = s.replicas[1].velocities[i]

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
    bs = s.boundary.side_lengths
    z = zero(bs[1])
    bb = edges_to_box(bs, z)
    return unit(z) == NoUnits ? (bb)u"nm" : bb # Assume nm without other information
end

function Base.show(io::IO, s::ReplicaSystem)
    print(io, "ReplicaSystem containing ",  s.n_replicas, " replicas with ", length(s), " atoms, boundary ", s.boundary)
end