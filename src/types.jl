export ReplicaSystem

"""
    ReplicaSystem(; <keyword arguments>)

A wrapper for replicas in a replica exchange simulation.
Each individual replica is a [`System`](@ref).
Properties unused in the simulation or in analysis can be left with their default values.
`atoms`, `atoms_data` and the elements in `replica_coords` and `replica_velocities`
should have the same length.
The number of elements in `replica_coords`, `replica_velocities`, `replica_loggers` and 
the interaction arguments `replica_pairwise_inters`, `replica_specific_inter_lists`, 
`replica_general_inters` and `replica_constraints` should be equal to `n_replicas`.
This is a sub-type of `AbstractSystem` from AtomsBase.jl and implements the
interface described there.

# Arguments
- `atoms::A`: the atoms, or atom equivalents, in the system. Can be
    of any type but should be a bits type if the GPU is used.
- `atoms_data::AD`: other data associated with the atoms, allowing the atoms to
    be bits types and hence work on the GPU.
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e. interactions 
    between all or most atom pairs such as electrostatics (to be used if same for all replicas).
    Typically a `Tuple`. *Note: This is only used if no value is passed to the argument 
    `replica_pairwise_inters`*.
- `replica_pairwise_inters=[() for _ in 1:n_replicas]`: the pairwise interactions for 
    each replica.
- `specific_inter_lists::SI=()`: the specific interactions in the system, i.e. interactions 
    between specific atoms such as bonds or angles (to be used if same for all replicas). 
    Typically a `Tuple`. *Note: This is only used if no value is passed to the argument 
    `replica_specific_inter_lists`*.
- `replica_specific_inter_lists=[() for _ in 1:n_replicas]`: the specific interactions in 
    each replica.
- `general_inters::GI=()`: the general interactions in the system, i.e. interactions involving 
    all atoms such as implicit solvent (to be used if same for all replicas). Typically a `Tuple`. 
    *Note: This is only used if no value is passed to the argument `replica_general_inters`*.
- `replica_general_inters=[() for _ in 1:n_replicas]`: the general interactions for 
    each replica.
- `constraints::CN=()`: the constraints for bonds and angles in the system (to be used if same 
    for all replicas). Typically a `Tuple`.
- `replica_constraints=[() for _ in 1:n_replicas]`: the constraints for bonds and angles in each
    replica. *Note: This is only used if no value is passed to the argument `replica_constraints`*.
- `n_replicas::Integer`: the number of replicas of the system.
- `replica_coords`: the coordinates of the atoms in each replica.
- `replica_velocities=[zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]`:
    the velocities of the atoms in each replica.
- `boundary::B`: the bounding box in which the simulation takes place.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `exchange_logger::EL=ReplicaExchangeLogger(n_replicas)`: the logger used to record
    the exchange of replicas.
- `replica_loggers=[() for _ in 1:n_replicas]`: the loggers for each replica 
    that record properties of interest during a simulation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k`: the Boltzmann constant, which may be modified in some
    simulations.
- `gpu_diff_safe::Bool`: whether to use the code path suitable for the
    GPU and taking gradients. Defaults to `isa(replica_coords[1], CuArray)`.
"""
mutable struct ReplicaSystem{D, G, T, CU, A, AD, RS, B, EL, F, E, K} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    n_replicas::Int
    replicas::RS
    boundary::B
    exchange_logger::EL
    force_units::F
    energy_units::E
    k::K
end

function ReplicaSystem(;
                        atoms,
                        atoms_data=[],
                        pairwise_inters=(),
                        replica_pairwise_inters=nothing,
                        specific_inter_lists=(),
                        replica_specific_inter_lists=nothing,
                        general_inters=(),
                        replica_general_inters=nothing,
                        constraints=(),
                        replica_constraints=nothing,
                        n_replicas,
                        replica_coords,
                        replica_velocities=nothing,
                        boundary,
                        neighbor_finder=NoNeighborFinder(),
                        exchange_logger=nothing,
                        replica_loggers=[() for _ in 1:n_replicas],
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        k=Unitful.k,
                        gpu_diff_safe=isa(replica_coords[1], CuArray))
    D = n_dimensions(boundary)
    G = gpu_diff_safe
    T = float_type(boundary)
    CU = isa(replica_coords[1], CuArray)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    C = typeof(replica_coords[1])
    B = typeof(boundary)
    NF = typeof(neighbor_finder)
    F = typeof(force_units)
    E = typeof(energy_units)

    if isnothing(replica_velocities)
        if force_units == NoUnits
            replica_velocities = [zero(replica_coords[1]) for _ in 1:n_replicas]
        else
            replica_velocities = [zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]
        end
    end
    V = typeof(replica_velocities[1])

    if isnothing(exchange_logger)
        exchange_logger = ReplicaExchangeLogger(T, n_replicas)
    end
    EL = typeof(exchange_logger)
    
    if isnothing(replica_pairwise_inters)
        replica_pairwise_inters = [pairwise_inters for _ in 1:n_replicas]
    elseif length(replica_pairwise_inters) != n_replicas
        throw(ArgumentError("Number of pairwise interactions ($(length(replica_pairwise_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    if isnothing(replica_specific_inter_lists)
        replica_specific_inter_lists = [specific_inter_lists for _ in 1:n_replicas]
    elseif length(replica_specific_inter_lists) != n_replicas
        throw(ArgumentError("Number of specific interaction lists ($(length(replica_specific_inter_lists)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    if isnothing(replica_general_inters)
        replica_general_inters = [general_inters for _ in 1:n_replicas]
    elseif length(replica_general_inters) != n_replicas
        throw(ArgumentError("Number of general interactions ($(length(replica_general_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    PI = eltype(replica_pairwise_inters)
    SI = eltype(replica_specific_inter_lists)
    GI = eltype(replica_general_inters)

    if isnothing(replica_constraints)
        replica_constraints = [constraints for _ in 1:n_replicas]
    elseif length(replica_constraints) != n_replicas
        throw(ArgumentError("Number of constraints ($(length(replica_general_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end
    CN = eltype(replica_constraints)
    
    if !all(y -> typeof(y) == C, replica_coords)
        throw(ArgumentError("The coordinates for all the replicas are not of the same type"))
    end
    if !all(y -> typeof(y) == V, replica_velocities)
        throw(ArgumentError("The velocities for all the replicas are not of the same type"))
    end

    if length(replica_coords) != n_replicas
        throw(ArgumentError("There are $(length(replica_coords)) coordinates for replicas but $n_replicas replicas"))
    end
    if length(replica_velocities) != n_replicas
        throw(ArgumentError("There are $(length(replica_velocities)) velocities for replicas but $n_replicas replicas"))
    end
    if length(replica_loggers) != n_replicas
        throw(ArgumentError("There are $(length(replica_loggers)) loggers but $n_replicas replicas"))
    end

    if !all(y -> length(y) == length(replica_coords[1]), replica_coords)
        throw(ArgumentError("Some replicas have different number of coordinates"))
    end
    if !all(y -> length(y) == length(replica_velocities[1]), replica_velocities)
        throw(ArgumentError("Some replicas have different number of velocities"))
    end

    if length(atoms) != length(replica_coords[1])
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(replica_coords[1])) coordinates"))
    end
    if length(atoms) != length(replica_velocities[1])
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(replica_velocities[1])) velocities"))
    end
    if length(atoms_data) > 0 && length(atoms) != length(atoms_data)
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(atoms_data)) atom data entries"))
    end

    n_cuarray = sum(y -> isa(y, CuArray), replica_coords)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("The coordinates for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, CuArray) && n_cuarray != n_replicas
        throw(ArgumentError("The atoms are on the GPU but the coordinates are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, CuArray)
        throw(ArgumentError("The coordinates are on the GPU but the atoms are not"))
    end

    n_cuarray = sum(y -> isa(y, CuArray), replica_velocities)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("The velocities for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, CuArray) && n_cuarray != n_replicas
        throw(ArgumentError("The atoms are on the GPU but the velocities are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, CuArray)
        throw(ArgumentError("The velocities are on the GPU but the atoms are not"))
    end

    k_converted = convert_k_units(T, k, energy_units)
    K = typeof(k_converted)

    replicas = Tuple(System{D, G, T, CU, A, AD, PI, SI, GI, CN, C, V, B, NF, typeof(replica_loggers[i]), F, E, K}(
            atoms, atoms_data, replica_pairwise_inters[i], replica_specific_inter_lists[i],
            replica_general_inters[i], replica_constraints[i], replica_coords[i], 
            replica_velocities[i], boundary, deepcopy(neighbor_finder), replica_loggers[i],
            force_units, energy_units, k_converted) for i in 1:n_replicas)
    RS = typeof(replicas)

    return ReplicaSystem{D, G, T, CU, A, AD, RS, B, EL, F, E, K}(
            atoms, atoms_data, n_replicas, replicas, boundary, 
            exchange_logger, force_units, energy_units, k_converted)
end

Molly.is_gpu_diff_safe(::ReplicaSystem{D, G}) where {D, G} = G

Molly.float_type(::ReplicaSystem{D, G, T}) where {D, G, T} = T

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

# Convert the Boltzmann constant k to suitable units and float type
function convert_k_units(T, k, energy_units)
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
    return k_converted
end

function AtomsBase.bounding_box(s::ReplicaSystem)
    bs = s.boundary.side_lengths
    z = zero(bs[1])
    bb = Molly.edges_to_box(bs, z)
    return unit(z) == NoUnits ? (bb)u"nm" : bb # Assume nm without other information
end

function Base.show(io::IO, s::ReplicaSystem)
    print(io, "ReplicaSystem containing ",  s.n_replicas, " replicas with ", length(s), " atoms, boundary ", s.boundary)
end