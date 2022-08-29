using ReplicaExchangeMD
using Statistics
using Random
using Test

@testset "Replica System" begin
    n_atoms = 100
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm") 
    temp = 298.0u"K"
    atom_mass = 10.0u"u"
    
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary, 0.3u"nm")
    replica_velocities = nothing
    pairwise_inters = (LennardJones(nl_only=true),)
    n_replicas = 4

    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=nb_matrix,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    repsys = ReplicaSystem(;
        n_replicas=n_replicas,
        atoms=atoms,
        coords=coords,
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
    )

    sys = System(;
        atoms=atoms,
        coords=coords,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
    )

    for i in 1:n_replicas
        @test all(
            [getfield(repsys.replicas[i], f) for f in fieldnames(System)] .== [getfield(sys, f) for f in fieldnames(System)]
        )
    end

    repsys2 = ReplicaSystem(;
        n_replicas=n_replicas,
        atoms=atoms,
        coords=coords,
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
        replica_loggers=[(temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas],
        neighbor_finder=neighbor_finder,
    )

    sys2 = System(;
        atoms=atoms,
        coords=coords,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
        loggers=(temp=TemperatureLogger(10), coords=CoordinateLogger(10)),
        neighbor_finder=neighbor_finder,
    )

    for i in 1:n_replicas
        l1 = repsys2.replicas[i].loggers
        l2 = sys2.loggers
        @test typeof(l1) == typeof(l2)
        @test propertynames(l1) == propertynames(l2)

        nf1 = [getproperty(repsys2.replicas[i].neighbor_finder, p) for p in propertynames(repsys2.replicas[i].neighbor_finder)]
        nf2 = [getproperty(sys2.neighbor_finder, p) for p in propertynames(sys2.neighbor_finder)]
        @test all(nf1 .== nf2)
    end
end

@testset "Temperature REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    pairwise_inters = (LennardJones(nl_only=true),)

    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=nb_matrix,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    replica_loggers = [(temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=nothing,
        n_replicas=n_replicas,
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        replica_loggers=replica_loggers,
        neighbor_finder=neighbor_finder,
    )

    temp_vals = [120.0u"K", 180.0u"K", 240.0u"K", 300.0u"K"]
    simulator = TemperatureREMD(
        dt=0.005u"ps",
        temperatures=temp_vals,
        simulators=[
            Langevin(
                dt=0.005u"ps",
                temperature=temp,
                friction=0.1u"ps^-1",
            )
            for temp in temp_vals],
        exchange_time=2.5u"ps",
    )

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true )
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold, but it's a good tests for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"

    for id in eachindex(repsys.replicas)
        mean_temp = mean(values(repsys.replicas[id].loggers.temp))
        @test (0.9 * temp_vals[id]) < mean_temp < (1.1 * temp_vals[id])
    end
end

@testset "Hamiltonian REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    temp = 100.0u"K"
    velocities = [velocity(10.0u"u", temp) for i in 1:n_atoms]

    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=nb_matrix,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    λ_vals = [0.0, 0.1, 0.25, 0.4]
    replica_pairwise_inters = [(LennardJonesSoftCore(α=1, λ=λ_vals[i], p=2, nl_only=true),) for i in 1:n_replicas]

    replica_loggers = [(temp=TemperatureLogger(10), ) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=nothing,
        n_replicas=n_replicas,
        boundary=boundary,
        replica_pairwise_inters=replica_pairwise_inters,
        replica_loggers=replica_loggers,
        neighbor_finder=neighbor_finder,
    )

    simulator = HamiltonianREMD(
        dt=0.005u"ps",
        temperature=temp,
        simulators=[
            Langevin(
                dt=0.005u"ps",
                temperature=temp,
                friction=0.1u"ps^-1",
            )
            for _ in 1:n_replicas],
        exchange_time=2.5u"ps",
    )

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true )
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold, but it's a good tests for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"
    for id in eachindex(repsys.replicas)
        mean_temp = mean(values(repsys.replicas[id].loggers.temp))
        @test (0.9 * temp) < mean_temp < (1.1 * temp)
    end

    # TODO: Possibly more tests?
end
