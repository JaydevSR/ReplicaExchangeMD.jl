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
    pairwise_inters = (LennardJones(),)
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
        replica_loggers=Tuple((temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas),
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

@testset "Temperature Replica Exchange" begin
    n_atoms = 100
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
    coords = place_atoms(n_atoms, boundary, 0.3u"nm")

    pairwise_inters = (LennardJones(),)

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

    repsys = ReplicaSystem(
        atoms=atoms,
        coords=coords,
        replica_velocities=nothing,
        n_replicas=n_replicas,
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        replica_loggers=Tuple((temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas),
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

    rng = MersenneTwister()
    @time simulate!(repsys, simulator, 10_000; assign_velocities=true, rng=rng);
    @time simulate!(repsys, simulator, 10_000; assign_velocities=false, rng=rng);

    exchange_performance = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test exchange_performance > 0.4
end