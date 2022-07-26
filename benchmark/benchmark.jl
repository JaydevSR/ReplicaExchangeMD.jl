using ReplicaExchangeMD
using BenchmarkTools

## Set up the simulation
n_atoms = 1000
atom_mass = 10.0u"u"
atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms];
boundary = CubicBoundary(5.0u"nm", 5.0u"nm", 5.0u"nm")
coords = place_atoms(n_atoms, boundary, 0.3u"nm");

temp0 = 100.0u"K"
velocities = [velocity(atom_mass, temp0) for i in 1:n_atoms];

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

## Normal simulation on 2 threads

sys = System(
    atoms=atoms,
    pairwise_inters=pairwise_inters,
    coords=coords,
    velocities=velocities,
    boundary=boundary,
    neighbor_finder=neighbor_finder,
)

sim0 = Langevin(
    dt=0.005u"ps",
    temperature=temp0,
    friction=0.1u"ps^-1",
);

@time Molly.simulate!(sys, sim0, 100, n_threads=2)
@time Molly.simulate!(sys, sim0, 5_000; n_threads=2)

# Replica exchange simulation with 4 replicas on 8 threads

n_replicas = 4
temp_vals = [120.0u"K", 160.0u"K", 200.0u"K", 240.0u"K"]

replica_velocities = [random_velocities(sys, temp_vals[i]) for i in 1:n_replicas];
repsys = ReplicaSystem(
    atoms=atoms,
    coords=coords,
    replica_velocities=replica_velocities,
    n_replicas=n_replicas,
    boundary=boundary,
    pairwise_inters=pairwise_inters,
    neighbor_finder=neighbor_finder,
    log_exchanges=false,
)

repsim = TemperatureREMD(
    dt=0.005u"ps",
    temperatures=temp_vals,
    simulators=[
        Langevin(
            dt=0.005u"ps",
            temperature=tmp,
            friction=0.1u"ps^-1",
        )
        for tmp in temp_vals],
    exchange_time=2.5u"ps",
);

@time simulate!(repsys, repsim, 1_00; n_threads=8)
@time simulate!(repsys, repsim, 5_000; n_threads=8)

# Results

# 81.149985 seconds (582.40 k allocations: 4.966 GiB, 0.73% gc time)
# System with 1000 atoms, boundary CubicBoundary{Quantity{Float64, 𝐋 , Unitful.FreeUnits{(nm,), 𝐋 , nothing}}}(Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}  [5.0 nm, 5.0 nm, 5.0 nm])

# 196.544874 seconds (1.55 M allocations: 20.401 GiB, 2.21% gc time)
# ReplicaSystem containing 4 replicas with 1000 atoms, boundary CubicBoundary{Quantity{Float64, 𝐋 , Unitful.FreeUnits{(nm,), 𝐋, nothing}}}(Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}  [5.0 nm, 5.0 nm, 5.0 nm])