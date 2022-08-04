using ReplicaExchangeMD
using Profile
using PProf

##
n_atoms = 400
atom_mass = 10.0u"u"
atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms];
boundary = CubicBoundary(5.0u"nm", 5.0u"nm", 5.0u"nm")
coords = place_atoms(n_atoms, boundary, 0.3u"nm");

temp0 = 100.0u"K"
velocities = [velocity(atom_mass, temp0) for i in 1:n_atoms];

nb_matrix = trues(n_atoms, n_atoms);
for i in 1:(n_atoms ÷ 2)
    nb_matrix[i, i + (n_atoms ÷ 2)] = false
    nb_matrix[i + (n_atoms ÷ 2), i] = false
end

neighbor_finder = DistanceNeighborFinder(
    nb_matrix=nb_matrix,
    n_steps=5,
    dist_cutoff=1.5u"nm",
)

pairwise_inters = (LennardJones(nl_only=true),)
##

# Replica exchange simulation with 4 replicas on 8 threads

n_replicas = 4
temp_vals = [120.0u"K", 160.0u"K", 200.0u"K", 240.0u"K"]

repsys = ReplicaSystem(
    atoms=atoms,
    coords=coords,
    replica_velocities=nothing,
    n_replicas=n_replicas,
    boundary=boundary,
    pairwise_inters=pairwise_inters,
    neighbor_finder=neighbor_finder,
    exchange_logger=nothing,
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

##

Profile.clear()

simulate!(repsys, repsim, 1000; n_threads=8, assign_velocities=true)
@profile simulate!(repsys, repsim, 1000; n_threads=8)

pprof()
