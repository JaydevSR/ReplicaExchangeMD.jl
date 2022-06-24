export TemperatureREMD
struct TemperatureREMD{N, T, S, SD}
    temps::StaticVector{N, T}
    indices::StaticVector{N, Int}
    simulators::SD
end

n_simulators(::TemperatureREMD{N}) where {N} = N 

function TemperatureREMD(;
    simulator,
    temps, dt, 
    kwargs...)
    S = simulator
    T = eltype(temps)
    N = length(temps)
    temps = SA[temps...]
    indices = SA[collect(1:length(temps))...]

    coupling_required = false
    if simulator ∈ [VelocityVerlet, Verlet, StormerVerlet]
        coupling_required = true
        coupling = kwargs[:coupling]
    end

    if coupling_required && coupling ∈ [AndersenThermostat, BerendsenThermostat]
        couplings = [coupling(temps[i], kwargs[:coupling_const]) for i in indices]
    elseif coupling_required && coupling ∈ [RescaleThermostat]
        couplings = [coupling(temps[i]) for i in indices]
    elseif coupling_required
        error("Temperature coupling required for TemperatureREMD using $(simulator) simulator.")
    end

    if simulator==Langevin
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], friction=kwargs[:friction],
            remove_CM_motion=kwargs[:remove_CM_motion])]
            for i in indices
        )
    elseif simulator ∈ [VelocityVerlet, Verlet]
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], coupling=couplings[i],
            remove_CM_motion=kwargs[:remove_CM_motion])]
            for i in indices
        )
    elseif simulator == StormerVerlet
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], coupling=couplings[i])]
            for i in indices
        )
    else
        error("Simulator type $(simulator) not supported.")
    end
    SD = typeof(simulators)

    return TemperatureREMD{N, T, S, SD}(temps, indices, simulators)
end

# simulate temperature replica exchange
function simulate!(;
    sys::ReplicaSystem,
    sim::TemperatureREMD,
    n_cycles::Int,
    cycle_length::Int;
    parallel=true
    )

    if sys.n_replicas != n_simulators(sim)
        error("Number of replicas in ReplicaSystem and simulators in TemperatureREMD must match.")
    end
    for cycle=1:n_cycles
        for idx in sim.indices  # FLAG: this needs to be parallelized
            # sys.replicas[i] is IndividualReplica not System
            simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; parallel=parallel)  # FLAG: not working
        end

        if cycle != n_cycles
            n = rand(1:sys.n_replicas)
            m = mod(n, sys.n_replicas) + 1
            β_n, β_m = 1/sim.temps[n], 1/sim.temps[m]
            # sys.replicas[i] is IndividualReplica not System
            V_n, V_m = potential_energy(sys.replicas[n]), potential_energy(sys.replicas[m])  # FLAG: not working
            Δ = (β_m - β_n)*(V_n - V_m)
            if Δ <= 0 || rand() < exp(-Δ)
                sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
                # scale velocities
                sys.replicas[n].velocities *= sqrt(β_n/β_m)
                sys.replicas[m].velocities *= sqrt(β_m/β_n)
            end
        end
    end
end