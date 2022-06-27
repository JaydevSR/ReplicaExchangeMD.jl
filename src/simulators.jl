export TemperatureREMD
struct TemperatureREMD{N, T, S, SD, ET, DT}
    temps::StaticVector{N, T}
    simulators::SD
    exchange_time::ET
    dt::DT
end

n_simulators(::TemperatureREMD{N}) where {N} = N 

function TemperatureREMD(;
    simulators, exchange_time, dt, temps,
    kwargs...)
    S = eltype(simulators)
    T = eltype(temps)
    N = length(temps)
    ET = typeof(exchange_time)
    DT = typeof(dt)
    temps = SA[temps...]
    if length(simulators) != length(temps)
        error("Number of simulators and temperatures values must be equal.")
    end
    simulators = Dict([i, simulators[i]] for i in 1:N)
    SD = typeof(simulators)
    
    return TemperatureREMD{N, T, S, SD, ET}(temps, simulators, exchange_time, dt)
end

# simulate temperature replica exchange
function simulate!(;
    sys::ReplicaSystem,
    sim::TemperatureREMD,
    n_steps,
    parallel=true
    )

    if sys.n_replicas != length(sim.simulators)
        error("Number of replicas in ReplicaSystem and simulators in TemperatureREMD must match.")
    end
    # calculate n_cycles and n_steps_per_cycle from dt and exchange_time
    n_cycles = (n_steps * sim.dt) ÷ sim.exchange_time
    cycle_length = n_steps ÷ n_cycles
    remaining_steps = n_steps % n_cycles

    for cycle=1:n_cycles
        for idx in eachindex(sim.simulators)  # FLAG: this needs to be parallelized
            simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; parallel=parallel)  # FLAG: check if this is working
        end

        if cycle != n_cycles
            n = rand(1:sys.n_replicas)
            m = mod(n, sys.n_replicas) + 1
            β_n, β_m = 1/sim.temps[n], 1/sim.temps[m]
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
    for idx in eachindex(sim.simulators)  # FLAG: this needs to be parallelized
        simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps; parallel=parallel)  # FLAG: check if this is working
    end
end