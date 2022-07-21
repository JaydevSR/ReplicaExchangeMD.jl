export TemperatureREMD
struct TemperatureREMD{N, T, S, ST, ET, DT}
    temps::StaticVector{N, T}
    simulators::ST
    exchange_time::ET
    dt::DT
end

function TemperatureREMD(;
                temps,
                simulators,
                exchange_time,
                dt, kwargs...)
    S = eltype(simulators)
    T = eltype(temps)
    N = length(temps)
    ET = typeof(exchange_time)
    DT = typeof(dt)
    temps = SA[temps...]
    if length(simulators) != length(temps)
        error("Number of simulators and temperatures values must be equal.")
    end
    simulators = Tuple(simulators[i] for i in 1:N)
    ST = typeof(simulators)
    
    return TemperatureREMD{N, T, S, ST, ET, DT}(temps, simulators, exchange_time, dt)
end

# simulate temperature replica exchange
function simulate!(sys::ReplicaSystem,
                    sim::TemperatureREMD,
                    n_steps::Int;
                    n_threads::Int=Threads.nthreads())
    if sys.n_replicas != length(sim.simulators)
        error("Number of replicas in ReplicaSystem and simulators in TemperatureREMD must match.")
    end

    if n_threads > sys.n_replicas
        thread_div = equal_parts(n_threads, sys.n_replicas)
    else # pass 1 thread per replica
        thread_div = equal_parts(sys.n_replicas, sys.n_replicas)
    end

    # calculate n_cycles and n_steps_per_cycle from dt and exchange_time
    n_cycles = convert(Int64, (n_steps * sim.dt) ÷ sim.exchange_time)
    cycle_length = n_steps ÷ n_cycles
    remaining_steps = n_steps % n_cycles
    n_exchanges = 0

    # scale to correct temperatures
    for i in eachindex(sys.replicas)
        sys.replicas[i].velocities .*= sqrt(sim.temps[i] / temperature(sys.replicas[i]))
    end

    for cycle=1:n_cycles
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; n_threads=thread_div[idx])
        end

        if cycle != n_cycles
            n = rand(1:sys.n_replicas)
            m = mod(n, sys.n_replicas) + 1
            k_b = sys.k
            β_n, β_m = 1/(k_b*sim.temps[n]), 1/(k_b*sim.temps[m])
            V_n, V_m = potential_energy(sys.replicas[n]), potential_energy(sys.replicas[m])
            Δ = ustrip((β_m - β_n)*(V_n - V_m))
            if Δ <= 0 || rand() < exp(-Δ)
                n_exchanges += 1
                sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
                # scale velocities
                sys.replicas[n].velocities .*= sqrt(β_n/β_m)
                sys.replicas[m].velocities .*= sqrt(β_m/β_n)
            end
        end
    end

    # run for remaining_steps (if >0) for all replicas
    if remaining_steps > 0
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps; n_threads=thread_div[idx])
        end
    end
    return sys
end
