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
                    n_steps::Integer;
                    parallel::Bool=true)
    if sys.n_replicas != length(sim.simulators)
        error("Number of replicas in ReplicaSystem and simulators in TemperatureREMD must match.")
    end
    # calculate n_cycles and n_steps_per_cycle from dt and exchange_time
    n_cycles = convert(Int64, (n_steps * sim.dt) ÷ sim.exchange_time)
    cycle_length = n_steps ÷ n_cycles
    remaining_steps = n_steps % n_cycles
    n_exchanges = 0
    @info "Total number of cycles: $n_cycles (cycle length: $cycle_length, remaining steps: $remaining_steps)"

    for cycle=1:n_cycles
        @sync for idx in eachindex(sim.simulators)
            if parallel
                Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; parallel=parallel)
            else
                Molly.simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; parallel=parallel)
            end
        end

        #! Check replica exchage algorithm
        if cycle != n_cycles
            n = rand(1:sys.n_replicas)
            m = mod(n, sys.n_replicas) + 1
            β_n, β_m = 1/sim.temps[n], 1/sim.temps[m]
            V_n, V_m = potential_energy(sys.replicas[n]), potential_energy(sys.replicas[m])  # FLAG: not working
            Δ = ustrip((β_m - β_n)*(V_n - V_m))
            if Δ <= 0 || rand() < exp(-Δ)
                n_exchanges += 1
                sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
                # scale velocities
                sys.replicas[n].velocities *= sqrt(β_n/β_m)
                sys.replicas[m].velocities *= sqrt(β_m/β_n)
            end
        end
    end
    # run for remaining_steps (if >0) for all replicas
    if remaining_steps > 0
        @sync for idx in eachindex(sim.simulators)
            if parallel
                Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps; parallel=parallel)
            else
                Molly.simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps; parallel=parallel)
            end
        end
    end
    @info "Number of exchanges: $n_exchanges"
    return sys
end
