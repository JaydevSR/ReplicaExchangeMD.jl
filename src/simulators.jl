export TemperatureREMD
"""
    TemperatureREMD(; <keyword arguments>)

A simulator for a parallel temperature replica exchange (TREX) simulation on a [`ReplicaSystem`](@ref). More information on this algorithm can be found in [Sugita Y., Okamoto Y. 1999](https://doi.org/10.1016/S0009-2614(99)01123-9).
The corresponding [`ReplicaSystem`](@ref) should have the same number of replicas as the number of temperatures in the simulator.

arguments:
- `dt::S`: the time step of the simulation.
- `temperatures::TP`: the temperatures corresponding to the replicas.
- `simulators::ST`: individual simulators for simulating each replica.
- `exchange_time::ET`: the time interval between replica exchange attempt.
"""
struct TemperatureREMD{N, T, S, DT, TP, ST, ET}
    dt::DT
    temperatures::TP
    simulators::ST
    exchange_time::ET
end

function TemperatureREMD(;
                dt,
                temperatures,
                simulators,
                exchange_time,
                kwargs...)
    S = eltype(simulators)
    T = eltype(temperatures)
    N = length(temperatures)
    DT = typeof(dt)
    TP = typeof(temperatures)
    ET = typeof(exchange_time)
    if length(simulators) != length(temperatures)
        throw(ArgumentError("Number of temperatures must match number of simulators"))
    end
    if exchange_time <= dt
        throw(ArgumentError("Exchange time must be greater than the time step"))
    end
    simulators = Tuple(simulators[i] for i in 1:N)
    ST = typeof(simulators)
    
    return TemperatureREMD{N, T, S, DT, TP, ST, ET}(dt, temperatures, simulators, exchange_time)
end

function simulate!(sys::ReplicaSystem{D, G, T},
                    sim::TemperatureREMD,
                    n_steps::Int;
                    assign_velocities::Bool=false,
                    rng=Random.GLOBAL_RNG,
                    n_threads::Int=Threads.nthreads()) where {D, G, T}
    if sys.n_replicas != length(sim.simulators)
        throw(ArgumentError("Number of replicas in ReplicaSystem and simulators in TemperatureREMD do not match."))
    end

    if n_threads > sys.n_replicas
        thread_div = equal_parts(n_threads, sys.n_replicas)
    else # pass 1 thread per replica
        thread_div = equal_parts(sys.n_replicas, sys.n_replicas)
    end

    # calculate n_cycles and n_steps_per_cycle from dt and exchange_time
    n_cycles = convert(Int64, (n_steps * sim.dt) ÷ sim.exchange_time)
    cycle_length = (n_cycles > 0) ? n_steps ÷ n_cycles : 0
    remaining_steps = (n_cycles > 0) ? n_steps % n_cycles : n_steps

    if assign_velocities
        for i in eachindex(sys.replicas)
            random_velocities!(sys.replicas[i], sim.temperatures[i]; rng=rng)
        end
    end

    for cycle=1:n_cycles
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length; n_threads=thread_div[idx])
        end

        if cycle != n_cycles
            cycle_parity = cycle % 2
            for n in 1+cycle_parity:2:sys.n_replicas-1
                m = n + 1
                if dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
                    k_b = sys.k * T(Unitful.Na)
                else
                    k_b = sys.k
                end
                T_n, T_m = sim.temperatures[n], sim.temperatures[m]
                β_n, β_m = 1/(k_b*T_n), 1/(k_b*T_m)
                V_n, V_m = potential_energy(sys.replicas[n]), potential_energy(sys.replicas[m])
                Δ = (β_m - β_n)*(V_n - V_m)
                if Δ <= 0 || rand(rng) < exp(-Δ)
                    # exchange coordinates and velocities
                    sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
                    sys.replicas[n].velocities, sys.replicas[m].velocities = sys.replicas[m].velocities, sys.replicas[n].velocities
                    # scale velocities
                    sys.replicas[n].velocities .*= sqrt(T_n/T_m)
                    sys.replicas[m].velocities .*= sqrt(T_m/T_n)
                    if !isnothing(sys.exchange_logger)
                        Molly.log_property!(sys.exchange_logger, sys, nothing, cycle*cycle_length; indices=(n, m), delta=Δ, n_threads=n_threads)
                    end
                end
            end
        end
    end

    # run for remaining_steps (if >0) for all replicas
    if remaining_steps > 0
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps; n_threads=thread_div[idx])
        end
    end

    if !isnothing(sys.exchange_logger)
        finish_logs!(sys.exchange_logger, n_steps)
    end

    return sys
end
