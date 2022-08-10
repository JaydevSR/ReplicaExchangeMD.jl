export TemperatureREMD

"""
    TemperatureREMD(; <keyword arguments>)

A simulator for a parallel temperature replica exchange (TREX) simulation on a
[`ReplicaSystem`](@ref).
See [Sugita and Okamoto 1999](https://doi.org/10.1016/S0009-2614(99)01123-9).
The corresponding [`ReplicaSystem`](@ref) should have the same number of replicas as
the number of temperatures in the simulator.
When calling [`simulate!`](@ref), the `assign_velocities` keyword argument determines
whether to assign random velocities at the appropriate temperature for each replica.
Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::DT`: the time step of the simulation.
- `temperatures::TP`: the temperatures corresponding to the replicas.
- `simulators::ST`: individual simulators for simulating each replica.
- `exchange_time::ET`: the time interval between replica exchange attempts.
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
                         exchange_time)
    S = eltype(simulators)
    T = eltype(temperatures)
    N = length(temperatures)
    DT = typeof(dt)
    TP = typeof(temperatures)
    ET = typeof(exchange_time)

    if length(simulators) != length(temperatures)
        throw(ArgumentError("Number of temperatures ($(length(temperatures))) must match " *
                            "number of simulators ($(length(simulators)))"))
    end
    if exchange_time <= dt
        throw(ArgumentError("Exchange time ($exchange_time) must be greater than the time step ($dt)"))
    end

    simulators = Tuple(simulators[i] for i in 1:N)
    ST = typeof(simulators)
    
    return TemperatureREMD{N, T, S, DT, TP, ST, ET}(dt, temperatures, simulators, exchange_time)
end

function simulate!(sys::ReplicaSystem{D, G, T},
                    sim::TemperatureREMD,
                    n_steps::Integer;
                    assign_velocities::Bool=false,
                    rng=Random.GLOBAL_RNG,
                    n_threads::Integer=Threads.nthreads()) where {D, G, T}
    if sys.n_replicas != length(sim.simulators)
        throw(ArgumentError("Number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
                "and simulators in TemperatureREMD ($(length(sim.simulators))) do not match."))
    end

    if assign_velocities
        for i in eachindex(sys.replicas)
            random_velocities!(sys.replicas[i], sim.temperatures[i]; rng=rng)
        end
    end

    simulate_remd!(sys, sim, n_steps, tremd_exchange!; rng=rng, n_threads=n_threads)
        end

function tremd_exchange!(sys::ReplicaSystem{D,G,T},
                        sim::TemperatureREMD,
                        n::Integer,
                        m::Integer;
                        n_threads::Int=Threads.nthreads(),
                        rng=Random.GLOBAL_RNG) where {D,G,T}
    if dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k_b = sys.k * T(Unitful.Na)
    else
        k_b = sys.k
    end

    T_n, T_m = sim.temperatures[n], sim.temperatures[m]
    β_n, β_m = inv(k_b * T_n), inv(k_b * T_m)
    neighbors_n = find_neighbors(sys.replicas[n], sys.replicas[n].neighbor_finder;
                                    n_threads=n_threads)
    neighbors_m = find_neighbors(sys.replicas[m], sys.replicas[m].neighbor_finder;
                                    n_threads=n_threads)
    V_n = potential_energy(sys.replicas[n], neighbors_n)
    V_m = potential_energy(sys.replicas[m], neighbors_m)
    Δ = (β_m - β_n) * (V_n - V_m)
    should_exchange = Δ <= 0 || rand(rng) < exp(-Δ)

    if should_exchange
        # exchange coordinates and velocities
        sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
        sys.replicas[n].velocities, sys.replicas[m].velocities = sys.replicas[m].velocities, sys.replicas[n].velocities
        # scale velocities
        sys.replicas[n].velocities .*= sqrt(T_n / T_m)
        sys.replicas[m].velocities .*= sqrt(T_m / T_n)
    end

    return Δ, should_exchange
end

function simulate_remd!(sys::ReplicaSystem{D,G,T},
                        remd_sim,
                        n_steps::Int,
                        make_exchange!::Function;
                        rng=Random.GLOBAL_RNG,
                        n_threads::Int=Threads.nthreads()) where {D,G,T}
    if sys.n_replicas != length(remd_sim.simulators)
        throw(ArgumentError("Number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
                "and simulators in TemperatureREMD ($(length(remd_sim.simulators))) do not match."))
    end

    if n_threads > sys.n_replicas
        thread_div = equal_parts(n_threads, sys.n_replicas)
    else # pass 1 thread per replica
        thread_div = equal_parts(sys.n_replicas, sys.n_replicas)
    end

    # calculate n_cycles and n_steps_per_cycle from dt and exchange_time
    n_cycles = convert(Int, (n_steps * remd_sim.dt) ÷ remd_sim.exchange_time)
    cycle_length = (n_cycles > 0) ? n_steps ÷ n_cycles : 0
    remaining_steps = (n_cycles > 0) ? n_steps % n_cycles : n_steps
    n_attempts = 0

    for cycle = 1:n_cycles
        @sync for idx in eachindex(remd_sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], remd_sim.simulators[idx], remaining_steps;
                                     n_threads=thread_div[idx])
                end

        # Alternate checking even pairs 2-3/4-5/6-7/... and odd pairs 1-2/3-4/5-6/...
        cycle_parity = cycle % 2
        for n in (1 + cycle_parity):2:(sys.n_replicas - 1)
            n_attempts += 1
            m = n + 1
            Δ, exchanged = make_exchange!(sys, remd_sim, n, m; rng=rng, n_threads=n_threads)
            if exchanged && !isnothing(sys.exchange_logger)
                log_property!(sys.exchange_logger, sys, nothing, cycle * cycle_length;
                                    indices=(n, m), delta=Δ, n_threads=n_threads)
            end
        end
    end

    if remaining_steps > 0
        @sync for idx in eachindex(remd_sim.simulators)
            Threads.@spawn Molly.simulate!(sys.replicas[idx], remd_sim.simulators[idx], remaining_steps;
                                     n_threads=thread_div[idx])
        end
    end

    if !isnothing(sys.exchange_logger)
        finish_logs!(sys.exchange_logger; n_steps=n_steps, n_attempts=n_attempts)
    end

    return sys
end

# Calculate k almost equal patitions of n
@inline function equal_parts(n::Int, k::Int)
    ndiv = n ÷ k
    nrem = n % k
    n_parts = ntuple(i -> (i <= nrem) ? ndiv+1 : ndiv, k)
    return n_parts
end
