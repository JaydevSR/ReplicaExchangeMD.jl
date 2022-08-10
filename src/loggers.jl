"""
    ReplicaExchangeLogger(n_steps)
    ReplicaExchangeLogger(T, n_steps)

A logger that records exchanges in a replica exchange simulation.
The logged quantities include the number of exchange attempts (`n_attempts`),
number of successful exchanges (`n_exchanges`), exchanged replica indices (`indices`),
exchange steps (`steps`) and the value of Δ i.e. the argument of metropolis rate for
the exchanges (`deltas`).
"""
mutable struct ReplicaExchangeLogger{T}
    n_replicas::Int
    n_attempts::Int
    n_exchanges::Int
    indices::Vector{Tuple{Int, Int}}
    steps::Vector{Int}
    deltas::Vector{T}
    end_step::Int
end

function ReplicaExchangeLogger(T, n_replicas::Integer)
    return ReplicaExchangeLogger{T}(n_replicas, 0, 0, Tuple{Int, Int}[], Int[], T[], 0)
end

ReplicaExchangeLogger(n_replicas::Integer) = ReplicaExchangeLogger(DefaultFloat, n_replicas)

function log_property!(rexl::ReplicaExchangeLogger,
                       sys::ReplicaSystem,
                       neighbors=nothing,
                       step_n::Integer=0;
                       indices,
                       delta,
                       n_threads::Integer=Threads.nthreads())
    push!(rexl.indices, indices)
    push!(rexl.steps, step_n + rexl.end_step)
    push!(rexl.deltas, delta)
    rexl.n_exchanges += 1
end

function finish_logs!(rexl::ReplicaExchangeLogger; n_steps::Integer=0, n_attempts::Integer=0)
    rexl.end_step += n_steps
    rexl.n_attempts += n_attempts
end
