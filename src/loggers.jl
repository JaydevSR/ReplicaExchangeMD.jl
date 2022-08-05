"""
    ReplicaExchangeLogger(n_steps::Int, T::DataType=DefaultFloat)

A logger that records exchanges in a replica exchange simulation. The logged quantities include: number of exchange attempts (`n_attempts`),
number of successful exchanges (`n_exchanges`), exchanged replica indices (`indices`), exchange steps (`steps`) and
the value of Δ i.e. the argument of metropolis rate for the exchanges (`delta`).
"""
mutable struct ReplicaExchangeLogger{T}
    n_replicas::Int
    n_attempts::Int
    n_exchanges::Int
    indices::Vector{Tuple{Int,Int}}
    steps::Vector{Int}
    delta::Vector{T}
    end_step::Int
end

ReplicaExchangeLogger(n_replicas::Int, T::DataType=DefaultFloat) = ReplicaExchangeLogger{T}(n_replicas, 0, 0, Tuple{Int, Int}[], Int[], T[], 0)

function Molly.log_property!(
                    rexl::ReplicaExchangeLogger,
                    sys::ReplicaSystem,
                    neighbors=nothing,
                    step_n::Int=0;
                    indices::Tuple{Int,Int},
                    delta::Real,
                    n_threads=Threads.nthreads())
    push!(rexl.indices, indices)
    push!(rexl.steps, step_n+rexl.end_step)
    push!(rexl.delta, delta)
    rexl.n_exchanges += 1
end

function finish_logs!(rexl::ReplicaExchangeLogger; n_steps::Int=0, n_attempts::Int=0)
    rexl.end_step += n_steps
    rexl.n_attempts += n_attempts
end
