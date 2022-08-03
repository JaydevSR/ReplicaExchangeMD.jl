mutable struct ReplicaExchangeLogger{T}
    n_replicas::Int
    n_attempts::Int
    n_exchanges::Int
    indices::Vector{Tuple{Int,Int}}
    steps::Vector{Int}
    delta::Vector{T}
    end_step::Int
end

ReplicaExchangeLogger{T}(n_replicas::Int) where T = ReplicaExchangeLogger{T}(n_replicas, 0, 0, Tuple{Int, Int}[], Int[], T[], 0)

function Molly.log_property!(
                    rexl::ReplicaExchangeLogger,
                    sys::ReplicaSystem,
                    neighbors=nothing,
                    step_n::Int=0;
                    indices::Tuple{Int,Int},
                    delta::Float64,
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