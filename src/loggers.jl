mutable struct ReplicaExchangeLogger
    n_replicas::Int
    n_exchanges::Int
    indices::Vector{Tuple{Int,Int}}
    steps::Vector{Int}
    delta::Vector{Float64}
    end_step::Int
end

ReplicaExchangeLogger(n_replicas::Int) = ReplicaExchangeLogger(n_replicas, 0, Tuple{Int, Int}[], Int[], Float64[], 0)

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

function finish_logs!(rexl::ReplicaExchangeLogger)
    if !isempty(rexl.indices)
        rexl.end_step = last(rexl.steps)
    end
end