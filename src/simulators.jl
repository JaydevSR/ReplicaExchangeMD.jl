export TemperatureREMD
struct TemperatureREMD{T, I, S, SD}
    temps::T
    indices::I
    simulators::SD
    n_cycles::Int
    cycle_len::Int
end

function TemperatureREMD(;
    simulator,
    coupling=NoCoupling,
    temps, n_cycles, cycle_len, dt,
    remove_CM_motion=true, 
    kwargs...)
    S = simulator
    C = coupling
    T = typeof(temps)

    if simulator ∉ [Langevin, Verlet, VelocityVerlet, StormerVerlet]
        error("Simulator type $(simulator) not supported.")
    end

    indices = collect(1:length(temps))
    I = typeof(indices)

    if simulator==Langevin
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], friction=kwargs[:friction], remove_CM_motion=remove_CM_motion)]
            for i in indices
            )
    else
        #TODO: use other simulators
        error("Use of simulator $(simulator) not implemented.")
    end
    SD = typeof(simulators)

    return TemperatureREMD{T, I, S, SD}(temps, indices, simulators, n_cycles, cycle_len)
end