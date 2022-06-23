export TemperatureREMD
struct TemperatureREMD{N, T, S, SD}
    temps::StaticVector{N, T}
    indices::StaticVector{N, Int}
    simulators::SD
    n_cycles::Int
    cycle_len::Int
end

function TemperatureREMD(;
    simulator,
    temps, n_cycles, cycle_len, dt, 
    kwargs...)
    S = simulator
    T = eltype(temps)
    N = length(temps)
    temps = SA[temps...]
    indices = SA[collect(1:length(temps))...]

    coupling_required = false
    if simulator ∈ [VelocityVerlet, Verlet, StormerVerlet]
        coupling_required = true
        coupling = kwargs[:coupling]
    end

    if coupling_required && coupling ∈ [AndersenThermostat, BerendsenThermostat]
        couplings = [coupling(temps[i], kwargs[:coupling_const]) for i in indices]
    elseif coupling_required && coupling ∈ [RescaleThermostat]
        couplings = [coupling(temps[i]) for i in indices]
    else
        error("Temperature coupling required for TemperatureREMD using $(simulator) simulator.")
    end

    if simulator==Langevin
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], friction=kwargs[:friction],
            remove_CM_motion=kwargs[:remove_CM_motion])]
            for i in indices
        )
    elseif simulator ∈ [VelocityVerlet, Verlet, StormerVerlet]
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], coupling=couplings[i],
            remove_CM_motion=kwargs[:remove_CM_motion])]
            for i in indices
        )
    elseif simulator == StormerVerlet
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], coupling=couplings[i])]
            for i in indices
        )
    else
        error("Simulator type $(simulator) not supported.")
    end
    SD = typeof(simulators)

    return TemperatureREMD{N, T, S, SD}(temps, indices, simulators, n_cycles, cycle_len)
end