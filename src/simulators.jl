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
    remove_CM_motion=true, 
    kwargs...)
    S = simulator
    T = eltype(temps)
    N = length(temps)
    temps = SA[temps...]
    indices = SA[[i for i=1:length(temps)]...]

    if simulator==Langevin
        simulators = Dict(
            [i, simulator(dt=dt, temperature=temps[i], friction=kwargs[:friction], remove_CM_motion=remove_CM_motion)]
            for i in indices
            )
    elseif simulator==VelocityVerlet || simulator==Verlet
        if kwargs[:coupling] ∈ [AndersenThermostat, BerendsenThermostat]
            coupling = kwargs[:coupling]
            simulators = Dict(
                [i, simulator(dt=dt, coupling=coupling(temps[i], kwargs[:coupling_const]), remove_CM_motion=remove_CM_motion)] for i in indices
            )
        elseif  kwargs[:coupling] ∈ [RescaleThermostat]
            coupling = kwargs[:coupling]
            simulators = Dict(
                [i, simulator(dt=dt, coupling=coupling(temps[i]), remove_CM_motion=remove_CM_motion)] for i in indices
            )
        else
            error("Temperature coupling required for TemperatureREMD.")
        end
    elseif simulator==StormerVerlet
        if kwargs[:coupling] ∈ [AndersenThermostat, BerendsenThermostat]
            coupling = kwargs[:coupling]
            simulators = Dict(
                [i, simulator(dt=dt, coupling=coupling(temps[i], kwargs[:coupling_const]))] for i in indices
            )
        elseif  kwargs[:coupling] ∈ [RescaleThermostat]
            coupling = kwargs[:coupling]
            simulators = Dict(
                [i, simulator(dt=dt, coupling=coupling(temps[i]))] for i in indices
            )
        else
            error("Temperature coupling required for TemperatureREMD.")
        end
    else
        error("Simulator type $(simulator) not supported.")
    end
    SD = typeof(simulators)

    return TemperatureREMD{N, T, S, SD}(temps, indices, simulators, n_cycles, cycle_len)
end