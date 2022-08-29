# ReplicaExchangeMD.jl

> This repository contains work done by Jaydev Singh Rao towards the completion of Google Summer of Code 2022 ([Project Link](https://summerofcode.withgoogle.com/programs/2022/projects/BK7TI3rX)).
>
> All the work done and the process has be logged in [this page](https://jaydevsr.notion.site/GSoC-Contribution-Journal-dc12886cc9644f90b3be1dbd6c748710) as a journal. 

Implementation of parallel Replica Exchange Molecular Dynamics simulation based on [Molly.jl](https://github.com/JuliaMolSim/Molly.jl).

The simulators that are implemented include:
1. `TemperatureREMD` : for parallel temperature replica exchange. 
2. `HamiltonianREMD` : for parallel hamiltonian replica exchange.

These simulations are performed on a `ReplicaSystem` type which is a subtype of  `AbstractSystem` type from `AtomsBase.jl`.
`ReplicaSystem` acts as a container for different replicas of the `System` for a replica-exchange simulation.

The interface provided by in this codebase also allows one to define their own custom REMD simulators. Refer to Molly.jl [docs](https://juliamolsim.github.io/Molly.jl/stable) for more information.

> NOTE: All of the code from this repositary has been merged into Molly.jl. So anyone can directly use these simulators and provided interface from the releases of the main library.

## Documentation

For documentation, please refer to the Molly.jl [docs](https://juliamolsim.github.io/Molly.jl/stable).
