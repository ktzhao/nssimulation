include("GridModule.jl") 
include("DomainDecomposition.jl")
include("Communication.jl")
include("AdvancedParallel.jl")
using .GridModule
using .DomainDecomposition
using .Communication
using .AdvancedParallel
# Define domain limits and grid spacing.
