using Test

@testset "Types & Dispatch — Full Test Suite" begin
    include("test_types.jl")
    include("test_dispatch.jl")
    include("test_unitful_ops.jl")
    include("test_spectral_estimate.jl")
end
