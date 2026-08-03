using Test

@testset "JLab Module — Full Test Suite" begin
    include("test_wavelets.jl")
    include("test_multivariate.jl")
    include("test_spectral.jl")
    include("test_timeseries.jl")
    include("test_ellipse.jl")
    include("test_validation.jl")
    include("test_jdata.jl")
end
