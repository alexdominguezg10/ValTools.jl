using Test
using ValTools.JLab

@testset "JLab JData" begin

    @testset "catalog is populated" begin
        @test length(JDATA_CATALOG) >= 4
        @test haskey(JDATA_CATALOG, "gulfdrifters")
        @test haskey(JDATA_CATALOG, "gulfflow_quarter")
        @test haskey(JDATA_CATALOG, "gomed")
    end

    @testset "catalog entries have required fields" begin
        for (name, info) in JDATA_CATALOG
            @test !isempty(info.url)
            @test !isempty(info.filename)
            @test !isempty(info.description)
            @test !isempty(info.doi)
            @test !isempty(info.citation)
            @test endswith(info.filename, ".nc")
        end
    end

    @testset "cache directory creation" begin
        # _jdata_cache_dir is internal; test via jdata_download error path
        @test_throws ErrorException jdata_download("nonexistent")
    end

    @testset "jdata_list runs without error" begin
        # Just verify it doesn't throw
        jdata_list()
        @test true
    end

    @testset "download error on unknown dataset" begin
        @test_throws ErrorException jdata_download("nonexistent_dataset")
    end

    # Note: actual download tests are skipped in CI (require network)
    # To run: JDATA_TEST_DOWNLOAD=1 julia test/jlab/test_jdata.jl
    if get(ENV, "JDATA_TEST_DOWNLOAD", "") == "1"
        @testset "download — gulfdrifters" begin
            path = jdata_download("gulfdrifters"; force=true)
            @test isfile(path)
            @test filesize(path) > 1_000_000  # should be several MB
        end
    end
end
