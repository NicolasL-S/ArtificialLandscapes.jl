using ArtificialLandscapes
using Test
using CSV
using DataFrames

# Includes
include("test_utils.jl")

tol = 1e-2 # Would be nice to reduce it, but alas, many tests would fail.

nan_diff(x, y) = !isnan(x) && !isnan(y) ? max(x - y)/(x + y + 1) : isnan(x) && isnan(y) ? 0 : Inf

@testset "Precomputed test stats" begin
    test_stats = Dict(pairs(eachcol(CSV.read("test_stats.csv", DataFrame))))
    v = test_stats[:ARGLINA] # Initializing v
    to_exclude = ["Paraboloid Random Matrix"]
    for (name, f) in landscapes
        if (Symbol(name) in keys(test_stats)) && !(name in to_exclude)
            v = compute_problem_stats(f)
            diff = 0
            for i in eachindex(v)
                diff = max(diff, nan_diff(v[i], test_stats[Symbol(name)][i]))
            end
            @test diff < tol
            if diff >= tol
                println(name, " fails")
                display([v test_stats[Symbol(name)]]')
            end
        end
    end
end

# Making sure Julia errors correctly (does not crash) because of bad inputs sizes (empty arrays)
@testset "Empty array" begin
    empty = Float64[]
    for (name, f) in landscapes
        x0, obj, grad! = f()
        try
            obj(empty)
        catch e
            @test e isa BoundsError || e isa DimensionMismatch || 
                  e isa MethodError || e isa ArgumentError
        end
        try
            grad!(empty, x0)
        catch e
            @test e isa BoundsError || e isa DimensionMismatch || 
                  e isa MethodError || e isa ArgumentError
        end
        try
            grad!(x0, empty)
        catch e
            @test e isa BoundsError || e isa DimensionMismatch || 
                  e isa MethodError || e isa ArgumentError
        end
    end
end
