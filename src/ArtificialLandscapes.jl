"""
$(read(joinpath((@__DIR__)[1:end-4], "README.md"), String))
"""
module ArtificialLandscapes

	# ---- Imports ----
	import FastPow.@fastpow
	using LinearAlgebra

	# ---- Includes ----
	include("utilities.jl")
	include("problem_dictionary.jl")

	# ---- Exports ----
	export landscapes
end
