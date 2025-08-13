module ArtificialLandscapes

	# ---- Imports ----
	import FastPow.@fastpow
	using SparseArrays
	using LinearAlgebra

	# ---- Includes ----
	include("utilities.jl")
	include("problem_dictionary.jl")

	# ---- Exports ----
	export landscapes
end
