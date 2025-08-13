# CUTEst requires gfortran, does not work on some systems like windows, and can take very long 
# to compute. Also, autodiff does not work for many problems because of the way the objectives are 
# computed. So it can't be used for testing either.
# Therefore, we test the accuracy of the problems by precomputing statistics from the original 
# packages (when possible). Since starting points and gradients can be 5000 dimentional or more, it
# wouldn't be practical to store all dimensions. Given that the largest discrepancies are likely to 
# come from the largest components in magnitude, these are the ones we should check. Thus, for each 
# problem of dimension d and a number lmax of elements to check, we store a vector containing 
# (stacked one after another):
# 1- The objective at the starting point;
# 2- The objective at the starting point + range(0,1, length = d);
# 3- The lmax largest value of the starting point (in order in which they appear);
# 4- The lmax largest value of the gradient at starting point;
# 5- The lmax largest value of the gradient at starting point + range(0,1, length = d);
# lmax is 20 (defined in utilities.jl). If the problem has fewer than dimensions than lmax, trailing 
# zeros are added for padding.
# These statistics are pre-computed and saved in a .csv file to be opened by runtests.

using CSV, CUTEst, OptimTestProblems, OptimTestProblems.MultivariateProblems
include("utilities.jl")

# CUTEst
global CUTEst_problem = CUTEstModel("ARGLINA") # Initializing CUTEst
function gen_problem_from_cutest(name)
	finalize(CUTEst_problem)
    global CUTEst_problem = CUTEstModel(name)
	return (x0 = CUTEst_problem.meta.x0, 
		obj = x -> CUTEst.obj(CUTEst_problem, x), 
		grad! = (gradient, x) -> CUTEst.grad!(CUTEst_problem, x, gradient))
end

CUTEst_problem_names = CUTEst.select(contype = "unc") # Only unconstrained problems for now

# OptimTestProblems
muvp_problems = MultivariateProblems.UnconstrainedProblems.examples # Only unconstrained problems for now
muvp_problem_names = [name for (name, p) in muvp_problems]

function gen_problem_from_OptimTestProblems(name)
	p = muvp_problems[name]
	return (x0 = p.initial_x, obj = objective(p), grad! = gradient(p))
end

function compute_stats!(gen_problem, problem_stats, names)
	for name in names
		println(name)
		try
			problem_stats[name] = compute_problem_stats(() -> gen_problem(name))
			println(problem_stats[name])
		catch
			println("Failed!")
		end
	end
end

problem_stats = Dict{AbstractString, Vector{Float64}}()
compute_stats!(gen_problem_from_OptimTestProblems, problem_stats, muvp_problem_names)
compute_stats!(gen_problem_from_cutest, problem_stats, CUTEst_problem_names)
CSV.write("test_stats.csv", problem_stats)