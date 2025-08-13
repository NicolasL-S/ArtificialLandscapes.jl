# For testing
# Extracting the lmax largest components of the vector v in magnitude, padding with zeros if v is 
# shorter than lmax. It is used as quick check that CUTEst functions are accurate.

function largest_components(v; lmax = 20)
	l = min(lmax, length(v))
	out = zeros(lmax)
	out[1:l] = v[sort(sortperm(abs.(v))[end - l + 1:end])]
	return out
end

# To compute statistics to test the validity of each problem. See gen_test_stats.jl for details.
function compute_problem_stats(gen_problem)
    x0, obj, grad! = gen_problem()
    d = length(x0)
    grad = similar(x0)
    grad!(grad, x0)
    grad1 = similar(x0)
    grad!(grad1, x0 .+ range(0,1, length = d))
    return [obj(x0); obj(x0 .+ range(0,1, length = d)); largest_components(x0); 
        largest_components(grad); largest_components(grad1)]
end