function tridia(x, α, β, γ, δ, cache)
	check_x_indices(x, cache)
	return γ * (x[begin] * δ - 1)^2 + sumi(i -> i*(-β * x[i-1]+α * x[i])^2, cache, dindices(x,1,0))
end
	
function tridia_grad!(gradient, x, α, β, γ, δ)
	al = a = zero(eltype(x))
	for i in dindices(x, 1, 0)
		al = a
		a = i * 2(-β * x[i - 1] + α * x[i])
		gradient[i - 1] = -β * a + α * al
	end
	gradient[end] = α * a
	gradient[begin] += γ * 2(x[begin] * δ - 1) * δ
	return gradient
end

"""
NAME          TRIDIA

*   Problem :
*   *********

*   Shanno's TRIDIA quadratic tridiagonal problem

*   Source: problem 8 in
*   Ph.L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   See also Buckley#40 (p.96)

*   SIF input: Ph. Toint, Dec 1989.

*   classification QUR2-AN-V-0

*   This problem is decomposed in n linear groups, the last n-1 of which
*   are 2 x 2 and singular.

*   N is the number of variables

*IE N                   10             -PARAMETER
*IE N                   20             -PARAMETER
*IE N                   30             -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_tridia(; n = 5000, T = typeof(1.))
	α, β, γ, δ = T.((2, 1, 1, 1))
	return (x0 = ones(T, n),
			obj = x -> tridia(x, α, β, γ, δ, Vector{T}(undef, n)), 
			grad! = (gradient, x) -> tridia_grad!(gradient, x, α, β, γ, δ))
end