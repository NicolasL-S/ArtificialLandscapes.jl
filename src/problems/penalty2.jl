function penalty2(x, y, expx)
	check_x_indices(x, expx)
	T = eltype(x)
	T0 = zero(T)
    n = lastindex(x)
    a = T(-exp(-0.1))
    expx .= exp.(T(0.1) * x)
	Ti = T.(eachindex(x))
	Tnp1 = T(lastindex(x) + 1)
	s1 = sumi(i -> @inbounds(expx[i] + expx[i - 1] - y[i])^2, T0, dindices(x, 1, 0))
	s2 = sumi(i -> @inbounds((expx[i] + a)^2), T0, dindices(x, 1, 0))
	s3 = sumi(i -> @inbounds((Tnp1 - Ti[i]) * x[i]^2), T0, eachindex(x))
    return T(1e-5) * (s1 + s2) + (s3 - 1)^2 + (x[begin]- T(0.2))^2
end

function penalty2_grad!(gradient, x, y, expx)
	check_x_indices(x, expx)
    check_gradient_indices(gradient, x)
	T = eltype(x)
    n = length(x)
    a = T(-exp(-0.1))
    expx .= exp.(T(0.1) .* x)
	Ti = T.(eachindex(x))
	Tnp1 = T(n + 1)
	T0 = c = cl = zero(T)
    d3 = 2sumi(i -> @inbounds((Tnp1 - Ti[i]) * x[i]^2), T0, eachindex(x)) - 2
	T2em5 = T(2e-5)
	T1em1 = T(0.1)
	@inbounds for i in dindices(x, 1, 0)
		b = T2em5 * (expx[i] + expx[i-1] - y[i])
		cl = c
		c = T1em1 * expx[i] * (b + T2em5 * (T1em1 * expx[i] + a)) + d3 * (Tnp1 - Ti[i]) * 2x[i]
		gradient[i - 1] = b * T1em1 * expx[i - 1] + cl
	end
    gradient[begin] += 2x[begin] - 4T1em1 + d3 * n * 2x[begin]
	gradient[end] = c
    return gradient
end

"""
NAME          PENALTY2

*   Problem :
*   --------

*   The second penalty function

*   This is a nonlinear least-squares problem with M=2*N groups.
*    Group 1 is linear.
*    Groups 2 to N use 2 nonlinear elements.
*    Groups N+1 to M-1 use 1 nonlinear element.
*    Group M uses N nonlinear elements.
*   The Hessian matrix is dense.

*   Source:  Problem 24 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#112 (p. 80)

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   Number of variables

*IE N                   4              -PARAMETER     original value
*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   200            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
...

Keyword argument:
; n = 200, T = typeof(1.)
"""
function gen_penalty2(; n = 200, T = typeof(1.))
    x0 = ones(n)/2
    y = T[exp(0.1i) + exp(0.1(i-1)) for i in 1:n]
    expx = similar(x0)
    return (x0 = x0,
            obj = x -> penalty2(x, y, expx),
            grad! = (gradient, x) -> penalty2_grad!(gradient, x, y, expx))
end