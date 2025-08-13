function sinquad2(x, cache)
	check_x_indices(x, cache)
	return ((x[begin] - 1)^2)^2 + (x[end]^2 - x[begin]^2)^2 +
    	sumi(i -> (sin(x[i] - x[end]) - x[begin]^2 + x[i]^2)^2, cache, dindices(x,1,-1))
end

function sinquad2_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	gbegin = gend = zero(eltype(x))
	xb, xe = x[begin], x[end]
	xb_sq = xb^2
	@inbounds for i in dindices(x, 1, -1)
		a = 2(sin(x[i] - xe) - xb_sq + x[i]^2)
		c = cos(x[i] - xe)
		gbegin += -2xb * a
		gend += -c * a
		gradient[i] = (2x[i] + c) * a
	end
	a = 2(xe^2 - xb_sq)
	gradient[begin] = 4(xb - 1)^3 - 2xb * a + gbegin
	gradient[end] = 2xe * a + gend
	return gradient
end

"""
NAME          SINQUAD2

*   Problem :
*   *********

*   Another function with nontrivial groups and
*   repetitious elements.

*   Source:
*   N. Gould, private communication.

*   SIF input: N. Gould, Dec 1989.
*   modifield version of SINQUAD (formulation corrected) May 2024

*   classification OUR2-AY-V-0

*   number of variables

*IE N                   5              -PARAMETER     original value
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
function gen_sinquad2(; n = 5000, T = typeof(1.))
    return (x0 = 0.1ones(T, n), 
            obj = x -> sinquad2(x, Vector{T}(undef, n)), 
            grad! = sinquad2_grad!)
end