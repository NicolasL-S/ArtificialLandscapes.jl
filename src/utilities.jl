# To create a range on the elements of x without some elements at the start or the end
function dindices(x, dbegin, dend)
	dbegin >= 0 || throw(BoundsError("dbegin should be at least zero."))
	dend <= 0 || throw(BoundsError("dend should be at most zero."))
	return firstindex(x) + dbegin:lastindex(x) + dend
end

# Note: We could have automatically made n = min(n, length(x)). But this seems more prudent to avoid user mistakes.
function firstindices(x, n)
    length(x) < n && throw(BoundsError("x has fewer than $n elements"))
    return firstindex(x):firstindex(x) - 1 + n
end

# Note: sum_serial does not check bounds so it should only be called by sumx.
@inline function sum_serial(x, range)
    s = zero(eltype(x))
    @inbounds @simd for i in range
        s += x[i]
    end
    return s
end

@inline function sum_serial(f, x, range)
    s = zero(eltype(x))
    @inbounds @simd for i in range
        s += f(x[i])
    end
    return s
end

function sumx(x, range; d = 1024)
    first(range) >= firstindex(x) && last(range) <= lastindex(x) || throw(BoundsError("range ($(range)) is out of the bounds of x ($(eachindex(x)))."))
    l = length(range)
    if l ≤ d
        return sum_serial(x, range)
    else
        mid = first(range) - 1 + ((l - 1) ÷ (2d) + 1) * d # This division favours chunks of size d, which helps the processor (with d = 1024)
        return sumx(x, first(range): mid) + sumx(x, mid+1:last(range))
    end
end

sumx(x; d = 1024) = sumx(x, eachindex(x); d)

function sumi(f, s :: T, range; simd = true) :: T where T
	if simd
		@simd for i in range
			s += f(i)
		end
	else
		for i in range
			s += f(i)
		end
	end
    return s
end

# Contrary to sumx, a temporary array stores the partial sums because recursions with functions (f)
# slow things down. Then we do the cascading sum with the partial sums stored in cache.

function sumi(f, cache :: AbstractVector, range; d = 1024, simd = true)
    m = length(range) ÷ d
    sta, en = extrema(range)
    for g in 1:m
        gd = g * d
        cache[g] = sumi(i -> f(i), zero(eltype(cache)), sta - d + gd:sta - 1 + gd; simd)
    end
    cache[m + 1] = sumi(i -> f(i), zero(eltype(cache)), sta + d * m:en; simd)
    return sumx(cache, 1:m+1; d=2)
end

quadratic_form(mat, vec, cache) = dot(vec, mul!(cache, mat, vec))

@inline check_gradient_indices(gradient, x) = eachindex(gradient) == eachindex(x) || 
    throw(BoundsError("gradient and x should have the same indices."))

function check_x_indices(x, cache :: AbstractArray; regenerate = true)
    if eachindex(x) != eachindex(cache)
        message = "x should have indices $(eachindex(cache))."
        regenerate && (message *= "Regenerate the problem to change the size of x.")
        throw(BoundsError(message))
    end
end

function check_x_indices(x, n :: Number; regenerate = true)
    if eachindex(x) != Base.OneTo(n)
        message = "x should have indices $(Base.OneTo(n))."
        regenerate && (message *= "Regenerate the problem to change the size of x.")
        throw(BoundsError(message))
    end
end