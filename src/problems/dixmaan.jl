function dixmaan(x, α, β, γ, δ, K)
    inv_N = eltype(x)(1 / lastindex(x))
    M = length(x) ÷ 3
    s = eltype(x)(1)
    s = sumi(i -> α * x[i]^2 * (i * inv_N)^K[1], eltype(x)(1), eachindex(x))
    s = sumi(i -> β * x[i]^2 * (x[i + 1] + x[i + 1]^2)^2 * (i * inv_N)^K[2], s, dindices(x, 0, -1))
    s = sumi(i -> γ * x[i]^2 * x[i + M]^4 * (i * inv_N)^K[3], s, firstindices(x, 2M))
    s = sumi(i -> δ * x[i] * x[i + 2M] * (i * inv_N)^K[4], s, firstindices(x, M))
    return s
end

function dixmaan_grad!(gradient, x, α, β, γ, δ, K)
    check_gradient_indices(gradient, x)

    inv_N = eltype(x)(1 / length(x))
    M = length(x) ÷ 3

    @inbounds for i in eachindex(x)
        gradient[i] = α * 2x[i] * (i * inv_N)^K[1]
    end

    @inbounds for i in dindices(x, 0, -1)
        a = x[i + 1] + x[i + 1]^2
        b = β * 2x[i] * (i * inv_N)^K[2] * a
        gradient[i]     += b * a
        gradient[i + 1] += b * (1 + 2x[i + 1]) * x[i]
    end

    @inbounds for i in firstindices(x, 2M)
        a = γ * 2x[i] * x[i + M]^3 * (i * inv_N)^K[3]
        gradient[i]     += a * x[i + M]
        gradient[i + M] += a * 2x[i]
    end

    @inbounds for i in firstindices(x, M)
        a = δ * (i * inv_N)^K[4]
        gradient[i]      += a * x[i + 2M]
        gradient[i + 2M] += a * x[i]
    end

    gradient
end

"""
This creates 
DIXMAANA1, DIXMAANB, DIXMAANC, DIXMAAND, DIXMAANE1, DIXMAANF, DIXMAANG, DIXMAANH, DIXMAANI1, 
DIXMAANJ, DIXMAANK, DIXMAANL, DIXMAANM1, DIXMAANN, DIXMAANO, DIXMAANP

Here is the top of the DIXMAANA1 SIF file:

NAME          DIXMAANA1

*   Problem :
*   *********
*   The Dixon-Maany test problem (version A) but removing elements/groups 
*   of type 2 since the parameter beta=0

*   Source:
*   L.C.W. Dixon and Z. Maany,
*   "A family of test problems with sparse Hessians for unconstrained
*   optimization",
*   TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

*   See also Buckley#221 (p. 49)
*   SIF input: Ph. Toint, Dec 1989.
*              correction by Ph. Shott, January 1995.
*              update Nick Gould, August 2022, to remove beta=0 terms.

*   classification OUR2-AN-V-0

*   M is equal to the third of the number of variables

*IE M                   5              -PARAMETER n = 15  original value 
*IE M                   30             -PARAMETER n = 90
*IE M                   100            -PARAMETER n = 300
*IE M                   500            -PARAMETER n = 1500
 IE M                   1000           -PARAMETER n = 3000
*IE M                   3000           -PARAMETER n = 9000
...

Keyword arguments:
α, β, γ, δ, K; n = 3000, T = typeof(1.)
"""
function gen_dixmaan(α, β, γ, δ, K; n = 3000, T = typeof(1.))
    return (x0 = 2ones(T, n),
            obj = x -> dixmaan(x, α, β, γ, δ, K),
            grad! = (gradient, x) -> dixmaan_grad!(gradient, x, α, β, γ, δ, K))
end