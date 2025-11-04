# """
#     min_golden(f, a::Real, b::Real, tol::Real=1e-6)

# Golden section search. Minimize `f(x)` where `x ∈ (a, b)`.

# Never evaluates `f` at endpoints.
# """
# function min_golden(f, a::Real, b::Real, tol::Real=1e-6; maxiters::Integer=100)
#     @assert a < b
#     ϕ = (3 - sqrt(5)) / 2 # actually (1-ϕ)???
    
#     c = a + ϕ * (b - a)
#     d = b - ϕ * (b - a)
#     fa = fb = Inf
#     fc, fd = f(c), f(d)
#     for _ in 1:maxiters
#         if fc < fd
#             # Keep left interval: a < c < d
#             a, d, b = a, c, d
#             fa, fd, fb = fa, fc, fd
#             c = a + ϕ * (b - a)
#             fc = f(c)
#         else
#             # Keep right interval: c < d < b
#             a, c, b = c, d, b
#             fa, fc, fb = fc, fd, fb
#             d = b - ϕ * (b - a)
#             fd = f(d)
#         end

#         @show (fa, fb)
#         (isfinite(fc) && !isfinite(fd) && (b - a < tol)) && break
#     end

#     (fc < fd) ? c : d # minimizer
# end

"""
    min_golden(f, a::Real, b::Real, tol::Real=1e-6)

Golden section search. Minimize `f(x)` where `x ∈ (a, b)`.

Never evaluates `f` at endpoints.
"""
function min_golden(f, xl::Real, xu::Real, tol::Real=1e-6; maxiters::Integer=100)
    @assert xl < xu
    ϕ = (sqrt(5) - 1) / 2
    
    x1 = xu - ϕ * (xu - xl)
    x2 = xl + ϕ * (xu - xl)
    fl = fu = Inf
    f1, f2 = f(x1), f(x2)
    for _ in 1:maxiters
        if f1 < f2
            # Keep left interval
            xu, x2 = x2, x1
            fu, f2 = f2, f1

            x1 = xu - ϕ * (xu - xl)
            f1 = f(x1)
        else
            # Keep right interval
            xl, x1 = x1, x2
            fl, f1 = f1, f2

            x2 = xl + ϕ * (xu - xl)
            f2 = f(x2)
        end

        (isfinite(fl) && isfinite(fu) && (xu - xl < tol)) && break
    end

    (xl + xu)/2 # minimizer
end