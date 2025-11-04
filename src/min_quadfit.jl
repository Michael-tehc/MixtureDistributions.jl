"Fit quadratic `y = ax^2 + bx + c` through 3 points (x, y), return coefficients `(a, b)`."
function _fit_quadratic(x1, y1, x2, y2, x3, y3)
    @assert x1 <= x2 <= x3
    denom = (x1-x2) * (x1-x3) * (x2-x3) # < 0
    a = (
        x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)
    ) / denom
    b = (
        x3^2 * (y1-y2) + x1^2 * (y2-y3) + x2^2 * (y3-y1)
    ) / denom
    (a, b)
end

"""
    min_quadfit(fn, lo::Real, hi::Real)

Minimize `fn(x)` over `x ∈ [a, b]` using quadratic fit search.
"""
function min_quadfit(fn, lo::Real, hi::Real)
    @assert lo < hi

    x1, x2, x3 = lo, (lo + hi)/2, hi # invariant: x1 < x2 < x3
    xopt = x2
    y1, y2, y3 = fn(x1), fn(x2), fn(x3)
    a, b = _fit_quadratic(x1, y1, x2, y2, x3, y3)
    if a < 0
        @warn "Quadratic points up instead of down: a=$a."
        return xopt
    end
    (!isfinite(a) || !isfinite(b)) && return xopt
    niter = 0
    while x3 - x1 > 1e-6
        niter += 1
        xopt = -b / (2a)

        if !(lo <= xopt <= hi)
            @info "$xopt not in [$lo, $hi]: $x1 < $x2 < $x3"
            @show [x1, y1, x2, y2, x3, y3]
            @assert false
        end
        yopt = fn(xopt)

        if xopt < x2
            if yopt > y2
                x1, y1 = xopt, yopt
            else
                x2, x3 = xopt, x2
                y2, y3 = yopt, y2
            end
        else # xopt > x2
            if yopt > y2
                x3, y3 = xopt, yopt
            else
                x1, x2 = x2, xopt
                y1, y2 = y2, yopt
            end
        end

        a, b = _fit_quadratic(x1, y1, x2, y2, x3, y3)
        if !isfinite(a) || !isfinite(b)
            (abs(x2 - x1) < 1e-6 || abs(x3 - x2) < 1e-6) && break
            @warn "Failed to build quadratic" (; niter, x1, y1, x2, y2, x3, y3)
            break
        end
    end
    xopt
end