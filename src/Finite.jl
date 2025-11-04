export FiniteMixture, FiniteMixtureSkew, FiniteMixtureFull

struct FiniteMixture{T<:Real} <: CompoundMixture
    ps::Vector{T}
    mu::T
    vs::Vector{T}

    function FiniteMixture(ps::AbstractVector{T}, mu::T, vs::AbstractVector{T}) where T<:Real
        @assert length(ps) == length(vs)
        @assert all(>=(0), ps)
        @assert sum(ps) ≈ 1
        @assert all(>(0), vs)

        new{T}(ps, mu, vs)
    end
end

Distributions.params(d::FiniteMixture) = (d.ps, d.mu, d.vs)

function Random.rand(rng::Random.AbstractRNG, d::FiniteMixture)
	ps, mu, vs = params(d)
	k = rand(rng, Categorical(ps))
	mu + sqrt(vs[k]) * randn(rng)
end

Distributions.mean(d::FiniteMixture) = d.mu
Distributions.var(d::FiniteMixture) = dot(d.ps, d.vs)

Distributions.logpdf(d::FiniteMixture, x::Real) = log(sum(
    p * pdf(Normal(d.mu, sqrt(v)), x)
    for (p, v) in zip(d.ps, d.vs)
))

"""
    FiniteMixtureSkew(ps::AbstractVector{T}, mu::T, p::T, vs::AbstractVector{T}) where T<:Real

Skewed finite mixture of the form:

    X = μ + p V + sqrt(V)ε
    ε ~ N(0, 1)
    V ~ Categorical([p1, p2, ..., pK], [v1, v2, ..., vK])

Parameter `p` captures potential skewness.
"""
struct FiniteMixtureSkew{T<:Real} <: CompoundMixture
    ps::Vector{T}
    mu::T # location
    p::T # skew
    vs::Vector{T}

    function FiniteMixtureSkew(ps::AbstractVector{T}, mu::T, p::T, vs::AbstractVector{T}) where T<:Real
        @assert length(ps) == length(vs)
        @assert all(>=(0), ps)
        @assert sum(ps) ≈ 1
        @assert all(>(0), vs)

        new{T}(ps, mu, p, vs)
    end
end

"""
    FiniteMixtureSkew(ps::AbstractVector{T}, mu::T, vs::AbstractVector{T}) where T<:Real

Construct `FiniteMixtureSkew(ps, mu, 0, vs)`.
"""
function FiniteMixtureSkew(ps::AbstractVector{T}, mu::T, vs::AbstractVector{T}) where T<:Real
    FiniteMixtureSkew(ps, mu, zero(T), vs)
end

"""
    FiniteMixtureSkew(ps::AbstractVector{T}, vs::AbstractVector{T}) where T<:Real

Construct `FiniteMixtureSkew(ps, 0, 0, vs)`.
"""
function FiniteMixtureSkew(ps::AbstractVector{T}, vs::AbstractVector{T}) where T<:Real
    FiniteMixtureSkew(ps, zero(T), zero(T), vs)
end

Distributions.params(d::FiniteMixtureSkew) = (d.ps, d.mu, d.p, d.vs)

function Random.rand(rng::Random.AbstractRNG, d::FiniteMixtureSkew)
	ps, mu, p, vs = params(d)
	k = rand(rng, Categorical(ps))
    vk = vs[k]
	mu + p * vk + sqrt(vk) * randn(rng)
end

Distributions.mean(d::FiniteMixtureSkew) = d.mu + d.p * dot(d.ps, d.vs)
Distributions.var(d::FiniteMixtureSkew) = sum(components(d, :var))
Distributions.skewness(d::FiniteMixtureSkew) = sum(components(d, :skewness))
Distributions.kurtosis(d::FiniteMixtureSkew) = sum(components(d, :kurtosis)) - 3

Distributions.components(d::FiniteMixtureSkew, what::Symbol) = begin
    ps, mu, p, vs = params(d)
    EV = dot(ps, vs)
    if what == :var
        VV = sum(pk * (vk - EV)^2 for (pk, vk) in zip(ps, vs))
        (p^2 * VV, EV)
    elseif what == :skewness
        VV  = sum(pk * (vk - EV)^2 for (pk, vk) in zip(ps, vs))
        EV3 = sum(pk * (vk - EV)^3 for (pk, vk) in zip(ps, vs))
        s = std(d)
        (p^3 * EV3 / s^3, 3p * VV / s^3)
    elseif what == :kurtosis
        t1 = p^4 * sum(pk * (vk - EV)^4 for (pk, vk) in zip(ps, vs))
        t2 = 6 * p^2 * sum(pk * (vk - EV)^2 * vk for (pk, vk) in zip(ps, vs))
        t3 = 3 * sum(pk * vk^2 for (pk, vk) in zip(ps, vs))
        s = std(d)
        (t1 / s^4, t2 / s^4, t3 / s^4)
    else
        throw(ArgumentError("No components for $what"))
    end
end

Distributions.logpdf(d::FiniteMixtureSkew, x::Real) = log(sum(
    pk * pdf(Normal(d.mu + d.p * vk, sqrt(vk)), x)
    for (pk, vk) in zip(d.ps, d.vs)
))

struct FiniteMixtureFull{K, T<:Real} <: CompoundMixture
    ps::MVector{K, T}
    ms::MVector{K, T}
    vs::MVector{K, T}

    function FiniteMixtureFull(ps::MVector{K, T}, ms::MVector{K, T}, vs::MVector{K, T}) where {K, T<:Real}
        @assert all(>=(0), ps)
        @assert sum(ps) ≈ 1
        @assert all(>(0), vs)
        new{K, T}(ps, ms, vs)
    end
end

function FiniteMixtureFull(ps::AbstractVector{T}, ms::AbstractVector{T}, vs::AbstractVector{T}) where T<:Real
    FiniteMixtureFull(MVector(ps...), MVector(ms...), MVector(vs...))
end

Distributions.logpdf(d::FiniteMixtureFull, x::Real) = log(sum(
    pk * pdf(Normal(mk, sqrt(vk)), x)
    for (pk, mk, vk) in zip(d.ps, d.ms, d.vs)
))

function Random.rand(rng::Random.AbstractRNG, d::FiniteMixtureFull)
	(; ps, ms, vs) = d
	k = rand(rng, Categorical(ps))
	ms[k] + sqrt(vs[k]) * randn(rng)
end

Distributions.mean(d::FiniteMixtureFull) = sum(components(d, :mean))
Distributions.var(d::FiniteMixtureFull) = sum(components(d, :var))
Distributions.skewness(d::FiniteMixtureFull) = sum(components(d, :skewness))
Distributions.kurtosis(d::FiniteMixtureFull) = sum(components(d, :kurtosis)) - 3

Distributions.components(d::FiniteMixtureFull, what::Symbol) = begin
    (; ps, ms, vs) =  d
    (what == :mean) && return (dot(ps, ms), )

    EM = dot(ps, ms)
    EV = dot(ps, vs)
    if what == :var
        VM = sum(pk * (mk - EM)^2 for (pk, mk) in zip(ps, ms))
        (VM, EV)
    elseif what == :skewness
        EM3 = sum(pk * (mk - EM)^3 for (pk, mk) in zip(ps, ms))
        t1 = 3 * sum(pk * (mk - EM) * vk for (pk, mk, vk) in zip(ps, ms, vs))
        s = std(d)
        (EM3 / s^3, t1 / s^3)
    elseif what == :kurtosis
        t1 = sum(pk * (mk - EM)^4 for (pk, mk) in zip(ps, ms))
        t2 = 6 * sum(pk * (mk - EM)^2 * vk for (pk, mk, vk) in zip(ps, ms, vs))
        t3 = 3 * sum(pk * vk^2 for (pk, vk) in zip(ps, vs))
        s = std(d)
        (t1 / s^4, t2 / s^4, t3 / s^4)
    else
        throw(ArgumentError("No components for $what"))
    end
end

# ========== Fitting ==========

function Distributions.fit_mle(
    d::FiniteMixture{T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiter::Integer=200, tol::Real=TOL
) where T<:Real
    ps, mu, vs = params(d)
    N = sum(ws)
    xmin, xmax = minimum(xs), maximum(xs)

    ps, vs = copy(ps), copy(vs)
    qs = similar(ps)
    S1, S2, S3 = similar(qs), similar(qs), similar(qs)

    loglik_prev = loglikelihood(d, xs, ws)
    #mu = mean(xs, ws) # method of moments
    for e in 1:maxiter
        # E step
        S1 .= 0; S2 .= 0; S3 .= 0 # sufficient statistics
        for (x, w) in zip(xs, ws)
            @. qs = ps * exp(-0.5 * (x - mu)^2 / vs) / sqrt(vs)
            qs ./= sum(qs)

            @. S1 += w * qs * x^2
            @. S2 += w * qs * x
            @. S3 += w * qs
        end

        # M step
        @. ps = S3 / N
        mu = min_golden(
            m -> sum(
                s3 * log(m^2 * s3 - 2m * s2 + s1)
                for (s1, s2, s3) in zip(S1, S2, S3)
            ), xmin - 0.5 * (xmax - xmin), xmax + 0.5 * (xmax - xmin)
        )
        @. vs = (mu^2 * S3 - 2mu * S2 + S1) / S3

        # Monitor convergence
        loglik_curr = loglikelihood(FiniteMixture(ps, mu, vs), xs, ws)
        dloglik = loglik_curr - loglik_prev
        #@show dloglik
        @assert (dloglik >= 0) || (abs(dloglik) < 1e-5)
        (e > 100) && (dloglik < tol) && break
        loglik_prev = loglik_curr
    end

    FiniteMixture(ps, mu, vs)
end

Distributions.fit_mle(
    ::Type{FiniteMixture}, K::Integer, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiter::Integer=200, tol::Real=TOL, rng::Random.AbstractRNG=Random.Xoshiro()
) = Distributions.fit_mle(
        FiniteMixture(fill(1/K, K), mean(xs, ws), rand(rng, K)), xs, ws; maxiter, tol
    )

"""
```julia
fit_mle(
    d::FiniteMixtureSkew{T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiters::Integer=200, tol::Real=TOL
) where T<:Real
```

Fit using the ECM (Expectation Conditional Maximization) algorithm.
"""
function Distributions.fit_mle(
    d::FiniteMixtureSkew{T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiters::Integer=200, tol::Real=TOL
) where T<:Real
    """Solve system of 2 linear eqns:
    a1 + b1 μ + c1 p = 0
    a2 + b2 μ + c2 p = 0
    """
    @inline solve_system(a1, b1, c1, a2, b2, c2) = begin
        Det = b1 * c2 - b2 * c1
        μ = (a2 * c1 - a1 * c2) / Det
        p = -(a1 + b1 * μ) / c1
        (μ, p)
    end
    ps, mu, p, vs = params(d)
    N = sum(ws)

    ps, vs = copy(ps), copy(vs)
    qs = similar(ps)
    S0, S1, S2 = similar(qs), similar(qs), similar(qs)

    loglik_prev = loglikelihood(d, xs, ws)
    for e in 1:maxiters
        # E step
        S0 .= S1 .= S2 .= 0 # sufficient statistics
        for (x, w) in zip(xs, ws)
            @. qs = ps * exp(-0.5 * (x - mu - p * vs)^2 / vs) / sqrt(vs)
            qs ./= sum(qs)

            @. S0 += w * qs * x^0
            @. S1 += w * qs * x^1
            @. S2 += w * qs * x^2
        end

        # Conditional maximization 1
        ps .= S0 ./ N

        a1 = sum(s1/v for (s1, v) in zip(S1, vs))
        b1 = sum(s0/v for (s0, v) in zip(S0, vs))
        c1 = sum(S0)
        a2 = sum(S1)
        b2 = c1
        c2 = dot(S0, vs)
        mu, p = solve_system(a1, -b1, -c1, a2, -b2, -c2)

        # Conditional maximization 2
        @. vs = (S2 - 2mu * S1) / S0 + mu^2
        @. vs = 2vs / (1 + sqrt(1 + 4 * p^2 * vs))

        # Monitor convergence
        loglik_curr = loglikelihood(FiniteMixtureSkew(ps, mu, p, vs), xs, ws)
        dloglik = loglik_curr - loglik_prev
        #@show dloglik
        @assert (dloglik >= 0) || (abs(dloglik) < 1e-5)
        (e > 100) && (dloglik < tol) && break
        loglik_prev = loglik_curr
    end

    idxs = sortperm(vs)
    FiniteMixtureSkew(ps[idxs], mu, p, vs[idxs])
end

Distributions.fit_mle(
    ::Type{FiniteMixtureSkew}, K::Integer, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiters::Integer=200, tol::Real=TOL, rng::Random.AbstractRNG=Random.Xoshiro()
) = begin
    ps = fill(1/K, K)
    p = 0.0
    mu = mean(xs, ws)
    vs = rand(rng, K) # vs MUST be distinct! Otherwise can't solve linear system!!
    vs .= vs .* K .* var(xs, ws) ./ sum(vs)

    # Now mean(d0) == mean(xs, ws), var(d0) == var(xs, ws)
    d0 = FiniteMixtureSkew(ps, mu, p, vs)
    fit_mle(d0, xs, ws; maxiters, tol)
end


function Distributions.fit_mle(
    d::FiniteMixtureFull{K, T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    regV::Union{Nothing, <:AbstractRegularization}=nothing,
    maxiter::Integer=200, tol::Real=TOL
) where {K, T<:Real}
    update_v(::Nothing, S0::Real, S1::Real, S2::Real, m::Real) =
        (S2 - 2S1 * m + S0 * m^2) / S0

    update_v(r::RegInverseGamma, S0::Real, S1::Real, S2::Real, m::Real) = begin
        a = r.a
        (S2 - 2S1 * m + S0 * m^2 + 2 * (a + 1)) / (S0 + 2 * (a + 1))
    end

    update_v(r::RegLogNormal, S0::Real, S1::Real, S2::Real, m::Real) = begin
        σ = r.σ
        a = S0 + 2
        b = 2 / σ^2
        c = abs(S2 - 2S1 * m + S0 * m^2)
        # Solve a v + b v ln(v) = c
        c/b / lambertw_exp(a/b + log(c) - log(b)) # div by W(c/b * exp(a/b)) = W(exp(a/b + ln(c) - ln(b)))
    end

    (; ps, ms, vs) = d
    ps, ms, vs = copy(ps), copy(ms), copy(vs)
    N = sum(ws)

    qs = similar(ps)
    S0 = similar(qs); S1 = similar(qs); S2 = similar(qs)
    loglik_prev = loglikelihood(d, xs, ws)
    if regV !== nothing
        loglik_prev += sum(Base.Fix1(logpdf, regV), vs)
    end
    for e in 1:maxiter
        # E-step
        S0 .= S1 .= S2 .= 0
        for (x, w) in zip(xs, ws)
            # Posteriors (safe softmax, doesn't become NaN when vs[k] = 0)
            @. qs = log(ps) - 0.5 * (x - ms)^2 / vs - 0.5 * log(vs)
            qs .-= maximum(qs)
            @. qs = exp(qs)
            qs ./= sum(qs)

            # Sufficient statistics
            @. S0 += w * qs * x^0
            @. S1 += w * qs * x^1
            @. S2 += w * qs * x^2
        end

        # M-step
        @. ps = S0 / N
        @. ms = S1 / (S0 + 1e-6)
        vs .= update_v.(Ref(regV), S0, S1, S2, ms)

        # Monitor convergence
        loglik_curr = loglikelihood(FiniteMixtureFull(ps, ms, vs), xs, ws)
        if regV !== nothing
            loglik_curr += sum(Base.Fix1(logpdf, regV), vs)
        end
        dloglik = abs(loglik_curr - loglik_prev)
        #@assert (dloglik >= 0) || (abs(dloglik) < 1e-5) "Loglik decreased: $loglik_curr < $loglik_prev at iteration $e"
        (e > 100) && (dloglik < tol) && break
        loglik_prev = loglik_curr
    end

    idxs = sortperm(ps)
    FiniteMixtureFull(ps[idxs], ms[idxs], vs[idxs])
end

function Distributions.fit_mle(
    ::Type{<:FiniteMixtureFull{K}}, xs::AbstractVector{T}, ws::AbstractWeights=uweights(length(xs));
    regV::Union{Nothing, <:AbstractRegularization}=nothing,
    normalized::Bool=false,
    maxiter::Integer=200, tol::Real=TOL, rng::Random.AbstractRNG=Random.default_rng()
) where {K, T<:Real}
    μ, σ = if normalized
        zero(T), one(T)
    else
        mean(xs), std(xs)
    end
    xs_norm = if normalized
        xs
    else
        @. (xs - μ) / σ
    end

    ps = fill(1/K, K)
    ms = rand(rng, xs_norm, K)
    vs = rand(rng, K)
    mix = Distributions.fit_mle(
        FiniteMixtureFull(ps, ms, vs), xs_norm, ws; regV, maxiter, tol
    )
    FiniteMixtureFull(
        mix.ps,
        μ .+ σ .* mix.ms,
        σ^2 .* mix.vs
    )
end