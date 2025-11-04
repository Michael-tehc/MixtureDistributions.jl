export NIG

struct NIG{T<:Real} <: CompoundMixture
    z::T
    p::T
    μ::T
    σ::T
    function NIG(z::T, p::T, μ::T=zero(T), σ::T=one(T); check_args::Bool=true) where T<:Real
        check_args && @check_args NIG (z, z > zero(T)) (σ, σ > zero(T))
        new{T}(z, p, μ, σ)
    end
end

NIG(z::Real, p::Real=0, μ::Real=0, σ::Real=1) = NIG(promote(z, p, μ, σ)...)

Distributions.params(d::NIG) = (d.z, d.p, d.μ, d.σ)

function Random.rand(rng::Random.AbstractRNG, d::NIG)
    z, p, μ, σ = params(d)
    rand(rng, GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, -1/2))
end

for f in [:mean, :var, :skewness, :kurtosis, :mode]
    @eval function Distributions.$f(d::NIG)
        z, p, μ, σ = params(d)
        $f(GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, -1/2))
    end
end

for f in [:logpdf, :cdf, :mgf, :cf]
    @eval function Distributions.$f(d::NIG, x::Number)
        z, p, μ, σ = params(d)
        $f(GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, -1/2), x)
    end
end

function Distributions.fit_mle(
    d::NIG{T}, xs::AbstractVector{T}, ws::AbstractWeights=uweights(length(xs));
    maxiter::Integer=200, tol::Real=TOL
) where T<:Real
    # Expectation[V, V ~ GIG[a, b, -1]]
    @inline GIG_Ev(a, b) = begin
        tmp = sqrt(a / b)
        tmp * exp(lbesselk(0, tmp * b) - lbesselk(1, tmp * b))
    end
    # Expectation[1/V, V ~ GIG[a, b, -1]]
    @inline GIG_Evinv(a, b) = begin
        tmp = sqrt(a * b)
        b / tmp * exp(lbesselk(2, tmp) - lbesselk(1, tmp))
    end
    
    # sufficient statistics
    S0, S1 = sum(ws), sum(xs, ws)
    S2 = S3 = S4 = S5 = zero(T)
    # Q(z, p, μ, σ) = (
    #     S0 * (log(σ) - log(z) - z)
    #     - p * z * (S1 - S0 * μ) / σ
    #     + z/2 * S5 + z/2/σ^2 * (S3 - 2μ * S4 + μ^2 * S5)
    #     + z/2 * (1 + p^2) * S2
    # )
    z, p, μ, σ = params(d)
    nll_prev = Inf
    for e in 1:maxiter
        # E step
        S2 = S3 = S4 = S5 = zero(T)
        for (x, w) in zip(xs, ws)
            # Params of posterior GIG[a, b, -1]
            a = (1 + ((x - μ) / σ)^2) * z
            b = (1 + p^2) * z
            Ev = GIG_Ev(a, b)
            Evinv = GIG_Evinv(a, b)

            # Sum of expected sufficient statistics
            S2 += w * Ev
            S3 += w * x^2 * Evinv
            S4 += w * x^1 * Evinv
            S5 += w * x^0 * Evinv
        end
        #@show (S2, S3, S4, S5)

        # M-step
        C1 = S2 * S5 - S0^2
        C2 = max(1e-6, S2 + S5 - 2 * S0) # > 0
        #@assert C2 > 0
        D  = C1 * S3 + 2 * S0 * S1 * S4 - S2 * S4^2 - S1^2 * S5

        μ = -(S0 * S1 - S2 * S4) / C1
        p = sqrt(C2 / C1 / D) * abs(S0 * S4 - S1 * S5)
        σ = sqrt(D / C1 / C2)
        z = S0 / C2 # (z > 0, S0 > 0) => (C2 > 0)

        #p1, p2 = p, -p
        #p = ifelse(-p1 * (S1 - S0 * μ) < -p2 * (S1 - S0 * μ), p1, p2)
        p = ifelse(-p * (S1 - S0 * μ) < 0, p, -p)

        # Stopping criterion
        d = NIG(z, p, μ, σ)
        nll = -loglikelihood(d, xs, ws)
        @assert nll <= nll_prev
        (nll_prev - nll < tol) && break
        nll_prev = nll
    end

    NIG(z, p, μ, σ)
end

function Distributions.fit_mle(
    ::Type{NIG}, xs::AbstractVector{T}, ws::AbstractWeights=uweights(length(xs));
    maxiter::Integer=500, tol::Real=TOL
) where T<:Real
    # Initial method of moments estimates
    excess_kurt = kurtosis(xs, ws)
    p, μ = zero(T), mean(xs, ws)
    z = 3 / ifelse(excess_kurt < 0, T(1e-3), excess_kurt)
    σ = std(xs, ws) / sqrt(z)

    # EM from these initial conditions
    Distributions.fit_mle(NIG(z, p, μ, σ), xs, ws; maxiter, tol)
end