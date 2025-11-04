export GeneralizedHyperbolic

@doc raw"""
    GeneralizedHyperbolic(α, β, δ, μ=0, λ=1)

The *generalized hyperbolic (GH) distribution* with traditional parameters:

- $\alpha>0$ (shape);
- $-\alpha<\beta<\alpha$ (skewness);
- $\delta>0$ ("scale", but not really, because it appears as an argument to the modified Bessel function of the 2nd kind in the normalizing constant);
- $\mu\in\mathbb R$ (location);
- $\lambda\in\mathbb R$ is a shape parameter, where $\lambda\neq 1$ makes the distribution "generalized"

has probability density function:

```math
\frac{
 (\gamma/\delta)^{\lambda}
}{
 \sqrt{2\pi} K_{\lambda}(\delta \gamma)
}
e^{\beta (x-\mu)}
\frac{
 K_{\lambda-1/2}\left(\alpha\sqrt{\delta^2 + (x-\mu)^2}\right)
}{
 \left(\alpha^{-1} \sqrt{\delta^2 + (x-\mu)^2}\right)^{1/2 - \lambda}
}, \quad\gamma=\sqrt{\alpha^2 - \beta^2}
```

These parameters are actually stored in `struct GeneralizedHyperbolic{T<:Real}`.

External links:

* [Generalized hyperbolic distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution).
* [`HyperbolicDistribution` in Wolfram language](https://reference.wolfram.com/language/ref/HyperbolicDistribution.html).
"""
struct GeneralizedHyperbolic{T<:Real} <: CompoundMixture
    α::T
    β::T
    δ::T
    μ::T
    λ::T
    function GeneralizedHyperbolic(α::T, β::T, δ::T, μ::T=zero(T), λ::T=one(T); check_args::Bool=true) where T<:Real
        check_args && @check_args GeneralizedHyperbolic (α, α > zero(α)) (δ, δ > zero(δ)) (β, -α < β < α)
        new{T}(α, β, δ, μ, λ)
    end
end

GeneralizedHyperbolic(α::Real, β::Real, δ::Real, μ::Real=0, λ::Real=1; check_args::Bool=true) =
    GeneralizedHyperbolic(promote(α, β, δ, μ, λ)...; check_args)

@doc raw"""
    GeneralizedHyperbolic(Val(:locscale), z, p=0, μ=0, σ=1, λ=1)

Location-scale parameterization [1] of the generalized hyperbolic distribution with parameters

- $z>0$ (shape);
- $p\in\mathbb R$ measures skewness ($p=0$ results in a symmetric distribution);
- $\mu\in\mathbb R$ and $\sigma>0$ are location and scale;
- $\lambda\in\mathbb R$ is a shape parameter, where $\lambda\neq 1$ makes the distribution "generalized"

has probability density function:

```math
\frac{\sqrt z}{
 \sqrt{2\pi} K_{\lambda}(z)
}
e^{p z \cdot\varepsilon}
\sqrt{
 \left(\frac{1+\varepsilon^2}{1+p^2}\right)^{\lambda - 1/2}
}
K_{\lambda-1/2}\left[
 z \sqrt{(1+p^2)(1+\varepsilon^2)}
\right]
```

These parameters are _not_ stored in `struct GeneralizedHyperbolic`.
Use `params(d, Val(:locscale))`, where `d` is an instance of `GeneralizedHyperbolic`, to retrieve them.

Advantages of this parameterization:

- It's truly location-scale, whereas $\delta$ in the traditional parameterization isn't a true scale parameter.
- All parameters are either positive or unconstrained. The traditional parameterization has the complicated linear constraint $-\alpha<\beta<\alpha$.

References:

1. Puig, Pedro, and Michael A. Stephens. “Goodness-of-Fit Tests for the Hyperbolic Distribution.” The Canadian Journal of Statistics / La Revue Canadienne de Statistique 29, no. 2 (2001): 309–20. https://doi.org/10.2307/3316079.
"""
GeneralizedHyperbolic(::Val{:locscale}, z::Real, p::Real=0, μ::Real=0, σ::Real=1, λ::Real=1; check_args::Bool=true) =
	GeneralizedHyperbolic(z * sqrt(1 + p^2)/σ, z * p / σ, σ, μ, λ; check_args)

Distributions.params(d::GeneralizedHyperbolic) = (d.α, d.β, d.δ, d.μ, d.λ)
Distributions.params(d::GeneralizedHyperbolic, ::Val{:locscale}) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    (; z=δ * γ, p=β / γ, μ, σ=δ, λ)
end
Distributions.partype(::GeneralizedHyperbolic{T}) where T = T

Distributions.@distr_support GeneralizedHyperbolic -Inf Inf

"""
    mode(::GeneralizedHyperbolic)

- Exact formulae are used for λ=1 and λ=2.
- For other values of λ quadratic fit search is used. The initial bracket `lo < mode < hi` is computed based on
inequalities from [1, eq. 2.27].

## References

1. Robert E. Gaunt, Milan Merkle, "On bounds for the mode and median of the generalized hyperbolic and related distributions", Journal of Mathematical Analysis and Applications, Volume 493, Issue 1, 2021, 124508, ISSN 0022-247X, https://doi.org/10.1016/j.jmaa.2020.124508.
"""
function Distributions.mode(d::GeneralizedHyperbolic)
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    if λ ≈ 1
        μ + β * δ / γ # Wolfram
    elseif λ ≈ 2
        μ + β / α / γ^2 * (α + sqrt(β^2 + (α * δ * γ)^2)) # Wolfram
    else
        lo, hi = let # Bounds for the bracketing interval
            EX = mean(d)
            # eq. 2.27 from the paper: EX - upper < mode < EX - lower for β>0.
            # When β<0, the inequality is reversed.
            lower = β/γ^2 * (
                1/2 + sqrt(λ^2 + δ^2 * γ^2) - sqrt((λ - 1/2)^2 + δ^2 * γ^2)
            )
            upper = β/γ^2 * (
                5/2 + sqrt((λ + 1)^2 + δ^2 * γ^2) - sqrt((λ - 3/2)^2 + δ^2 * γ^2)
            )
            if β < 0
                upper, lower = lower, upper
            end
            EX - upper, EX - lower
        end

        min_quadfit(x -> -logpdf(d, x), lo, hi)
    end
end

Distributions.mean(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)
    μ + β * δ / γ * besselk(1 + λ, δ * γ) / besselk(λ, δ * γ)
end

Distributions.var(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    δ / γ * t1/t0 - (β * δ / γ * t1/t0)^2 + (β * δ / γ)^2 * t2/t0
end

Distributions.skewness(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    t3 = besselk(3 + λ, δ * γ)
    (
        -3β * (δ / γ * t1/t0)^2 + 2 * (β * δ / γ * t1/t0)^3 + 3β * (δ / γ)^2 * t2/t0
        - 3 * (β * δ / γ)^3 * t1*t2/t0^2 + (β * δ / γ)^3 * t3/t0
    ) / sqrt(
        δ / γ * t1/t0 - (β * δ / γ * t1/t0)^2 + (β * δ / γ)^2 * t2/t0
    )^3
end

Distributions.kurtosis(d::GeneralizedHyperbolic) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    t0 = besselk(0 + λ, δ * γ)
    t1 = besselk(1 + λ, δ * γ)
    t2 = besselk(2 + λ, δ * γ)
    t3 = besselk(3 + λ, δ * γ)
    t4 = besselk(4 + λ, δ * γ)
    (
        3 * γ^2 * t0^3 * t2 + 6 * β^2 * γ * δ * t0 * (t1^3 - 2t0 * t1 * t2 + t0^2 * t3)
        + β^4 * δ^2 * (-3 * t1^4 + 6t0 * t1^2 * t2 - 4 * t0^2 * t1 * t3 + t0^3 * t4)
    ) / (
        γ * t0 * t1 + β^2 * δ * (-t1^2 + t0 * t2)
    )^2 - 3 # EXCESS kurtosis
end

Distributions.logpdf(d::GeneralizedHyperbolic, x::Real) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    (
        -0.5log(2π) - lbesselk(λ, γ * δ) + λ * (log(γ) - log(δ))
        + β * (x - μ)
        + (λ - 1/2) * (0.5log(δ^2 + (x - μ)^2) - log(α))
        + lbesselk(λ - 1/2, α * sqrt(δ^2 + (x - μ)^2))
    )
end

Distributions.cdf(d::GeneralizedHyperbolic, x::Real) =
	if isinf(x)
		(x < 0) ? zero(x) : one(x)
	elseif isnan(x)
		typeof(x)(NaN)
	else
        quadgk(z -> pdf(d, z), -Inf, x, maxevals=10^4)[1]
	end

@quantile_newton GeneralizedHyperbolic

Distributions.mgf(d::GeneralizedHyperbolic, t::Number) = begin
    α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    g = sqrt(α^2 - (t + β)^2)
    exp(t * μ) / g^λ * sqrt((α - β) * (α + β))^λ * besselk(λ, g * δ) / besselk(λ, γ * δ)
end

Distributions.cf(d::GeneralizedHyperbolic, t::Number) = mgf(d, 1im * t)

@doc raw"""
    rand(::AbstractRNG, ::GeneralizedHyperbolic)

Sample from `GeneralizedHyperbolic(α, β, δ, μ, λ)` using its mixture representation:

```math
\begin{aligned}
\gamma &= \sqrt{\alpha^2 - \beta^2}\\
V &\sim \mathrm{GeneralizedInverseGaussian}(\delta / \gamma, \delta^2, \lambda)\\
\xi &= \mu + \beta V + \sqrt{V} \varepsilon, \quad\varepsilon \sim \mathcal N(0,1)
\end{aligned}
```

Then ξ is distributed as `GeneralizedHyperbolic(α, β, δ, μ, λ)`.

Verified in Wolfram Mathematica:

```
In:= TransformedDistribution[\[Mu] + \[Beta]*V + 
  Sqrt[V] \[Epsilon], {\[Epsilon] \[Distributed] NormalDistribution[],
   V \[Distributed] 
   InverseGaussianDistribution[\[Delta]/
     Sqrt[\[Alpha]^2 - \[Beta]^2], \[Delta]^2, \[Lambda]]}]

Out= HyperbolicDistribution[\[Lambda], \[Alpha], \[Beta], \[Delta], \[Mu]]
```

Note that here λ is the first parameter, while in this implementation it's the _last_ one.
"""
Random.rand(rng::AbstractRNG, d::GeneralizedHyperbolic) = begin
	α, β, δ, μ, λ = params(d)
    γ = sqrt(α^2 - β^2)

    V = Random.rand(rng, GeneralizedInverseGaussian(δ/γ, δ^2, λ))
    μ + β * V + sqrt(V) * randn(rng)
end

function step_EM(k::Integer, s::T, p::T, μ::T, τ::T, xs::AbstractVector{<:Real}) where T<:Real
    # E-step
    S0 = length(xs)
    S1 = S3 = S4 = S5 = S6 = 0.0
    for x in xs
        # Posterior GIG(a, b, k + 1/2 - 1/2) = GIG(a, b, k)
        a = s^2 * (1 + (x - μ)^2 * τ^2)
        b = s^2 * (1 + p^2)

        tmp = sqrt(a * b)
        tmp2 = sqrt(a / b)
        E_Vinv = b / tmp * besselk(k-1, tmp) / besselk(k, tmp)
        E_V = tmp2 * besselk(k+1, tmp2 * b) / besselk(k, tmp2 * b)
        
        S1 += x
        S3 += x^2 * E_Vinv # > 0
        S4 += x^1 * E_Vinv
        S5 += x^0 * E_Vinv # > 0
        S6 += E_V          # > 0
    end

    # M-step
    nll(θ::AbstractVector{<:Real}) = begin
        s_noabs, p, μ, τ_noabs = θ
        s, τ = abs(s_noabs), abs(τ_noabs)

        -(
              S0   * (log(τ) + log(s) - lbesselk_int(k, s^2) - μ * p * τ * s^2)
            + S1   * p * τ * s^2
            - S3/2 * τ^2 * s^2
            + S4   * μ * τ^2 * s^2
            - S5/2 * s^2 * (1 + μ^2 * τ^2)
            - S6/2 * s^2 * (1 + p^2)
        ) / S0
    end

    sol = Optim.optimize(nll, [s, p, μ, τ], Optim.LBFGS(), Optim.Options(f_calls_limit=50), autodiff=:forward)
    s_noabs, p, μ, τ_noabs = Optim.minimizer(sol)
    abs(s_noabs), p, μ, abs(τ_noabs)
end

"""
    fit(::Type{GeneralizedHyperbolic}, xs::AbstractVector{<:Real}, k::Integer; niter::Integer=50)

Fit using the `(λ=k + 1/2, z=s^2, p, μ, σ=1/τ)` parameterization.
"""
function Distributions.fit_mle(
    ::Type{GeneralizedHyperbolic}, xs::AbstractVector{<:Real}, k::Integer; niter::Integer=500
)
    z, p = 1.0, 0.0
    μ, σ = mean(xs), sqrt(
        z * var(xs) * exp(lbesselk_int(k, z) - lbesselk_int(k+1, z))
    )

    s, τ = sqrt(z), 1/σ
    for _ in 1:niter
        s, p, μ, τ = step_EM(k, s, p, μ, τ, xs)
    end
    z, σ = s^2, 1/τ

    # Transform back
    α = z * sqrt(1 + p^2) / σ
    β = z * p / σ
    δ = σ
    λ = k + 1/2
    GeneralizedHyperbolic(α, β, δ, μ, λ)
end