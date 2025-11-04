export LogNormalNormal

struct LogNormalNormal{T<:Real} <: CompoundMixture
	normal::Normal{T}
	mixing::LogNormal{T}
end

"""
```
LogNormalNormal(;
	m::Real=0.5 * log(3) - log(2), s::Real=sqrt(log(4/3)),
	μ::Real=0.0, σ::Real=1.0
)
```

LogNormal-Normal distribution.
Defaults ensure `var(x)==1` and `kurtosis(x)==1`.

```
X|V ~ Normal(μ, σ * sqrt(V)),
V   ~ LogNormal(m, s)
=> X~ LogNormalNormal(m, s, μ, σ)
```
"""
function LogNormalNormal(;
	m::Real=0.5 * log(3) - log(2), s::Real=sqrt(log(4/3)),
	μ::Real=0.0, σ::Real=1.0
)
	m, s, μ, σ = promote(m, s, μ, σ)
	LogNormalNormal(Normal(μ, σ), LogNormal(m, s))
end

Base.:+(d::LogNormalNormal, a::Real) = LogNormalNormal(d.normal + a, d.mixing)
Base.:/(d::LogNormalNormal, s::Real) = LogNormalNormal(d.normal / s, d.mixing)

function _integrate_lognormal(d::LogNormalNormal, func)
	# P(node_left < v < node_right) > 0.9999994 for log-normal,
	# so these points contain 99% of the probability density,
	# thus we need to integrate at least within this interval.
	node_left = exp(d.mixing.μ - 5d.mixing.σ)
	node_right = exp(d.mixing.μ + 5d.mixing.σ)
	myquadgk(
		func,
		# Extra quadrature nodes help when d.mixing.σ ≈ 0.
		# Otherwise mixture PDF becomes flat zero
		# bc quadrature misses narrow peak of mixing distribution.
		0, node_left, node_right, Inf
	)
end

function Distributions.logpdf(d::LogNormalNormal, x::Real)
	μ, σ = d.normal.μ, d.normal.σ
	_integrate_lognormal(
		d, v -> pdf(Normal(μ, σ * sqrt(v)), x) * pdf(d.mixing, v)
	) |> log
end

function Distributions.cdf(d::LogNormalNormal, x::Real)
	μ, σ = d.normal.μ, d.normal.σ
	_integrate_lognormal(
		d, v -> cdf(Normal(μ, σ * sqrt(v)), x) * pdf(d.mixing, v)
	)
end

function Random.rand(rng::Random.AbstractRNG, d::LogNormalNormal)
	m, s = d.normal.μ, d.normal.σ
	v = rand(rng, d.mixing)
	m + s * sqrt(v) * randn(rng)
end

Distributions.mean(d::LogNormalNormal) = mean(d.normal)
Distributions.var(d::LogNormalNormal) = var(d.normal) * mean(d.mixing)
Distributions.skewness(d::LogNormalNormal) = 0.0
Distributions.kurtosis(d::LogNormalNormal) = 3exp(d.mixing.σ^2) - 3 # EXCESS kurtosis

raw_moment(d::LogNormalNormal, ::Val{3}) = begin
	mu = mean(d.normal)
	mu^3 + 3 * mu * var(d.normal) * mean(d.mixing)
end
raw_moment(d::LogNormalNormal, ::Val{4}) = begin
	mu, s = mean(d.normal), std(d.normal)
	mu^3 + 6 * mu^2 * s^2 * mean(d.mixing) + 3 * s^4 * raw_moment(d.mixing, 2)
end

Distributions.mode(d::LogNormalNormal) = mode(d.normal)

"""`CVaR(d::LogNormalNormal, α::Real)`

Conditional VaR, aka Expected Shortfall.
Describes the _left_ tail of the distribution,
and is thus a _negative_ number.
"""
function CVaR(d::LogNormalNormal, α::Real)
	@assert α > 0
	μ, σ = d.normal.μ, d.normal.σ
	mix_VaR = VaR(d, α)
	_integrate_lognormal(
		d,
		v -> let
			dist = Normal(μ, σ * sqrt(v))
			(-σ^2 * v * pdf(dist, mix_VaR) + μ * cdf(dist, mix_VaR)) * pdf(d.mixing, v)
		end
	) / α
end
