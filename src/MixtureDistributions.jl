module MixtureDistributions
export VaR, CVaR

using Random, LinearAlgebra, Distributions
using Distributions: check_args, @check_args, @quantile_newton
import StatsBase: AbstractWeights, AnalyticWeights, uweights
using StaticArrays

import Polynomials: Polynomial
import QuadGK: quadgk
import Optim
import SpecialFunctions: beta, loggamma, digamma, besselk

@doc raw"""
    lambertw_exp(l::Real)

Computes $W_0(e^l)$ without computing $e^l$, so $|l|$ can be massive. Uses https://arxiv.org/abs/2008.06122.
"""
lambertw_exp(l::Real) = begin
	b = (l < 1) ? exp(l - 1) : (l - log(l))

	# eqns 21, 29 from paper
	tmp = 1 + l
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
	b = b / (1 + b) * (tmp - log(b))
end

@doc raw"""
    lambertw(x::Real)

Computes the Lambert W function $W_0(x)$ to `Float64` precision using https://arxiv.org/abs/2008.06122.
"""
lambertw(x::Real) = lambertw_exp(log(x))

const TOL = 1e-4

abstract type CompoundMixture <: Distributions.ContinuousUnivariateDistribution end

Distributions.loglikelihood(d::CompoundMixture, xs::AbstractVector{<:Real}, ws::AbstractWeights)=
	sum(
		w * logpdf(d, x)
		for (x, w) in zip(xs, ws)
	)

Distributions.fit_mle(
	d::Normal, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs)); kwargs...
) = Normal(mean(xs, ws), std(xs, ws))

myquadgk(func, nodes...) = quadgk(func, nodes..., maxevals=1000)[1]

"""
    eweights2(N::Integer, λ::Real)

Exponential weights proportional to `λ^{-t}`. Increase with `t ∈ 1:N`, sum to `N`.
"""
function eweights2(N::Integer, λ::Real)
	@assert 0 < λ < 1
	AnalyticWeights([
		N * (1-λ) / (1 - λ^N) * λ^(N - t) # (N-t) prevents overflow!
		for t in 1:N
	])
end

raw_moment(d::CompoundMixture, k::Integer) = raw_moment(d, Val(k))
raw_moment(d::CompoundMixture, ::Val{0}) = 1
raw_moment(d::CompoundMixture, ::Val{1}) = mean(d)
raw_moment(d::CompoundMixture, ::Val{2}) = var(d) + mean(d)^2

"Central kth moment"
function Distributions.moment(d::CompoundMixture, k::Integer, m::Real=mean(d))
	@assert k >= 0
	if k == 0
		raw_moment(d, 0)
	elseif k == 1
		raw_moment(d, 1) - m
	elseif k == 2
		raw_moment(d, 2) - 2m * mean(d) + m^2
	elseif k == 3
		raw_moment(d, 3) - 3 * m * raw_moment(d, 2) + 3 * m^2 * mean(d) - m^3
	elseif k == 4
		(
			raw_moment(d, 4) - 4 * m * raw_moment(d, 3) + 6 * m^2 * raw_moment(d, 2)
			- 4 * m^3 * raw_moment(d, 1) + m^4
		)
	else
		ArgumentError("Moments of order $k not implemented") |> throw
	end
end

Distributions.quantile(d::CompoundMixture, q::Real) =
	Distributions.quantile_newton(d, q)

VaR(data::AbstractVector{<:Real}, α::Real) = Distributions.quantile(data, α)

"""`VaR(d::ContinuousUnivariateDistribution, α::Real)`

Value-at-risk (left α-quantile), a _negative_ number.
"""
VaR(d::Distributions.ContinuousUnivariateDistribution, α::Real) = Distributions.quantile(d, α)

CVaR(d::Normal, α::Real) = d.μ - d.σ/α * pdf(Normal(), quantile(Normal(), α))

"""`CVaR(data::AbstractVector{<:Real}, α::Real)`

Conditional VaR, aka Expected Shortfall.
Describes the _left_ tail of the distribution,
and is thus a _negative_ number.
"""
function CVaR(data::AbstractVector{<:Real}, α::Real)
	empirical_VaR = VaR(data, α)
	mean(
		obs * (obs < empirical_VaR)
		for obs in data
	) / α
end

abstract type AbstractRegularization end

struct RegInverseGamma{T} <: AbstractRegularization
	a::T

	"""
	    RegInverseGamma(a::Real)

	Prior distribution `InverseGamma(a, a+1)` with mode at 1.
	"""
	function RegInverseGamma(a::T) where T<:Real
		@assert a > 0
		new{T}(a)
	end
end

Distributions.logpdf(d::RegInverseGamma, x::Real) = logpdf(InverseGamma(d.a, d.a+1), x)

struct RegLogNormal{T} <: AbstractRegularization
	σ::T

	"""
	    RegLogNormal(σ)

	Prior distribution `LogNormal(0, σ)` with mode at 1.
	"""
	function RegLogNormal(σ::T) where T<:Real
		@assert σ >= 0
		new{T}(σ)
	end
end

RegLogNormal() = RegLogNormal(1)

Distributions.logpdf(d::RegLogNormal, x::Real) = logpdf(LogNormal(0, d.σ), x)

include("lbesselk.jl")
include("min_quadfit.jl")
include("min_golden.jl")

include("Student.jl")
include("StudentSkewed.jl")
include("InverseGaussian.jl")
include("Hyperbolic.jl")
include("NIG.jl")
include("VarianceGamma.jl")
include("Finite.jl")

end # module MixtureDistributions
