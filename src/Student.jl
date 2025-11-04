export Student

"""`StudentStandard(nu::Real)`

Student's t distribution. Mixing distribution is `InverseGamma(nu/2, nu/2)`.
"""
struct StudentStandard{T<:Real} <: CompoundMixture
	nu::T
	function StudentStandard(nu::T) where T<:Real
		@assert nu > 0
		new{T}(nu)
	end
end

Distributions.logpdf(d::StudentStandard, x::Real) =
	(
		d.nu * log(d.nu) - (1+d.nu) * log(d.nu + x^2)
		- 2log(beta(d.nu/2, 1/2))
	) / 2

@doc raw"""`cdf(d::StudentStandard, x::Real)`

Uses the fact that if $X \sim \mathrm{St}(\nu)$, then
$Y = \frac{\nu}{\nu + X^2} \sim B(\nu/2, 1/2)$.

https://stats.stackexchange.com/a/394983
"""
function Distributions.cdf(d::StudentStandard, x::Real)
	(abs(x) < 1e-5) && return 0.5

	y = d.nu / (d.nu + x^2)
	beta_cdf = Distributions.cdf(Distributions.Beta(d.nu/2, 1/2), y) 
	(x < 0) ? beta_cdf/2 : (1 - beta_cdf/2)
end

struct Student{T<:Real} <: CompoundMixture
	ν::T
	μ::T
	σ::T
	function Student(ν::T, μ::T=zero(T), σ::T=one(T)) where T<:Real
		new{T}(ν, μ, σ)
	end
end

function Distributions.logpdf(d::Student, x::Real)
	(; ν, μ, σ) = d
	-log(σ) + Distributions.logpdf(StudentStandard(ν), (x - μ)/σ)
end

function Distributions.cdf(d::Student, x::Real)
	(; ν, μ, σ) = d
	Distributions.cdf(StudentStandard(ν), (x - μ) / σ)
end

function Random.rand(rng::Random.AbstractRNG, d::Student)
	(; ν, μ, σ) = d
	v = rand(rng, InverseGamma(ν/2, ν/2))
	μ + σ * sqrt(v) * randn(rng)
end

Distributions.mean(d::Student) = d.μ
function Distributions.var(d::Student{T})::T where T<:Real
	(; ν, μ, σ) = d
	EV = mean(InverseGamma(ν/2, ν/2))
	σ^2 * EV
end

function Distributions.skewness(d::Student{T})::T where T<:Real
	(; ν, μ, σ) = d
	(ν > 3) ? zero(T) : T(NaN)
end

"EXCESS kurtosis"
function Distributions.kurtosis(d::Student{T})::T where T<:Real
	(; ν, μ, σ) = d
	(ν > 4) ? (6/(ν - 4)) : T(NaN)
end

Distributions.mode(d::Student) = d.μ

# function CVaR(d::Student, α::Real)
# 	(; ν, μ, σ) = d
# 	mix_VaR = VaR(d, α)
# 	z = (mix_VaR - m)/s
# 	(
# 		-s^2 * (n + z^2)/(n - 1) * pdf(d, mix_VaR)
# 		+ m * cdf(d, mix_VaR)
# 	) / α
# end

function Distributions.fit_mle(
    d::Student{T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
    maxiter::Integer=200, tol::Real=TOL
) where {T<:Real}
	N = sum(ws)
	(; ν, μ, σ) = d
	v = σ^2

	for _ in 1:maxiter
		S0 = S1 = S2 = S3 = 0
		# E step
		for (w, x)  in zip(ws, xs)
			# Posterior InvGamma(a, λ)
			a = (1 + ν)/2
			λ = ((x - μ)^2 / σ^2 + ν) / 2
			EVinv = a / λ
			ElnV = -digamma(a) + log(λ)

			S0 += w * EVinv
			S1 += w * EVinv * x^1
			S2 += w * EVinv * x^2
			S3 += w * ElnV
		end

		# M step
		μ = S1 / S0
		v = (S2 - 2μ * S1 + μ^2 * S0) / N
		ν = min_golden(
			nu -> -nu * log(nu) + 2 * loggamma(nu/2) + nu * (log(2) + S3/N + S0/N),
			0, 3 * ν
		)
	end
	Student(ν, μ, sqrt(v))
end