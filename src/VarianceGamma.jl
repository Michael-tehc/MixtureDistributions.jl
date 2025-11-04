export VarianceGamma

@doc raw"""Variance Gamma distribution

```math
X  = \mu + \sigma p V + \sigma \sqrt{V}\varepsilon, \quad V \sim Gamma(\lambda, 1/2)
```
"""
struct VarianceGamma{T<:Real} <: CompoundMixture
	λ::T
	p::T
	μ::T
	σ::T
	function VarianceGamma(λ::Real, p::Real=0, μ::Real=0, σ::Real=1)
		λ, p, μ, σ = promote(λ, p, μ, σ)
		@assert λ > 0
		@assert σ > 0
		new{typeof(p)}(λ, p, μ, σ)
	end
end

Distributions.params(d::VarianceGamma) = (d.λ, d.p, d.μ, d.σ)

function Distributions.logpdf(d::VarianceGamma, x::Real)
	λ, p, μ, σ = params(d)
	p2 = sqrt(1 + p^2)
	z = (x - μ) / σ

	(
		-0.5 * log(pi) - (λ - 0.5) * (log(2) + log(p2)) - loggamma(λ)
		- log(σ)
		+ (λ - 0.5) * log(abs(z)) + p * z + lbesselk(λ - 0.5, abs(z) * p2)
	)
end

# function Distributions.cdf(d::VarianceGamma, x::Real)
# 	ϕ, γ, μ, σ = d.ϕ, d.γ, d.normal.μ, d.normal.σ
# 	mixing_mode = mode(d.mixing)
# 	myquadgk(
# 		v -> cdf(Normal(μ + (ϕ-γ)/2 * v, σ * sqrt(v)), x) * pdf(d.mixing, v),
# 		0, 0.5mixing_mode, mixing_mode, 1.5mixing_mode, Inf
# 	)
# end

"X = μ + σ p V + σ √V ε, V ~ Gamma(λ, 1/2)"
function Random.rand(rng::Random.AbstractRNG, d::VarianceGamma)
	λ, p, μ, σ = params(d)
	v = rand(rng, Gamma(λ, 2))
	μ + σ * p * v + σ * sqrt(v) * randn(rng)
end

function Distributions.mean(d::VarianceGamma)
	λ, p, μ, σ = params(d)
	μ + p * σ * 2λ
end

function Distributions.var(d::VarianceGamma)
	λ, p, μ, σ = params(d)
	σ^2 * (1 + 2 * p^2) * 2λ
end

function Distributions.skewness(d::VarianceGamma)
	λ, p, μ, σ = params(d)
	sqrt(2/λ) * p * (3 + 4 * p^2) / (1 + 2 * p^2)^(3/2)
end

function Distributions.kurtosis(d::VarianceGamma)
	λ, p, μ, σ = params(d)
	(3 + 24 * p^2 * (1 + p^2)) / (1 + 2 * p^2)^2 / λ # EXCESS kurtosis!!
end

function Distributions.fit_mle(
	d::VarianceGamma{T}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
	maxiter::Integer=200, tol::Real=TOL, quiet::Bool=true
) where T<:Real
	@assert maxiter > 0
	@assert tol > 0
	λ, p, μ, σ = params(d)

	"Derivative of ``K_p(x)`` wrt `p`"
	d1besselk(p::Real, x::Real, dp::Real=1e-7) = (besselk(p, x) - besselk(p - dp, x)) / dp

	d1lbesselk(p::Real, x::Real, dp::Real=1e-7) = (lbesselk(p, x) - lbesselk(p - dp, x)) / dp

	loglik_prev = loglikelihood(d, xs, ws)
	S0, S1 = sum(ws), sum(xs, ws)
	for _ in 1:maxiter
		if !quiet
			@show loglik_prev
		end

		# E step
		S2 = S3 = S4 = S5 = S6 = zero(T) # sufficient statistics
		for (x, w) in zip(xs, ws)
			# Posterior GIG(a, b, p)
			a = ((x - μ) / σ)^2
			b = 1 + p^2
			λ_posterior = λ - 1/2

			if a < 1e-6
				# FIXME: danger of computing besselk(p, sqrt(0 * b))
				if λ_posterior > 1
					# Posterior GIG(0, b, p) = Gamma(shape=p, rate=b/2)
					EV = 2λ_posterior / b
					EVinv = b/2 / (λ_posterior - 1)
					ElnV = log(2/b) + digamma(λ_posterior)
				else
					# FIXME: what to do here?
					EV = EVinv = 1.0
					ElnV = 0.0
				end
			else
				# Posterior GIG(a, b, p)
				bm1 = lbesselk(λ_posterior-1, sqrt(a * b))
				b0  = lbesselk(λ_posterior+0, sqrt(a * b))
				bp1 = lbesselk(λ_posterior+1, sqrt(a * b))

				# Expectations wrt posterior
				EV = a / sqrt(a * b) * exp(bp1 - b0) #bp1 / b0
				EVinv = b / sqrt(a * b) * exp(bm1 - b0)#bm1 / b0

				# FIXME: somehow `- d1besselk(-λ_posterior, sqrt(a * b)) * exp(-b0)`
				# provides better results than `d1lbesselk(-λ_posterior, sqrt(a * b))`,
				# but they should be equivalent???
				# When `d1lbesselk` is used, log-likelihood doesn't increase monotonically
				ElnV = 0.5 * log(a / b) - d1besselk(-λ_posterior, sqrt(a * b)) * exp(-b0)
				# ElnV = 0.5 * log(a / b) + d1lbesselk(-λ_posterior, sqrt(a * b))
			end

			# Weighted sufficient statistics
			S2 += w * EV
			S3 += w * x^2 * EVinv
			S4 += w * x^1 * EVinv
			S5 += w * x^0 * EVinv
			S6 += w * ElnV
		end

		# M step
		# In[34]:= Module[{nll = 
		# S0 (Log[\[Sigma]] + p \[Mu]/\[Sigma]) - p/\[Sigma]*S1 + (
		# 	S3 - 2 \[Mu] S4 + \[Mu]^2 S5)/(2 \[Sigma]^2) + p^2/2 S2, nll2, 
		# sol},
		# sol = Solve[{
		# 	D[nll, \[Mu]] == 0,
		# 	D[nll, p] == 0
		# 	}, {\[Mu], p}][[1]] // FullSimplify;
		# nll2 = FullSimplify[nll //. sol //. {\[Sigma] -> Sqrt[v]}];
		# FullSimplify[{\[Mu], p, v} //. sol //. Solve[D[nll2, v] == 0, v][[1]]]
		# ]

		# Out[34]= {
		# 	μ -> (S0 S1 - S2 S4)/(S0^2 - S2 S5),
		# 	p -> (S0 S4 - S1 S5)/(S0^2 \[Sigma] - S2 S5 \[Sigma]),
		# 	σ^2 -> (S0^2 S3 - 2 S0 S1 S4 + S2 S4^2 + S1^2 S5 - S2 S3 S5)/(S0^3 - S0 S2 S5)
		# }
		μ = (S1 - S2 * S4 / S0) / (S0 - S2 * S5 / S0)
		σ_sq = max(1e-6, (
			S3 - 2S1*S4/S0 + (S2 * S4^2 + S1^2 * S5 - S2 * S3 * S5)/S0^2
		) / (
			S0 - S2*S5/S0
		))
		σ = sqrt(σ_sq)
		p = (S4 - S1 * S5 / S0) / (S0 - S2 * S5 / S0) / σ

		λ = min_golden(l -> loggamma(l) + l * (log(2) - S6 / S0), max(0.0, λ - 100), λ + 100)

		# Track weighted log-likelihood
		loglik_curr = loglikelihood(VarianceGamma(λ, p, μ, σ), xs, ws)
		@assert loglik_curr >= loglik_prev
		dloglik = loglik_curr - loglik_prev
		(dloglik < tol) && break
		loglik_prev = loglik_curr
	end

	VarianceGamma(λ, p, μ, σ)
end

function Distributions.fit_mle(
	::Type{VarianceGamma}, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
	maxiter::Integer=200,  tol::Real=TOL
)
	# Initial values: method of moments
	p = 0.0
	λ = 3 / max(1e-4, kurtosis(xs, ws)) # div by EXCESS kurtosis!!
	σ = std(xs, ws) / sqrt(2λ)
	μ = mean(xs, ws)

	Distributions.fit_mle(
		VarianceGamma(λ, p, μ, σ), xs, ws; maxiter, tol
	)
end