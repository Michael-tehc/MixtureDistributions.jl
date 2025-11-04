export StudentSkewed

struct StudentSkewed{T<:Real} <: ContinuousUnivariateDistribution
	ν::T
	p::T
	μ::T
	σ::T
	"""
	    StudentSkewed(ν::Real, p=0, μ=0, σ=1)

	X_n = μ + p V_n + σ sqrt(V_n) ϵ_n
	"""
	function StudentSkewed(ν::T, p::T=zero(T), μ::T=zero(T), σ::T=one(T)) where T<:Real
		@assert σ > 0
		new{T}(ν, p, μ, σ)
	end
end

Random.rand(rng::AbstractRNG, d::StudentSkewed) = begin
	(; ν, p, μ, σ) = d
	V = rand(rng, InverseGamma(ν/2, ν/2))
	μ + p * V + σ * sqrt(V) * randn(rng)
end

Distributions.mean(d::StudentSkewed) = begin
	(; ν, p, μ, σ) = d
	μ + p * mean(InverseGamma(ν/2, ν/2))
end

Distributions.var(d::StudentSkewed) = begin
	(; ν, p, μ, σ) = d
	mixing = InverseGamma(ν/2, ν/2)
	p^2 * var(mixing) + σ^2 * mean(mixing)
end

function Distributions.fit_mle(
	d::StudentSkewed, xs::AbstractVector{<:Real}, ws::AbstractWeights=uweights(length(xs));
	maxiters::Int=100, tol::Real=TOL
)
	(; ν, p, μ, σ) = d
	v = σ^2
	N = sum(ws)
	
	#dbesselk(p::Real, x::Real, dp=1e-7) = (besselk(p, x) - besselk(p-dp, x)) / dp
	dlbesselk(p, x, dp=1e-7) = (lbesselk(p,x) - lbesselk(p-dp,x)) / dp

	for ep in 1:maxiters
		# E step
		S0 = S1 = S2 = S3 = S4 = S5 = 0
		for (w, x) in zip(ws, xs)
			# posterior GIG(lmb, a, b)
			lmb = -(1+ν)/2 # <0
			a = (x - μ)^2 / v + ν
			b = p^2 / v
			EV, EVinv, ElnV = if abs(b) > 1e-5
				t0 = lbesselk(lmb, sqrt(a*b))
				tm1 = lbesselk(lmb-1, sqrt(a*b))

				# EV = sqrt(a/b) * exp(
				# 	lbesselk(lmb+1, sqrt(a*b)) - lbesselk(lmb, sqrt(a*b))
				# )
				# Express K_{λ+1}(x) in terms of K_{λ-1}(x) and K_{λ}(x)
				(;
					EV = 2lmb/b + sqrt(a/b) * exp(tm1 - t0),
					EVinv = sqrt(b/a) * exp(tm1 - t0),
					ElnV = log(a/b)/2 - dlbesselk(-lmb, sqrt(a*b))
				)
			else
				if lmb < -1
					# posterior InvGamma(-lmb, a/2)
					(;
						EV=-a/2/(1+lmb),
						EVinv=-2lmb/a,
						ElnV=log(a/2) - digamma(-lmb)
					)
				else
					@assert false "Got $lmb >= -1, should be unreachable!"
				end
			end

			S0 += w * x^0 * EVinv
			S1 += w * x^1 * EVinv
			S2 += w * x^2 * EVinv
			S3 += w * x
			S4 += w * EV
			S5 += w * ElnV
		end

		# M step
		μ = (S3/N - S1*S4 / N^2) / (1 - S0*S4 / N^2)
		p = (S1/N - S0*S3 / N^2) / (1 - S0*S4 / N^2)
		objective(v::Real) = (
			(p^2 * S4/N)/(2v) - (p * S3/N)/v + S2/N/(2v) + p * μ/v - (μ * S1/N)/v +
			μ^2*S0/N/(2v) + log(v)/2
		)
		v = min_golden(objective, 0, 2v, 1e-3)
		ν = min_golden(
			nu -> let
				-nu/2 * log(nu/2) + loggamma(nu/2) + nu/2 * (S0 + S5)/N
			end, 0, 100ν, 1e-3
		)
		#@show (; ν, p, μ, σ=sqrt(v))
	end
	StudentSkewed(ν, p, μ, sqrt(v))
end