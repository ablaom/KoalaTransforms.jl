module KoalaTransforms

# new:
export UnivariateStandardizationScheme
export HotEncodingScheme
export UnivariateBoxCoxScheme, BoxCoxScheme
export fit_transform!

# extended:
export transform, inverse_transform, fit! # from `Koala`

# for use in this module:
import Koala: BaseType, params, type_parameters
import DataFrames: names, AbstractDataFrame, DataFrame
import Distributions

# to be extended:
import Koala: transform, fit!, inverse_transform
import Base: show, showall

# constants:
const N_VALUES_THRESH = 16


## Abstract types

abstract type Scheme <: BaseType
end


## Checklist for new schemes

# - make subtype of `Scheme`.
#
# - If a scheme is already fitted, then a warning should be issued if fit again
#
# - Don't forget to set `fitted = true` at end of `fit!` definition
#
# - For each scheme type `S` provide a constructor of the form
#      function S(X; verbosity=1, args...)
#          s = S(args...)
#          fit!(s, X; verbosity=verbosity)
#          return s
#      end
#
# - Every `fit!` method should have a `verbosity=1` keyword argument
#   and every method calling `fit!` should
#   accept `verbosity=1` as  a keyword and pass to the `fit!` call.


## Fall-back methods

function fit_transform!(s::Scheme, x; verbosity=1)
    fit!(s, x; verbosity=verbosity)
    return transform(s, x)
end

function Base.show(stream::IO, s::Scheme)
    abbreviated(n) = "..."*string(n)[end-2:end]
    type_params = type_parameters(s)
    prefix = (s.fitted ? "" : "unfitted ") 
    if isempty(type_params)
        type_string = ""
    else
        type_string = string("{", ["$T," for T in type_params]..., "}")
    end
    print(stream, string(prefix, typeof(s).name.name,
                         type_string,
                         "@", abbreviated(hash(s))))
end

## `UnivariateStandardizationScheme`

mutable struct UnivariateStandardizationScheme <: Scheme
    
    # hyperparameters: None
    
    # post-fit parameters:
    mu::Float64
    sigma::Float64

    fitted::Bool

    function UnivariateStandardizationScheme()
        ret = new()
        ret.fitted = false
        return ret
    end
    
end

function fit!(s::UnivariateStandardizationScheme, v::AbstractVector{T};
              verbosity=1) where T <: Real

    !s.fitted || warn("Refitting a previously trained transformation scheme.")

    s.mu, s.sigma = (mean(v), std(v))
    s.fitted = true
    return s

end

function transform(s::UnivariateStandardizationScheme, x:: Real)
    if !s.fitted
        throw(Base.error("Attempting to transform according to unfitted scheme."))
    end
    return (x - s.mu)/s.sigma
end

transform(s::UnivariateStandardizationScheme, v::AbstractVector{T} where T <: Real) =
    [transform(s,x) for x in v]

function inverse_transform(s::UnivariateStandardizationScheme, y::Real)
    s.fitted || error("Attempting to transform according to unfitted scheme.")
    return s.mu + y*s.sigma
end

inverse_transform(s::UnivariateStandardizationScheme,
                  w::AbstractVector{T} where T <: Real) =
                      [inverse_transform(s,y) for y in w]

function fit_transform!(s::UnivariateStandardizationScheme,
                        v::AbstractVector{T}; verbosity=1) where T <: Real
    fit!(s, v; verbosity=verbosity)
    return transform(s, v)
end

function UnivariateStandardizationScheme(v::AbstractVector{T};
                                         verbosity=1, args...) where T <: Real
    s = UnivariateStandardizationScheme(; args...)
    return fit!(s, v; verbosity=verbosity)
end


## `HotEncodingScheme`

mutable struct HotEncodingScheme <: Scheme
    
    # hyperparameters:
    drop_last::Bool
    
    # post-fit parameters:
    features::Vector{Symbol} # feature labels
    spawned_features::Vector{Symbol} # feature labels after one-hot encoding
    values_given_feature::Dict{Symbol,Vector{String}}

    fitted::Bool

    function HotEncodingScheme(;drop_last::Bool=false)
        ret = new(drop_last)
        ret.fitted = false
        return ret
    end

end

function fit!(s::HotEncodingScheme, X::AbstractDataFrame; verbosity=1)

    !s.fitted || warn("Refitting an existing transformation scheme.")
    
    s.features = names(X)
    s.values_given_feature = Dict{Symbol,Vector{String}}()
    
    for ft in s.features 
        if eltype(X[ft]) <: AbstractString
            s.values_given_feature[ft] = sort!(unique(X[ft]))
            if s.drop_last
                s.values_given_feature[ft] = s.values_given_feature[ft][1:(end - 1)]
            end
            if verbosity > 0
                n_values = length(keys(s.values_given_feature[ft]))
                println("Spawned $n_values columns to hot-encode $ft.")
            end
        elseif eltype(X[ft]) == Char
            warn("A feature of Char type has been encountered and "*
                 "is being ignored. To be hot-encoded "*
                 "it must first be converted to some AbstractString type.")
        end  
    end

    s.spawned_features = Symbol[]

    for ft in s.features
        if eltype(X[ft]) <: AbstractString
            for value in s.values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in (rare) case subft is not a new feature name:
                while subft in s.features
                    subft = Symbol(string(subft,"_"))
                end

                push!(s.spawned_features, subft)
            end
        else
            push!(s.spawned_features, ft)
        end
    end
    

    s.fitted = true
    return s
end

function transform(s::HotEncodingScheme, X::AbstractDataFrame)
    s.fitted || error("Cannot transform according to unfitted scheme.")

    Xout = DataFrame()
    for ft in s.features
        if eltype(X[ft]) <: AbstractString
            for value in s.values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in case subft is not a new feature name:
                while subft in s.features
                    subft = Symbol(string(subft,"_"))
                end

                Xout[subft] = map(X[ft]) do x
                    x == value ? 1.0 : 0.0
                end 
            end
        else
            Xout[ft] = X[ft]
        end
    end
    return Xout
end

function HotEncodingScheme(X::AbstractDataFrame; verbosity=1, args...)
    s = HotEncodingScheme(;args...)
    return fit!(s, X; verbosity=verbosity)
end


## `UnivariateBoxCoxScheme`

function normalise(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end
                   
function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)

    n  = length(v)
    v = normalise(convert(Vector{Float64}, v))

    # sort and replace with midpoints
    v = midpoints(sort!(v))

    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w= map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end

    return cor(v, w)

end

function boxcox{T<:Real}(lambda, c, x::T)
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::Vector) = [boxcox(lambda, c, x) for x in v]    

function boxcox(v::Vector; n=171, shift::Bool = false)
    m = minimum(v)
    m >= 0 || error("Cannot perform a BoxCox transformation on negative data.")

    c = 0.0 # default
    if shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || throw(DomainError) 
    end
  
    lambdas = linspace(-0.4,3,n)
    scores = [normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[indmax(scores)]
    return  lambda, c, boxcox(lambda, c, v)
end

"""
## `mutable struct UnivariateBoxCoxScheme`

A type for encoding a Box-Cox transformation of a single variable
taking non-negative values, with a possible preliminary shift. Such a
transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

###  Usage

    `s = UnivariateBoxCoxScheme(; n=171, shift=false)`

Returns an unfitted wrapper that on fitting to data (see below) will
try `n` different values of the Box-Cox exponent λ (between `-0.4` and
`3`) to find an optimal value, stored as the post-fit parameter
`s.lambda`. If `shift=true` and zero values are encountered in the
data then the transformation sought includes a preliminary positive
shift, stored as `s.c`. The value of the shift is always `0.2` times
the data mean. If there are no zero values, then `s.c=0`.

    `fit!(s, v)`

Attempts fit an `UnivariateBoxCoxScheme` instance `s` to a
vector `v` (eltype `Real`). The elements of `v` must
all be positive, or a `DomainError` will be thrown. If `s.shift=true`
zero-valued elements are allowed as discussed above. 

    `s = UnivariateBoxCoxScheme(v; n=171, shift=false)`

Combines the previous two steps into one.

    `w = transform(s, v)`

Transforms the vector `v` according to the Box-Cox transformation
encoded in the `UnivariateBoxCoxScheme` instance `s` (which must be
first fitted to some data). Stores the answer as `w`.

See also `BoxCoxScheme` a transformer for selected ordinals in a DataFrame. 

"""
mutable struct UnivariateBoxCoxScheme <: Scheme
    
    # hyperparameters
    n::Int      # nbr values tried in optimizing exponent lambda
    shift::Bool # whether to shift data away from zero
    
    # post-fit parameters:
    lambda::Float64
    c::Float64

    function UnivariateBoxCoxScheme(n::Int, shift::Bool)
        ret = new(n, shift)
        ret.c = -1.0 # indicating scheme not yet fitted
        return ret
    end
    
end

UnivariateBoxCoxScheme(; n=171, shift=false) = UnivariateBoxCoxScheme(n, shift)


function show(stream::IO, s::UnivariateBoxCoxScheme)
    if s.c >= 0
        print(stream, "UnivariateBoxCoxScheme($(s.lambda), $(s.c))")
    else
        print(stream, "unfitted UnivariateBoxCoxScheme()")
    end
end

function fit!(s::UnivariateBoxCoxScheme, v::Vector; verbosity=1)
    s.c < 0 ||
        warn("Refitting a previously trained transformation scheme.")   
    s.lambda, s.c, _ = boxcox(v; n=s.n, shift=s.shift)
    return s
end

function UnivariateBoxCoxScheme(v::Vector; verbosity=1, args...)
    s = UnivariateBoxCoxScheme(; args...)
    return fit!(s, v; verbosity=verbosity)
end

function transform(s::UnivariateBoxCoxScheme, x::T) where T <: Real
    if s.c < 0
        throw(Base.error("Attempting to transform according to unfitted scheme."))
    end
    return boxcox(s.lambda, s.c, x)
end

function inverse_transform(s::UnivariateBoxCoxScheme, x::T) where  T <:Real
    if s.c < 0
        throw(Base.error("Attempting to transform according to unfitted scheme."))
    end
    if s.lambda == 0
        return exp(x) - s.c
    else
        return (s.lambda*x + 1)^(1/s.lambda) - s.c
    end
end

transform(s::UnivariateBoxCoxScheme, v::AbstractVector) = boxcox(s.lambda, s.c, v)

inverse_transform(s::UnivariateBoxCoxScheme,
                  v::AbstractVector) = [inverse_transform(s,y) for y in v]


## `BoxCoxScheme` 

"""
## `mutable struct BoxCoxScheme`

Type for encoding Box-Cox transformations to each ordinal fields of a
`DataFrame` object.

### Method calls 

To calculate the compute Box-Cox transformation schemes for a data frame `X`:

    julia> s = BoxCoxScheme(X)    
    julia> XX = transform(s, Y) # transform data frame `Y` according to the scheme `s`
    
### Keyword arguments

Calls to the first method above may be issued with the following keyword arguments:

- `shift=true`: allow data shift in case of fields taking zero values
(otherwise no transformation will be applied).

- `n=171`: number of values of exponent `lambda` to try during optimization.

## See also

`UnivariateBoxCoxScheme`: The single variable version of the scheme
implemented by `BoxCoxScheme`.

"""

mutable struct BoxCoxScheme <: Scheme

    # hyperparameters:
    n::Int                     # number of values considered in exponent optimizations
    shift::Bool                # whether or not to shift features taking zero as value
    features::Vector{Symbol}   # features to attempt fitting a
                               # transformation (empty means all)
    
    # post-fit parameters:
    schemes::Vector{UnivariateBoxCoxScheme}
    feature_is_transformed::Vector{Bool} # keep track of which features are transformed

    fitted::Bool
    
    function BoxCoxScheme(n::Int, shift::Bool, features::Vector{Symbol})
        ret = new(n, shift, features)
        ret.fitted = false
        return ret
    end
    
end

BoxCoxScheme(; n=171, shift = false, features=Symbol[]) = BoxCoxScheme(n, shift, features)

function BoxCoxScheme(X; verbosity=1, args...)
    s =  BoxCoxScheme(; args...)
    fit!(s, X, verbosity=verbosity) # defined below
    return s
end



function fit!(s::BoxCoxScheme, X::AbstractDataFrame; verbosity=1)

    !s.fitted || warn("Refitting a previously trained transformation
     scheme.")

    # determine indices of features to be transformed
    features_to_try = (isempty(s.features) ? names(X) : s.features)
    s.feature_is_transformed = Array{Bool}(size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: Real && minimum(X[j]) >= 0
            s.feature_is_transformed[j] = true
        else
            s.feature_is_transformed[j] = false
        end
    end

    # fit each of those features with best Box Cox transformation
    s.schemes = Array{UnivariateBoxCoxScheme}(size(X, 2))
    verbosity < 1 ||
        println("Box-Cox transformations: ")
    for j in 1:size(X,2)
        if s.feature_is_transformed[j]
            if minimum(X[j]) == 0 && !s.shift
                verbosity < 1 ||
                    println("  :$(names(X)[j])    "*
                            "(*not* transformed, contains zero values)")
                s.feature_is_transformed[j] = false
                s.schemes[j] = UnivariateBoxCoxScheme()
            else
                n_values = length(unique(X[j]))
                if n_values < N_VALUES_THRESH
                    verbosity < 1 ||
                        println("  :$(names(X)[j])    "*
                                "(*not* transformed, less than $N_VALUES_THRESH values)")
                    s.feature_is_transformed[j] = false
                    s.schemes[j] = UnivariateBoxCoxScheme()
                else                    
                    uscheme = UnivariateBoxCoxScheme(collect(X[j]); shift=s.shift, n=s.n)
                    if uscheme.lambda in [-0.4, 3]
                        verbosity < 1 ||
                            println("  :$(names(X)[j])    "*
                                    "(*not* transformed, lambda too extreme)")
                        s.feature_is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    elseif uscheme.lambda == 1.0
                        verbosity < 1 ||
                            println("  :$(names(X)[j])    "*
                                    "(*not* transformed, not skewed)")
                        s.feature_is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    else
                        s.schemes[j] = uscheme
                        verbosity <1 ||
                            println("  :$(names(X)[j])    "*
                                    "lambda=$(s.schemes[j].lambda)  "*
                                    "shift=$(s.schemes[j].c)")
                    end
                end
            end
        else
            s.schemes[j] = UnivariateBoxCoxScheme()
        end
    end

    if !s.shift && verbosity < 1
        info("To transform non-negative features with zero values use shift=true.")
    end
    
    s.fitted = true
    
    return s
end

function transform(s::BoxCoxScheme, X::AbstractDataFrame)
    Xnew = copy(X)
    for j in 1:size(X, 2)
        if s.feature_is_transformed[j]
            try
                Xnew[j] = transform(s.schemes[j], collect(X[j]))
            catch DomainError
                warn("Data outside of the domain of the fitted Box-Cox"*
                      " transformation scheme encountered in feature "*
                      "$(names(df)[j]). Transformed to zero.")
            end
        end
    end
    return Xnew
end


end # module
