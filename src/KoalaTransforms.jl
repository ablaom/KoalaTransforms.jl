module KoalaTransforms

# new:
export UnivariateStandardizer, Standardizer
export OneHotEncoder
export UnivariateBoxCoxTransformer, BoxCoxTransformer

# extended:
export transform, inverse_transform, fit # from `Koala`

# transformers rexported from Koala:
export FeatureTruncater, DataFrameToArrayTransformer, IdentityTransformer

# for use in this module:
import Koala: BaseType, params, type_parameters
import DataFrames: names, AbstractDataFrame, DataFrame
import Distributions

# to be extended:
import Koala: Transformer, transform, fit, inverse_transform
import Base: show, showall

# to be exported from Koala.jl:
import Koala: NullTransformer, FeatureTruncater
import Koala: IdentityTransformer

# constants:
const N_VALUES_THRESH = 16


## Converting DataFrames with ordinal features into arrays

mutable struct DataFrameToArrayTransformer <: Transformer
    features::Vector{Symbol} # empty means use all
end
DataFrameToArrayTransformer(;features=Symbol[]) = DataFrameToArrayTransformer(features)
function fit(transformer::DataFrameToArrayTransformer, X, parallel, verbosity)
    if isempty(transformer.features)
        return names(X)
    else
        return transformer.features
    end
end
function transform(transformer::DataFrameToArrayTransformer, scheme, X)
    issubset(Set(scheme), Set(names(X))) || throw(DomainError)
    return convert(Array{Float64}, X[scheme])
end 


## Univariate standardization 

struct UnivariateStandardizer <: Transformer end

function fit(transformer::UnivariateStandardizer, v::AbstractVector{T},
             parallel, verbosity) where T <: Real
    return  mean(v), std(v)
end

# for transforming single value:
function transform(transformer::UnivariateStandardizer, scheme, x::Real)
    mu, sigma = scheme
    return (x - mu)/sigma
end

# for transforming vector:
transform(transformer::UnivariateStandardizer, scheme,
          v::AbstractVector{T}) where T <: Real =
              [transform(transformer, scheme, x) for x in v]

# for single values:
function inverse_transform(transformer::UnivariateStandardizer, scheme, y::Real)
    mu, sigma = scheme
    return mu + y*sigma
end

# for vectors:
inverse_transform(transformer::UnivariateStandardizer, scheme,
                  w::AbstractVector{T}) where T <: Real =
                      [inverse_transform(transformer, scheme, y) for y in w]


## Standardization of ordinal features of a DataFrame

mutable struct Standardizer <: Transformer
    features::Vector{Symbol} # features to be standardized; empty means all
end

# lazy keyword constructor:
Standardizer(; features=Symbol[]) = Standardizer(features)

struct StandardizerScheme <: BaseType
    schemes::Matrix{Float64}
    features::Vector{Symbol} # all the feature labels of the data frame fitted
    is_transformed::Vector{Bool}
end

function fit(transformer::Standardizer, X, parallel, verbosity)
    
    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? names(X) : transformer.features)
    is_transformed = Array{Bool}(size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: Real
            is_transformed[j] = true
        else
            is_transformed[j] = false
        end
    end

    # fit each of those features
    schemes = Matrix{Float64}(2, size(X, 2))
    verbosity < 1 || println("Features standarized: ")
    univ_transformer = UnivariateStandardizer()
    for j in 1:size(X, 2)
        if is_transformed[j]
            schemes[:,j] = [fit(univ_transformer, collect(X[j]), true, verbosity - 1)...]
            verbosity < 1 ||
                println("  :$(names(X)[j])    mu=$(schemes[1,j])  sigma=$(schemes[2,j])")
        else
            schemes[:,j] = Float64[0.0, 0.0]
        end
    end

    return StandardizerScheme(schemes, names(X), is_transformed)
    
end

function transform(transformer::Standardizer, scheme, X)

    names(X) == scheme.features ||
        error("Attempting to transform data frame with incompatible feature labels.")

    Xnew = copy(X)
    univ_transformer = UnivariateStandardizer()
    for j in 1:size(X, 2)
        if scheme.is_transformed[j]
            # extract the (mu, sigma) pair:
            univ_scheme = (scheme.schemes[1,j], scheme.schemes[2,j])  
            Xnew[j] = transform(univ_transformer, univ_scheme, collect(X[j]))
        end
    end
    return Xnew

end    


## One-hot encoding

mutable struct OneHotEncoder <: Transformer
    drop_last::Bool
end

# lazy keyword constructor:
OneHotEncoder(; drop_last::Bool=false) = OneHotEncoder(drop_last)

struct OneHotEncoderScheme <: BaseType
    features::Vector{Symbol}         # feature labels
    spawned_features::Vector{Symbol} # feature labels after one-hot encoding
    values_given_feature::Dict{Symbol,Vector{String}}
end

function fit(transformer::OneHotEncoder, X::AbstractDataFrame, parallel, verbosity)

    features = names(X)
    values_given_feature = Dict{Symbol,Vector{String}}()
        
    for ft in features 
        if eltype(X[ft]) <: AbstractString
            values_given_feature[ft] = sort!(unique(X[ft]))
            if transformer.drop_last
                values_given_feature[ft] = values_given_feature[ft][1:(end - 1)]
            end
            if verbosity > 0
                n_values = length(keys(values_given_feature[ft]))
                println("Spawned $n_values columns to hot-encode $ft.")
            end
        elseif eltype(X[ft]) == Char
            warn("A feature of Char type has been encountered and "*
                 "is being ignored. To be hot-encoded "*
                 "it must first be converted to some AbstractString type.")
        end  
    end

    spawned_features = Symbol[]

    for ft in features
        if eltype(X[ft]) <: AbstractString
            for value in values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in (rare) case subft is not a new feature label:
                while subft in features
                    subft = Symbol(string(subft,"_"))
                end

                push!(spawned_features, subft)
            end
        else
            push!(spawned_features, ft)
        end
    end

    return OneHotEncoderScheme(features, spawned_features, values_given_feature)
    
end

function transform(transformer::OneHotEncoder, scheme, X::AbstractDataFrame)

    Set(names(X)) == Set(scheme.features) ||
        error("Attempting to transform DataFrame with incompatible feature labels.")
    
    Xout = DataFrame()
    for ft in scheme.features
        if eltype(X[ft]) <: AbstractString
            for value in scheme.values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in case subft is not a new feature name:
                while subft in scheme.features
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


## Univariate Box-Cox transformations

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

function boxcox(lambda, c, x::Real) 
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::AbstractVector{T}) where T <: Real =
    [boxcox(lambda, c, x) for x in v]    


"""
## `struct UnivariateBoxCoxTransformer`

A type for encoding a Box-Cox transformation of a single variable
taking non-negative values, with a possible preliminary shift. Such a
transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

###  Usage

    `trf = UnivariateBoxCoxTransformer(; n=171, shift=false)`

Returns transformer that on fitting to data (see below) will try `n`
different values of the Box-Cox exponent λ (between `-0.4` and `3`) to
find an optimal value. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift by `0.2` times the data mean. If there are no zero
values, then `s.c=0`.

See also `BoxCoxScheme` a transformer for selected ordinals in a DataFrame. 

"""
mutable struct UnivariateBoxCoxTransformer <: Transformer
    n::Int      # nbr values tried in optimizing exponent lambda
    shift::Bool # whether to shift data away from zero
end

# lazy keyword constructor:
UnivariateBoxCoxTransformer(; n=171, shift=false) = UnivariateBoxCoxTransformer(n, shift)

function fit(transformer::UnivariateBoxCoxTransformer, v::AbstractVector{T},
    parallel, verbosity) where T <: Real 

    m = minimum(v)
    m >= 0 || error("Cannot perform a Box-Cox transformation on negative data.")

    c = 0.0 # default
    if transformer.shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || error("Zero value encountered in data being Box-Cox transformed.\n"*
                        "Consider calling `fit!` with `shift=true`.")
    end
  
    lambdas = linspace(-0.4,3,transformer.n)
    scores = Float64[normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[indmax(scores)]

    return  lambda, c

end

# for X scalar or vector:
transform(transformer::UnivariateBoxCoxTransformer, scheme, X) =
    boxcox(scheme..., X) 

# scalar case:
function inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           scheme, x::Real)
    lambda, c = scheme
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           scheme, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, scheme, y) for y in w]
end


## `BoxCoxScheme` 

"""
## `struct BoxCoxTransformer`

Transformer for Box-Cox transformations on each ordinal feature of a
`DataFrame` object.

### Method calls 

To calculate the compute Box-Cox transformations of a DataFrame `X`
and transform a new DataFrame `Y` according to the same
transformations:

    julia> transformer = BoxCoxTransformer()    
    julia> transformerM = TransformerMachine(transformer, X)
    julia> transform(transformerM, Y)
    
### Transformer parameters

Calls to the first method above may be issued with the following
keyword arguments, with defaluts as indicated:

- `shift=true`: allow data shift in case of fields taking zero values
(otherwise no transformation will be applied).

- `n=171`: number of values of exponent `lambda` to try during optimization.

## See also

`UnivariateBoxCoxTransformer`: The single variable version of the same transformer.

"""
mutable struct BoxCoxTransformer <: Transformer
    n::Int                     # number of values considered in exponent optimizations
    shift::Bool                # whether or not to shift features taking zero as value
    features::Vector{Symbol}   # features to attempt fitting a
                               # transformation to (empty means all)
end

# lazy keyword constructor:
BoxCoxTransformer(; n=171, shift = false, features=Symbol[]) =
    BoxCoxTransformer(n, shift, features)

struct BoxCoxTransformerScheme <: BaseType
    schemes::Matrix{Float64} # each col is a [lambda, c]' pair; one col per feature
    features::Vector{Symbol} # all features in the dataframe that was fit
    feature_is_transformed::Vector{Bool} # keep track of which features are transformed
end

# null scheme:
BoxCoxTransformerScheme() = BoxCoxTransformerScheme(zeros(0,0),Symbol[],Bool[])

function fit(transformer::BoxCoxTransformer, X, parallel, verbosity)

    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? names(X) : transformer.features)
    feature_is_transformed = Array{Bool}(size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: Real && minimum(X[j]) >= 0
            feature_is_transformed[j] = true
        else
            feature_is_transformed[j] = false
        end
    end

    # fit each of those features with best Box Cox transformation
    schemes = Array{Float64}(2, size(X,2))
    univ_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift,
                                               n=transformer.n)
    verbosity < 1 ||
        println("Box-Cox transformations: ")
    for j in 1:size(X,2)
        if feature_is_transformed[j]
            if minimum(X[j]) == 0 && !transformer.shift
                verbosity < 1 ||
                    println("  :$(names(X)[j])    "*
                            "(*not* transformed, contains zero values)")
                feature_is_transformed[j] = false
                schemes[:,j] = [0.0, 0.0]
            else
                n_values = length(unique(X[j]))
                if n_values < N_VALUES_THRESH
                    verbosity < 1 ||
                        println("  :$(names(X)[j])    "*
                                "(*not* transformed, less than $N_VALUES_THRESH values)")
                    feature_is_transformed[j] = false
                    schemes[:,j] = [0.0, 0.0]
                else
                    lambda, c = fit(univ_transformer, collect(X[j]), true, verbosity-1)
                    if lambda in [-0.4, 3]
                        verbosity < 1 ||
                            println("  :$(names(X)[j])    "*
                                    "(*not* transformed, lambda too extreme)")
                        feature_is_transformed[j] = false
                        schemes[:,j] = [0.0, 0.0]
                    elseif lambda == 1.0
                        verbosity < 1 ||
                            println("  :$(names(X)[j])    "*
                                    "(*not* transformed, not skewed)")
                        feature_is_transformed[j] = false
                        schemes[:,j] = [0.0, 0.0]
                    else
                        schemes[:,j] = [lambda, c]
                        verbosity <1 ||
                            println("  :$(names(X)[j])    lambda=$lambda  "*
                                    "shift=$c")
                    end
                end
            end
        else
            schemes[:,j] = [0.0, 0.0]
        end
    end

    if !transformer.shift && verbosity < 1
        info("To transform non-negative features with zero values use shift=true.")
    end

    return BoxCoxTransformerScheme(schemes, names(X), feature_is_transformed)

end

function transform(transformer::BoxCoxTransformer, scheme, X::AbstractDataFrame)

    names(X) == scheme.features ||
        error("Attempting to transform a data frame with  "*
              "incompatible feature labels.")
    
    univ_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift,
                                               n=transformer.n)

    Xnew = copy(X)
    for j in 1:size(X, 2)
        if scheme.feature_is_transformed[j]
            try
                # extract the (lambda, c) pair:
                univ_scheme = (scheme.schemes[1,j], scheme.schemes[2,j])  

                Xnew[j] = transform(univ_transformer, univ_scheme, collect(X[j]))
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
