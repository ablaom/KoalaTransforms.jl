module KoalaTransforms

# Univariate:
export ChainTransformer
export ToIntTransformer
export UnivariateStandardizer, Standardizer
export OneHotEncoder
export UnivariateBoxCoxTransformer, BoxCoxTransformer
export DataFrameToArrayTransformer, RegressionTargetTransformer
export MakeCategoricalsIntTransformer
export DataFrameToStandardizedArrayTransformer
export IntegerToInt64Transformer
export UnivariateDiscretizer, Discretizer

# extended:
export transform, inverse_transform, fit # from `Koala`

# transformers rexported from Koala:
export FeatureSelector, IdentityTransformer

# for use in this module:
import Koala: BaseType, params, type_parameters
import Koala: TransformerMachine, Machine
import DataFrames: names, AbstractDataFrame, DataFrame, eltypes
import Distributions

# to be extended:
import Koala: Transformer, transform, fit, inverse_transform
import Base: show, showall

# so above export from Koala works:
import Koala: EmptyTransformer, FeatureSelector
import Koala: IdentityTransformer

# development only:
# using ADBUtilities

# constants:
const N_VALUES_THRESH = 16 # for BoxCoxTransformation


## COMPOSITE TRANSFORMERS

mutable struct ChainTransformer <: Transformer
    chain::Vector{Transformer} # in order applied!
end

# constructor is given individual transforms in order opposite to that
# in which they will be applied:
ChainTransformer(transformers...) = ChainTransformer(reverse(Transformer[transformers...]))

function fit(transformer::ChainTransformer, X::Union{AbstractVector, AbstractDataFrame},
             parallel, verbosity)
    schemes = Array{Any}(length(transformer.chain))

    !isempty(transformer.chain) || error("Empty ChainTransformer.")

    # fit first transformer:
    schemes[1] = fit(transformer.chain[1], X, parallel, verbosity)

    # fit the rest:
    for i in 2:length(transformer.chain)
        X = transform(transformer.chain[i - 1], schemes[i - 1], X)
        schemes[i] = fit(transformer.chain[i], X, parallel, verbosity)
    end

    return schemes
end

function transform(transformer::ChainTransformer, schemes, 
                   X::Union{AbstractVector, AbstractDataFrame})
    for i in eachindex(schemes)
        X = transform(transformer.chain[i], schemes[i], X)
    end
    return X
end

function inverse_transform(transformer::ChainTransformer, schemes, 
                   X::Union{AbstractVector, AbstractDataFrame})
    for i in reverse(eachindex(schemes))
        X = inverse_transform(transformer.chain[i], schemes[i], X)
    end
    return X
end
    
function Base.getindex(machine::TransformerMachine{ChainTransformer}, i::Int)
    return TransformerMachine(machine.transformer.chain[end - i + 1],
                              machine.scheme[end - i + 1], :)
end


## FOR RELABELLING BY CONSECUTIVE INTEGERS STARTING AT 1

mutable struct ToIntTransformer <: Transformer
    sorted::Bool
    initial_label::Int # ususally 0 or 1
    map_unseen_to_minus_one::Bool # unseen inputs are transformed to -1
end

ToIntTransformer(; sorted=false, initial_label=1,
                 map_unseen_to_minus_one=false) =
                     ToIntTransformer(sorted, initial_label,
                                      map_unseen_to_minus_one)

struct ToIntScheme{T} <: BaseType
    n_levels::Int
    int_given_T::Dict{T, Int}
    T_given_int::Dict{Int, T}
end

# null scheme constructor:
ToIntScheme(S::Type{T}) where T = ToIntScheme{T}(0, Dict{T, Int}(), Dict{Int, T}())

function fit(transformer::ToIntTransformer, v::AbstractVector{T},
             parallel, verbosity) where T
    int_given_T = Dict{T, Int}()
    T_given_int = Dict{Int, T}()
    vals = collect(Set(v)) 
    if transformer.sorted
        sort!(vals)
    end
    n_levels = length(vals)
    if n_levels > 2^62 - 1
        error("Cannot encode with integers a vector "*
                         "having more than $(2^62 - 1) values.")
    end
    i = transformer.initial_label
    for c in vals
        int_given_T[c] = i
        T_given_int[i] = c
        i = i + 1
    end
    return ToIntScheme{T}(n_levels, int_given_T, T_given_int)
end

# scalar case:
function transform(transformer::ToIntTransformer, scheme::ToIntScheme{T}, x::T) where T
    ret = 0 # otherwise ret below stays in local scope
    try 
        ret = scheme.int_given_T[x]
    catch exception
        if isa(exception, KeyError)
            if transformer.map_unseen_to_minus_one 
                ret = -1
            else
                throw(exception)
            end
        end 
    end
    return ret
end 
inverse_transform(transformer::ToIntTransformer, scheme, y::Int) = scheme.T_given_int[y]

# vector case:
function transform(transformer::ToIntTransformer, scheme::ToIntScheme{T},
                   v::AbstractVector{T}) where T
    return Int[transform(transformer, scheme, x) for x in v]
end
inverse_transform(transformer::ToIntTransformer, scheme::ToIntScheme{T},
                  w::AbstractVector{Int}) where T = T[scheme.T_given_int[y] for y in w]


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

# null scheme:
StandardizerScheme() = StandardizerScheme(zeros(0,0), Symbol[], Bool[])

function fit(transformer::Standardizer, X, parallel, verbosity)
    
    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? names(X) : transformer.features)
    is_transformed = Array{Bool}(size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: AbstractFloat
            is_transformed[j] = true
        else
            is_transformed[j] = false
        end
    end

    # fit each of those features
    schemes = Matrix{Float64}(2, size(X, 2))
    verbosity < 1 || info("Features standarized: ")
    univ_transformer = UnivariateStandardizer()
    for j in 1:size(X, 2)
        if is_transformed[j]
            schemes[:,j] = [fit(univ_transformer, collect(X[j]), true, verbosity - 1)...]
            verbosity < 1 ||
                info("  :$(names(X)[j])    mu=$(schemes[1,j])  sigma=$(schemes[2,j])")
        else
            schemes[:,j] = Float64[0.0, 0.0]
        end
    end

    return StandardizerScheme(schemes, names(X), is_transformed)
    
end

function transform(transformer::Standardizer, scheme, X)

    names(X) == scheme.features ||
        error("Attempting to transform data frame with incompatible feature labels.")

    Xnew = X[1:end,:] # make a copy of X, working even for `SubDataFrames`
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

"""
    OneHotEncoder(drop_last=false)

Returns a transformer for one-hot encoding the categorical features of
an `AbstractDataFrame` object. Here "categorical" refers to any
feature whose eltype is not a subtype of `AbstractFloat`. All eltypes
must admit a `<` and `string` method.

"""
mutable struct OneHotEncoder <: Transformer
    drop_last::Bool
end

# lazy keyword constructor:
OneHotEncoder(; drop_last::Bool=false) = OneHotEncoder(drop_last)

struct OneHotEncoderScheme <: BaseType
    features::Vector{Symbol}         # feature labels
    spawned_features::Vector{Symbol} # feature labels after one-hot encoding
    values_given_feature::Dict{Symbol,Vector{Any}}
end

function fit(transformer::OneHotEncoder, X::AbstractDataFrame, parallel, verbosity)

    features = names(X)
    values_given_feature = Dict{Symbol, Any}()

    verbosity < 1 || info("One-hot encoding categorical features.")
    
    for ft in features 
        if !(eltype(X[ft]) <: AbstractFloat)
            values_given_feature[ft] = sort!(unique(X[ft]))
            if transformer.drop_last
                values_given_feature[ft] = values_given_feature[ft][1:(end - 1)]
            end
            if verbosity > 1
                n_values = length(keys(values_given_feature[ft]))
                info("Spawned $n_values columns to one-hot encode $ft.")
            end
        end  
    end

    spawned_features = Symbol[]

    for ft in features
        if !(eltype(X[ft]) <: AbstractFloat)
            for value in values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in the (rare) case subft is not a new feature label:
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
        if !(eltype(X[ft]) <: AbstractFloat)
            for value in scheme.values_given_feature[ft]
                subft = Symbol(string(ft,"__",value))

                # in case subft is not a new feature name:
                while subft in scheme.features
                    subft = Symbol(string(subft,"_"))
                end

                subft_col = map(X[ft]) do x
                    x == value ? 1.0 : 0.0
                end
                Xout[subft] = convert(Array{Float64}, subft_col)
            end
        else
            Xout[ft] = X[ft]
        end
    end
    return Xout

end


## Univariate Box-Cox transformations

function standardize(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end
                   
function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)

    n  = length(v)
    v = standardize(convert(Vector{Float64}, v))

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

Transformer for Box-Cox transformations on each numerical feature of a
`DataFrame` object. Here "numerical" means of eltype `T <:
AbstractFloat`.

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

    verbosity < 1 || info("Computing Box-Cox transformations "*
                          "on numerical features.")
    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? names(X) : transformer.features)
    feature_is_transformed = Array{Bool}(size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <:
            AbstractFloat && minimum(X[j]) >= 0
            feature_is_transformed[j] = true
        else
            feature_is_transformed[j] = false
        end
    end

    # fit each of those features with best Box Cox transformation
    schemes = Array{Float64}(2, size(X,2))
    univ_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift,
                                               n=transformer.n)
    verbosity < 2 ||
        info("Box-Cox transformations: ")
    for j in 1:size(X,2)
        if feature_is_transformed[j]
            if minimum(X[j]) == 0 && !transformer.shift
                verbosity < 2 ||
                    info("  :$(names(X)[j])    "*
                            "(*not* transformed, contains zero values)")
                feature_is_transformed[j] = false
                schemes[:,j] = [0.0, 0.0]
            else
                n_values = length(unique(X[j]))
                if n_values < N_VALUES_THRESH
                    verbosity < 2 ||
                        info("  :$(names(X)[j])    "*
                                "(*not* transformed, less than $N_VALUES_THRESH values)")
                    feature_is_transformed[j] = false
                    schemes[:,j] = [0.0, 0.0]
                else
                    lambda, c = fit(univ_transformer, collect(X[j]), true, verbosity-1)
                    if lambda in [-0.4, 3]
                        verbosity < 2 ||
                            info("  :$(names(X)[j])    "*
                                    "(*not* transformed, lambda too extreme)")
                        feature_is_transformed[j] = false
                        schemes[:,j] = [0.0, 0.0]
                    elseif lambda == 1.0
                        verbosity < 2 ||
                            info("  :$(names(X)[j])    "*
                                    "(*not* transformed, not skewed)")
                        feature_is_transformed[j] = false
                        schemes[:,j] = [0.0, 0.0]
                    else
                        schemes[:,j] = [lambda, c]
                        verbosity <1 ||
                            info("  :$(names(X)[j])    lambda=$lambda  "*
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

    Xnew = X[1:end,:] # make a copy of X
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

## General purpose transformer for supervised learning algorithms
## needing floating point matrix input.

"""
    DataFrameToArrayTransformer(boxcox=false, shift=true, 
    standardize=false, drop_last=false)

Use this transformer to prepare dataframe inputs for use with
supervised learning algorithms requiring `Matrix{T}` inputs, for
`T<:AbstractFloat`. Here `T` is forced to be `Float64`. Includes one-hot
encoding of categoricals (any feature not of eltype T<:AbstractFloat) and an
option for Box-Cox transformations. Note that all eltypes must admit
`<` and `string` methods.

### Keyword arguments:

- `boxcox`:  whether to apply Box-Cox transformations to the columns

- `shift`: do we shift away from zero in Box-Cox transformations?

- `standardize`: whether to standardize the columns

- `drop_last`: do we drop the last slot in the one-hot-encoding?

"""
mutable struct DataFrameToArrayTransformer <: Transformer
    boxcox::Bool 
    shift::Bool
    standardize::Bool
    drop_last::Bool 
end

# lazy keyword constructors:
DataFrameToArrayTransformer(; boxcox=false, shift=true, standardize=false,
                            drop_last=false) =
    DataFrameToArrayTransformer(boxcox, shift, standardize, drop_last)

struct Scheme_X <: BaseType
    boxcox::BoxCoxTransformerScheme
    stand::StandardizerScheme
    hot::OneHotEncoderScheme
    features::Vector{Symbol}
    spawned_features::Vector{Symbol} # ie after one-hot encoding
end

function fit(transformer::DataFrameToArrayTransformer, X::AbstractDataFrame, parallel, verbosity)

    features = names(X)
    
    # fit Box-Cox transformation to numerical features:
    if transformer.boxcox
        verbosity < 1 || info("Determining Box-Cox transformation parameters.")
        boxcox_transformer = BoxCoxTransformer(shift=transformer.shift)
        boxcox = fit(boxcox_transformer, X, true, verbosity - 1)
        X = transform(boxcox_transformer, boxcox, X)
    else
        boxcox = BoxCoxTransformerScheme() # null scheme
    end

    if transformer.standardize
        verbosity < 1 || info("Determining standardization parameters.")
        standardizer = Standardizer()
        stand = fit(standardizer, X, true, verbosity - 1)
        X = transform(standardizer, stand, X)
    else
        stand = StandardizerScheme() # null scheme
    end
    
    verbosity < 1 || info("Determining one-hot encodings for data frame categoricals.")
    hot_transformer = OneHotEncoder(drop_last=transformer.drop_last)
    hot =  fit(hot_transformer, X, true, verbosity - 1) 
    spawned_features = hot.spawned_features 

    return Scheme_X(boxcox, stand, hot, features, spawned_features)

end

function transform(transformer::DataFrameToArrayTransformer, scheme_X, X::AbstractDataFrame)
    issubset(Set(scheme_X.features), Set(names(X))) ||
        error("DataFrame feature incompatibility encountered.")
    X = X[scheme_X.features]

    if transformer.boxcox
        boxcox_transformer = BoxCoxTransformer(shift=transformer.shift)
        X = transform(boxcox_transformer, scheme_X.boxcox, X)
    end

    if transformer.standardize
        X = transform(Standardizer(), scheme_X.stand, X)
    end
    
    hot_transformer = OneHotEncoder(drop_last=transformer.drop_last)
    X = transform(hot_transformer, scheme_X.hot, X)
    return convert(Array{Float64}, X)
end

"""
## `RegressionTargetTransformer(boxcox=false, shift=true, standardize=true)::Transformer`

A general purpose transformer for target variables in regression
problems. Standardizes by default, and includes Box-Cox
transformations as an option.

### Keyword arguments:

- `boxcox`:  whether to apply Box-Cox transformations to the input patterns

- `shift`: do we shift away from zero in Box-Cox transformations?

- `standardize`: do we standardize (after any Box-Cox transformations)?

"""
mutable struct RegressionTargetTransformer <: Transformer
    boxcox::Bool # do we apply Box-Cox transforms to target (before any standarization)?
    shift::Bool # do we shift away from zero in Box-Cox transformations?
    standardize::Bool # do we standardize targets?
end

# lazy keyword constructors:
RegressionTargetTransformer(; boxcox=false, shift=true, standardize=true) =
    RegressionTargetTransformer(boxcox, shift, standardize)

struct Scheme_y <: BaseType
    boxcox::Tuple{Float64,Float64}
    standard::Tuple{Float64,Float64}
end

function fit(transformer::RegressionTargetTransformer, y, parallel, verbosity)

    if transformer.boxcox
        verbosity < 1 || info("Computing Box-Cox transformations for target.")
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        boxcox = fit(boxcox_transformer, y, true, verbosity - 1)
        y = transform(boxcox_transformer, boxcox, y)
    else
        boxcox = (0.0, 0.0) # null scheme
    end
    if transformer.standardize
        verbosity < 1 || info("Computing target standardization.")
        standard_transformer = UnivariateStandardizer()
        standard = fit(standard_transformer, y, true, verbosity - 1)
    else
        standard = (0.0, 1.0) # null scheme
    end
    return Scheme_y(boxcox, standard)
end 

function transform(transformer::RegressionTargetTransformer, scheme_y, y)
    if transformer.boxcox
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        y = transform(boxcox_transformer, scheme_y.boxcox, y)
    end
    if transformer.standardize
        standard_transformer = UnivariateStandardizer()
        y = transform(standard_transformer, scheme_y.standard, y)
    end 
    return y
end 

function inverse_transform(transformer::RegressionTargetTransformer, scheme_y, y)
    if transformer.standardize
        standard_transformer = UnivariateStandardizer()
        y = inverse_transform(standard_transformer, scheme_y.standard, y)
    end
    if transformer.boxcox
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        y = inverse_transform(boxcox_transformer, scheme_y.boxcox, y)
    end
    return y
end


## TRANSFORMER TO CONVERT CATEGORICALS TO INTEGER-VALUED COLUMNS

mutable struct MakeCategoricalsIntTransformer <: Transformer
    initial_label::Int
    sorted::Bool
end

MakeCategoricalsIntTransformer(; initial_label=1, sorted=false) = MakeCategoricalsIntTransformer(initial_label, sorted)

struct MakeCategoricalsIntScheme <: BaseType
    categorical_features::Vector{Symbol}
    schemes::Vector{ToIntScheme}
    to_int_transformer::ToIntTransformer
end

function fit(transformer::MakeCategoricalsIntTransformer, X::AbstractDataFrame, parallel, verbosity)
    categorical_features = Symbol[]
    schemes = ToIntScheme[]
    to_int_transformer = ToIntTransformer(sorted=transformer.sorted,
                                          initial_label=transformer.initial_label)
    types = eltypes(X)
    features = names(X)
    for j in eachindex(types)
        if !(types[j] <: AbstractFloat)
            push!(categorical_features, features[j])
            push!(schemes, fit(to_int_transformer, X[j], parallel, verbosity))
        end
    end
    verbosity < 1 || info("Input features treated as categorical: $categorical_features")
    return MakeCategoricalsIntScheme(categorical_features, schemes, to_int_transformer)
end

function transform(transformer::MakeCategoricalsIntTransformer, scheme_X, X::AbstractDataFrame)
    Xt = X[1:end,:] # make a copy of X, working even for `SubDataFrame`s
    for j in eachindex(scheme_X.categorical_features)
        ftr = scheme_X.categorical_features[j]
        Xt[ftr] = transform(scheme_X.to_int_transformer, scheme_X.schemes[j], X[ftr])
    end
    return Xt
end


## DISCRETIZATION OF CONTINUOUS VARIABLES

"""
    UnivariateDiscretizer(n_classes=512)

Returns a `Koala.Transformer` for discretizing vectors of `Real`
eltype, where `n_classes` describes the resolution of the
discretization. Transformed vectors are of eltype `Int46`. The
transformation is chosen so that the vector on which the transformer
is fit has, in transformed form, an approximately uniform distribution
of values.

### Example

    using Koala
    using KoalaTransforms
    t = UnivariateDiscretizer(n_classes=10)
    v = randn(1000)
    tM = Machine(t, v)   # fit the transformer on `v`
    w = transform(tM, v) # transform `v` according to `tM`
    StatsBase.countmap(w)

"""
mutable struct UnivariateDiscretizer <: Transformer
    n_classes::Int
end

# lazy keyword constructor:
UnivariateDiscretizer(; n_classes=512) = UnivariateDiscretizer(n_classes)

struct UnivariateDiscretizerScheme <: BaseType
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
end

function fit(transformer::UnivariateDiscretizer, v, parallel, verbosity)
    n_classes = transformer.n_classes
    quantiles = quantile(v, Array(linspace(0,1,2*n_classes+1)))  
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles
    
    # odd_quantiles for transforming, even_quantiles used for inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    return UnivariateDiscretizerScheme(odd_quantiles, even_quantiles)
end

# transforming scalars:
function transform(transformer::UnivariateDiscretizer, scheme, r::Real)
    k = 1
    for level in scheme.odd_quantiles
        if r > level
            k = k + 1
        end
    end
    return k
end

# transforming vectors:
function transform(transformer::UnivariateDiscretizer, scheme,
                   v::AbstractVector{T}) where T<:Real
    return [transform(transformer, scheme, r) for r in v]
end

# scalars:
function inverse_transform(transformer::UnivariateDiscretizer, scheme, k::Int)
    n_classes = length(scheme.even_quantiles)
    if k < 1
        return scheme.even_quantiles[1]
    elseif k > n_classes
        return scheme.even_quantiles[n_classes]
    end
    return scheme.even_quantiles[k]
end

# vectors:
function inverse_transform(transformer::UnivariateDiscretizer, scheme,
                           w::AbstractVector{T}) where T<:Integer
    return [inverse_transform(transformer, scheme, k) for k in w]
end


## CONVERTING ANY INTEGER TYPE TO INT16

mutable struct IntegerToInt64Transformer <: Transformer
end

fit(transformer::IntegerToInt64Transformer, X, parallel, verbosity) = nothing

transform(transformer::IntegerToInt64Transformer, scheme, X) = Int64[X...]


## DISCRETIZING ALL COLUMNS OF A DATAFRAME

struct NominalOrdinalIntArray <: BaseType
    A::Matrix{Int}
    features::Vector{Symbol}
    is_ordinal::Vector{Bool}
end

function showall(stream::IO, X::NominalOrdinalIntArray)
    features_plus = Array{String}(length(X.features))
    for j in eachindex(X.features)
        kind = X.is_ordinal[j] ? "ordinal" : "nominal"
        features_plus[j] = string(X.features[j], " ($kind)  ")
    end
    for str in features_plus
        print(stream, str)
    end
    println(stream)
    show(stream, MIME("text/plain"), X.A)
end

"""
    Discretizer(n_classes=512)

Returns a `Koala.Transformer` for discretizing the columns of a
`DataFrame`. Transformed objects are of type `NominalOrdinalIntArray`,
a lightweight wrapper for an `Int64` matrix whose fields `A`,
`features` and `is_ordinal` store, respectively, the integer matrix, a
vector of feature names, and a vector recording which features are
ordinal.

Columns are discretized according to the following conventions: If the
column eltype is a subtype of `AbstractFloat`, the column is
discretized using a `UnivariateDiscretizer` transformer. All remaining
eltypes are treated as nominal, with elements relabeled with integers,
starting with 1; nominal values not encountered during fitting the
transformer will throw a `KeyError` if there is an attempt to
transform them according to the fitted scheme.

"""
mutable struct Discretizer <: Transformer
    n_classes::Int
    features::Vector{Symbol} # features to be used; empty means all
end

# lazy keyword constructor:
Discretizer(; n_classes=512, features=Symbol[]) = Discretizer(n_classes, features)

struct DiscretizerScheme
    features::Vector{Symbol} # features actually used
    transformer_machines::Vector{TransformerMachine}
    is_ordinal::Vector{Bool}    
end

function fit(transformer::Discretizer, X::AbstractDataFrame, parallel, verbosity)

    to_int = ToIntTransformer(sorted=true)
    discrete = UnivariateDiscretizer()

    if isempty(transformer.features)
        features = names(X)
    else
        features = transformer.features
    end

    transformer_machines = Array{TransformerMachine}(length(features))
    is_ordinal = Array{Int}(length(features))
    
    j = 1
    for ftr in features
        if eltype(X[ftr]) <: AbstractFloat
            is_ordinal[j] = true
            transformer_machines[j] = Machine(discrete, X[ftr];
                parallel=false, verbosity=verbosity - 1)
        else
            is_ordinal[j] = false
            transformer_machines[j] = Machine(to_int, X[ftr];
            parallel=false, verbosity=verbosity - 1)
        end
        j +=1
    end
    
    return DiscretizerScheme(features, transformer_machines, is_ordinal)
    
end

function transform(transformer::Discretizer, scheme, X)

    issubset(Set(scheme.features), Set(names(X))) ||
        error("Provided DataFrame with incompatible features.")

    n_features = length(scheme.features)
    A = Array{Int}(size(X, 1), n_features)

    for j in 1:n_features
        A[:,j] = transform(scheme.transformer_machines[j],
                           X[:,scheme.features[j]])
    end

    return NominalOrdinalIntArray(A, scheme.features, scheme.is_ordinal)

end
            

end # module


