# KoalaTransforms

Common data transformations performed in machine learning, for use
with the [Koala](https://github.com/ablaom/Koala.jl) environment. 

## Installation

Refer to the [Koala](https://github.com/ablaom/Koala.jl) repository.

## Quickstart

The objects of a concrete subtype of `Transformer` store the
data-independent hyperparameters describing some transformation. These
objects are called *transformers*. For example, a
`UnivariateDiscretizer <: Transfomer` object stores the number of
classes (levels) to discretize some vector:

````
julia> using KoalaTransforms
julia> t = UnivariateDiscretizer()
UnivariateDiscretizer@...952: 
````

key or field            | value
------------------------|------------------------------------------------
n_classes               | 512

````
julia> t.n_classes = 1000
````

The objects that do the actual transforming are `TransformerMachine`
objects. These are instantiated by providing the transformer with
data:
    
````
julia> v = rand(100);
julia> tM = Machine(t, v)
TransformerMachine{UnivariateDiscretizer}@...930:
````

key or field            | value
------------------------|------------------------------------------------
scheme                  | *object of type KoalaTransforms.UnivariateDiscretizerScheme*
transformer             | UnivariateDiscretizer@...510

> transformer detail:
UnivariateDiscretizer@...510: 

key or field            | value
------------------------|------------------------------------------------
n_classes               | 1000

> scheme detail:
UnivariateDiscretizerScheme@...904: 

key or field            | value
------------------------|------------------------------------------------
even_quantiles          | [0.00333486, 0.00365538, 0.00397589, 0.0042964, 0.00461692, 0.00493743, 0.00525794, 0.00557846, 0.00589897, 0.00621948  …  0.983008, 0.984141, 0.985274, 0.986408, 0.987541, 0.988674, 0.989808, 0.990941, 0.992074, 0.993208]
odd_quantiles           | [0.00349512, 0.00381563, 0.00413615, 0.00445666, 0.00477717, 0.00509769, 0.0054182, 0.00573871, 0.00605923, 0.00637974  …  0.982441, 0.983574, 0.984708, 0.985841, 0.986974, 0.988108, 0.989241, 0.990374, 0.991508, 0.992641]

We can now transform any scalar:

````julia
julia> transform(tM, 0.51)
541
````

Or vector:

````
julia> w = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
julia> w_discrete = transform(tM, w)
6-element Array{Int64,1}:
    1
  167
  409
  607
  786
 1000
````

Transformations can be inverted if `inverse_transform` is implemented:

````
julia> inverse_transform(tM, w_discrete)
6-element Array{Float64,1}:
 0.003334863432704751
 0.1999601745313676
 0.4010114211975037
 0.5999763117413155
 0.7995624272453655
 0.9932075341405199
````

To see all the implemented transformers, use
`subtypes(Koala.Transformer)` and query each type's docstring for
details. For a template for implementing your own Koala transformers,
see
[here](https://github.com/ablaom/KoalaLow.jl/blob/master/src/TransformerTemplate.jl).
