# KoalaTransforms

## Quickstart

The objects of a concrete subtype of `Transformer` store the
data-independent hyperparameters describing some transformation. These
objects are called *transformers*. For example, a
`UnivariateDiscretizer <: Transfomer` object stores the number of
classes (levels) to discretize some vector:

````
julia> using KoalaTransforms
julia> t = UnivariateDiscretizer()
julia> showall(t)
UnivariateDiscretizer@...313

key                     | value
------------------------|------------------------
n_classes               |512

julia> t.n_classes = 1000

````

The objects that do the actual transforming are `TransformerMachine`
objects. These are instantiated by providing the transformer with
data:
    
````
julia> v = rand(100);
julia> tM = Machine(t, v)
TransformerMachine{UnivariateDiscretizer}@...498
````

We can now transform any scalar:

````
julia> transform(tM, 0.51)
490
````

Or vector:

````
julia> w = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
julia> w_discrete = transform(tM, w)
6-element Array{Int64,1}:
    1
  149
  390
  565
  761
 1000
````

Transformations can be inverted if `inverse_transform` is implemented:

````
julia> inverse_transform(tM, w_discrete)
6-element Array{Float64,1}:
 0.0174372
 0.199809 
 0.40006  
 0.59997  
 0.799158 
 0.998602 
````

To see all the implemented transformers, use
`subtypes(Koala.Transformer)` and query each type's docstring for
details. For a template for implementing your own Koala transformers,
see
[here](https://github.com/ablaom/KoalaLow.jl/blob/master/src/TransformerTemplate.jl).
