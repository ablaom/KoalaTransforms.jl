using Koala
using KoalaTransforms
using Base.Test
using DataFrames

const EPS = eps(Float64)

## `UnivariateStandardizer`

t = UnivariateStandardizer()
showall(t)
tM = Machine(t, [0, 2, 4])
showall(tM)

@test round.(Int, transform(tM, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(tM, [-1, 1, 3])) == [0, 4, 8] 

# create skewed non-negative vector with a zero value:
v = abs.(randn(1000))
v = v - minimum(v)
KoalaTransforms.normality(v)

t = UnivariateBoxCoxTransformer(shift=true)
showall(t)
tM = Machine(t, v)
@test sum(abs.(v - inverse_transform(tM,transform(tM, v)))) <= 5000*EPS

const X, y = load_ames();
const train, test = split(eachindex(y), 0.9);

# introduce a field of type `Char`:
X[:OverallQual] = map(Char, X[:OverallQual])

transformer = ToIntTransformer(sorted=true)
transformerM = Machine(transformer, X[:Neighborhood])
v = transform(transformerM, X[test,:Neighborhood])
@test X[test, :Neighborhood] == inverse_transform(transformerM, v)

transformer.map_unseen_to_minus_one = true
transformerM = Machine(transformer, [1,2,3,4])
@test transform(transformerM, 5) == -1
@test transform(transformerM, [5,1])[1] == -1 

transformer = DataFrameToArrayTransformer(boxcox=true, standardize=true)
transformerM = Machine(transformer, X)
transform(transformerM, X)

transformer = RegressionTargetTransformer(standardize=false, boxcox=true)
transformerM = Machine(transformer, y)
transform(transformerM, y)

t = Standardizer()
tM = Machine(t, X)
transform(tM, X)

t = Standardizer(features=[:GrLivArea])
tM = Machine(t, X)
transform(tM, X)

t = OneHotEncoder(drop_last=true)
tM = Machine(t, X)
Xt = transform(tM, X)

t = BoxCoxTransformer(shift=true)
tM = Machine(t, X)
transform(tM, X)

t = BoxCoxTransformer(shift=true, features=[:GrLivArea])
tM = Machine(t, X)
Xt = transform(tM, X)

t = DataFrameToArrayTransformer()
tM = Machine(t, X[[:OverallQual]], verbosity=3)
Xt = transform(tM, X)
@test size(Xt, 2) == length(unique(X[:OverallQual]))

t = MakeCategoricalsIntTransformer(initial_label=0, sorted=true)
tM = Machine(t, X)
transform(tM, X)

# IntegerToInt64:
t = IntegerToInt64Transformer()
v = UInt8[4, 5, 2, 3, 1]
tM = Machine(t, v)
@test transform(tM, UInt8[3, 7, 8]) == [3, 7, 8]

# Univariate discretization:
v = randn(10000)
t = UnivariateDiscretizer(n_classes=100)
tM = Machine(t, v)
w = transform(tM, v)
bad_values = filter(v - inverse_transform(tM, w)) do x
    abs(x) > 0.05
end
@test length(bad_values)/length(v) < 0.06

# Discretization of DataFrame:
X = DataFrame(n1=["a", "b", "a"], n2=["g", "g", "g"], n3=[7, 8, 9],
              n4 =UInt8[3,5,10],  o1=[4.5, 3.6, 4.0], )

t = Discretizer(features=[:o1, :n3, :n2, :n1])
tM = Machine(t, X)
Xt = transform(tM, X)
@test Xt.features == [:o1, :n3, :n2, :n1]
@test Xt.is_ordinal == [true, false, false, false]
@test Xt.A == [512 1 1 1; 1 2 1 2; 256 3 1 1]

