using Koala
using KoalaTransforms
using Base.Test

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

X, y = load_ames();
train, test = split(eachindex(y), 0.9);

transformer = ToIntTransformer(sorted=true)
transformerM = Machine(transformer, X[:Neighborhood])
v = transform(transformerM, X[test,:Neighborhood])
@test X[test, :Neighborhood] == inverse_transform(transformerM, v)

transformer.map_unseen_to_minus_one = true
transformerM = Machine(transformer, [1,2,3,4])
@test transform(transformerM, 5) == -1

transformer = DataFrameToArrayTransformer(boxcox=true)
transformerM = Machine(transformer, X)

transformer = RegressionTargetTransformer(standardize=false, boxcox=true)
transformerM = Machine(transformer, y)

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

t = MakeCategoricalsIntTransformer(initial_label=0, sorted=true)
tM = Machine(t, X)
transform(tM, X)

