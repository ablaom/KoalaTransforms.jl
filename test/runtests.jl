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

transformer = DataFrameToArrayTransformer(features=[:OverallQual, :GrLivArea])
transformerM = Machine(transformer, X)
@test transform(transformerM, X) == Array(X[[:OverallQual, :GrLivArea]])

## Standardizer for data frames

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



