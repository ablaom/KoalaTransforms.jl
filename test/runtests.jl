using Koala
using KoalaTransforms
using Base.Test

const EPS = eps(Float64)

## `UnivariateStandardizer`

t = UnivariateStandardizer()
showall(t)
tM = TransformerMachine(t, [0, 2, 4])
showall(tM)

@test round.(Int, transform(tM, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(tM, [-1, 1, 3])) == [0, 4, 8] 

# create skewed non-negative vector with a zero value:
v = abs.(randn(1000))
v = v - minimum(v)
KoalaTransforms.normality(v)

t = UnivariateBoxCoxTransformer(shift=true)
showall(t)
tM = TransformerMachine(t, v)
@test sum(abs.(v - inverse_transform(tM,transform(tM, v)))) <= 5000*EPS

X, y = load_ames();

t = OneHotEncoder(drop_last=true)
tM = TransformerMachine(t, X)
Xt = transform(tM, X)

t = BoxCoxTransformer(shift=true)
tM = TransformerMachine(t, X)
transform(tM, X)

t = BoxCoxTransformer(shift=true, features=[:GrLivArea])
tM = TransformerMachine(t, X)
Xt = transform(tM, X)



