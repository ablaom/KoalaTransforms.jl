using Koala
using KoalaTransforms
using Base.Test

const EPS = eps(Float64)

## `UnivariateStandardizationScheme`

s = UnivariateStandardizationScheme()
showall(s)
fit!(s, [0,2,4])
showall(s)
@test round.(Int, transform(s, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(s, [-1, 1, 3])) == [0, 4, 8] 

s = UnivariateStandardizationScheme([0,2,4])
@test round(Int, transform(s, 4)) == 1
@test round(Int, inverse_transform(s, 1)) == 4

# create skewed non-negative vector with a zero value:
v = abs.(randn(1000))
v = v - minimum(v)

s = UnivariateBoxCoxScheme(v, shift=true)
@test sum(abs.(v - inverse_transform(s,transform(s, v)))) <= 5000*EPS

s2 = UnivariateBoxCoxScheme(shift=true)
fit!(s2, v)
@test transform(s2, v) == transform(s, v)
s3 = UnivariateBoxCoxScheme(shift=true)
@test fit_transform!(s3, v) == transform(s, v)
               
X, y = load_ames();

s = BoxCoxScheme(X, shift=true, verbosity=0)

s2 = BoxCoxScheme(shift=true)
fit!(s2, X)
@test transform(s2, X) == transform(s, X)

s3 = BoxCoxScheme(shift=true)
@test fit_transform!(s3, X) == transform(s, X)

s4 = BoxCoxScheme()
X = fit_transform!(s4, X, verbosity=0)

s = HotEncodingScheme(drop_last=true)
fit!(s, X)

Xt = transform(s, X)

s3 =HotEncodingScheme(drop_last=true)
@test Xt == fit_transform!(s3, X, verbosity=0)

s2 = HotEncodingScheme(X, drop_last=true)
@test transform(s, X) == transform(s2, X)

