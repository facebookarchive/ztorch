ZTorch - Complex number support for Torch
=========================================

To use this package:
```lua
require 'ztorch'
```
It introduces two new tensor-types: **`torch.ZFloatTensor`** and **`torch.ZDoubleTensor`**

A brief summary:

- ALL functions in the torch.Tensor API are supported for arbitrary dimensions (except those for which you wouldn't have a complex operation). This includes:
  - All BLAS and LAPACK operations supported on a Float tensor.
  - All comparison operators, +,-,*,/ overloading
  - Convolution functions: conv2, xcorr2, conv3, xcorr3 etc.
- For comparison functions like :lt() :gt() etc. and :sort(), the complex absolute value is used to compare two complex numbers.
- You can copy from a (Float/Double/Int/Byte/etc.)Tensor to a ZFloatTensor.
  - There are two semantics to keep in mind for copies from Real to Complex
  - If your ZFloatTensor and RealTensor are **both of size AxB** then the Real Tensor is copied into the real part and the imaginary part is 0.
    a=torch.ZFloatTensor(2,3):copy(torch.randn(2,3))
  - If your ZFloatTensor is of size **AxB** and RealTensor is of size **AxBx2** then the last dimension is treated as real/imaginary pairs.
    a=torch.ZFloatTensor(2,3):copy(torch.randn(2,3,2))
- torch.save and torch.load work out of the box.
- Random generators are not supported. You have to generate a random float tensor and copy over.
- Additional functions added are:
  - conj - complex conjugate
  - proj - projection of z onto the Riemann sphere (http://pubs.opengroup.org/onlinepubs/009695399/functions/cproj.html)
  - arg  - argument (also called phase angle) of z, with a branch cut along the negative real axis. (http://pubs.opengroup.org/onlinepubs/009695399/functions/carg.html)
  - re   - Returns the real part as a FloatTensor (they dont share storages)
  - im   - Returns the imag part as a FloatTensor (they dont share storages)

# Examples #
This section is divided into examples for complex numbers, and complex tensors.

## Complex numbers ##

###Defining numbers

```lua
  a = 3+4i
  b = 2i
```

###Mathematical operations
```lua
  a+b
  > 3+6i

  cx=require 'ztorch.complex'
  c = a+b
  cx.abs(c)
  > 6.7082039324994
```

NOTE: look at this subtle difference. Always create complex numbers properly bracketed, or put them into variables:
```lua
3+4i*3+5i
> 3+17i -- WRONG
(3+4i)*(3+5i)
> -11+27i -- CORRECT
a=3+4i
b=3+5i
a*b
> -11+27i -- CORRECT
```

The following operations are defined in ztorch.complex:
```
- sin, cos, tan, asinh, acosh, atanh, sinh, cosh, tanh, asin, acos, atan
- log, exp
- pow, sqrt
- conj, abs, arg
```

##Complex Tensors

One new tensor type is introduced called torch.ZFloatTensor.
You would use it just like any other tensor.

###Constructors
```
a = torch.ZFloatTensor(2,3)  -- complex tensor of dimensions 2, 3
```

###Random tensors

```lua
a:normal() -- fill the tensor with normally distributed values with zero-mean, std-1
a:normal(-1, 10) -- normally distributed, -1 mean, 10 std
```
Similarly, you have all the RNG specified for a usual regular tensor, such as:
```
uniform, logNormal, bernoulli, cauchy, geometric, exponential, random
```

###Copying from a Real Tensor

You can construct a complex tensor from a float tensor in two ways:
####1. Complex tensor from purely real tensor:
```lua
-- Copy from (Float/Double/Byte/Int/...)Tensor
b=torch.randn(6)
print(b)
>  -0.1834
>   2.1850
>   0.5873
>   1.0355
>   1.5442
>   0.5493
>  [torch.DoubleTensor of dimension 6]
a:copy(b)
> -0.1834+0.0000i  2.1850+0.0000i  0.5873+0.0000i
>  1.0355+0.0000i  1.5442+0.0000i  0.5493+0.0000i
> [torch.ZFloatTensor of dimension 2x3]
```

####2. Complex tensor from a real tensor with last-dimension of size 2.
In this case, we the last dimension consists of pairs of real/imaginary parts.

```lua
-- when the RealTensor is of size AxBx2 where the ZFloatTensor is of size AxB,
-- the last dimension of the real tensor is used as real+imag pairs
b=torch.randn(2,3,2)
print(b)
> (1,.,.) =
>   0.1556 -2.6688
>   1.5863 -0.2113
>   0.0980 -0.0120
>
> (2,.,.) =
>   0.6998  1.7506
>   1.1829  0.9513
>   0.0612  0.6046
> [torch.DoubleTensor of dimension 2x3x2]
a:copy(b)
>  0.1556-2.6688i  1.5863-0.2113i  0.0980-0.0120i
>  0.6998+1.7506i  1.1829+0.9513i  0.0612+0.6046i
> [torch.ZFloatTensor of dimension 2x3]
```

You can treat this as any other tensor, making operations such as:
```
:view, :reshape, :index, :select etc.
```


### Copying to a Real Tensor
You cannot directly copy a complex tensor to a real tensor, because such an equivalent operation does not exist mathematically.
However, you can for example get the real and imaginary components separately, or get the element-wise absolute value as a real tensor.
```lua
a = torch.ZFloatTensor(4,3):normal()

-- get real part as a FloatTensor (does not share storage)
b = a:re()

-- get imaginary part as a FloatTensor (does not share storage)
c = a:im()

-- get absolute value as a FloatTensor
e = a:abs()
```

### Filling a tensor
```lua
a:fill(1+4i)
```

###Mathematical operations
```lua
-- fill
  a = torch.ZFloatTensor(2,3) -- complex tensor of dimensions 2, 3
a:fill(1+3i)
print(a)
>  1.0000+3.0000i  1.0000+3.0000i  1.0000+3.0000i
>  1.0000+3.0000i  1.0000+3.0000i  1.0000+3.0000i
>  [torch.ZFloatTensor of dimension 2x3]

-- norm
a:norm()
> 2.4494897427832+7.3484692283495i
a:norm(0)
> 6+0i

-- conjugate
a:conj()
>  1.0000-3.0000i  1.0000-3.0000i  1.0000-3.0000i
>  1.0000-3.0000i  1.0000-3.0000i  1.0000-3.0000i
> [torch.ZFloatTensor of dimension 2x3]

-- conjugate transpose
b = a:conj():t()


b = a:clone():fill(1+9i)

-- addition
c = a + b -- out-of-place
a:add(b) -- in-place

-- subtraction
c = a - b -- out-of-place
a:add(-1, b) -- in-place

-- dot product, i.e. a. conj(b)
c = a * b -- a and b are vectors

-- matrix vector product (new result buffer)
c = a * b

-- matrix vector product (existing result buffer)
c:addmv(1, a, b)

-- matrix matrix multiplication (new result buffer)
c = a * b

-- matrix matrix multiplication (existing result buffer)
c:addmm(1, a, b)

-- outer product of vectors (new result buffer)
c = t:ger(a, b)

-- outer product of vectors (existing result buffer)
c = t:addr(1, a, b)
```

### Neural networks

####Linear layer
```
require 'nn'
m = nn.Linear(10,20):type('torch.ZFloatTensor')
m:reset() -- to make sure the imaginary parts are filled with random values
-- use it like any regular neural net layer
input = torch.ZFloatTensor(10):normal()
output = m:forward(input)
gradients = torch.ZFloatTensor(20):normal()
m:backward(input, gradients)
```