--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.

local argcheck = require 'argcheck'
local C = require 'ztorch.THZ'
local torch = require 'torch'
local ztorch = require 'ztorch.env'
local ffi=require 'ffi'
local display=require 'ztorch.display'

for _,Real in ipairs{'Float', 'Double'} do
   local storageType = 'torch.Z' .. Real .. 'Storage'
   local typename = 'torch.Z' .. Real .. 'Tensor'
   local Tensor = torch[Real .. 'Tensor']
   local THZTensor = 'THZ' .. Real .. 'Tensor'
   local THZTensor_Real_abs = C[THZTensor .. '_' .. Real .. '_abs']
   local THZTensor_Real_arg = C[THZTensor .. '_' .. Real .. '_arg']
   local THZTensor_Real_im = C[THZTensor .. '_' .. Real .. '_im']
   local THZTensor_Real_re = C[THZTensor .. '_' .. Real .. '_re']
   local THZTensor_zabs = C[THZTensor .. '_zabs']
   local THZTensor_zarg = C[THZTensor .. '_zarg']
   local THZTensor_zim = C[THZTensor .. '_zim']
   local THZTensor_zre = C[THZTensor .. '_zre']
   local THZTensor_add = C[THZTensor .. '_add']
   local THZTensor_addcdiv = C[THZTensor .. '_addcdiv']
   local THZTensor_addcmul = C[THZTensor .. '_addcmul']
   local THZTensor_cadd = C[THZTensor .. '_cadd']
   local THZTensor_cat = C[THZTensor .. '_cat']
   local THZTensor_cdiv = C[THZTensor .. '_cdiv']
   local THZTensor_cmul = C[THZTensor .. '_cmul']
   local THZTensor_conv2Dcmul = C[THZTensor .. '_conv2Dcmul']
   local THZTensor_conv2Dmul = C[THZTensor .. '_conv2Dmul']
   local THZTensor_conv2Dmv = C[THZTensor .. '_conv2Dmv']
   local THZTensor_conv3Dcmul = C[THZTensor .. '_conv3Dcmul']
   local THZTensor_conv3Dmul = C[THZTensor .. '_conv3Dmul']
   local THZTensor_conv3Dmv = C[THZTensor .. '_conv3Dmv']
   local THZTensor_copy = C[THZTensor .. '_copy']
   local THZTensor_copyByte = C[THZTensor .. '_copyByte']
   local THZTensor_copyChar = C[THZTensor .. '_copyChar']
   local THZTensor_copyDouble = C[THZTensor .. '_copyDouble']
   local THZTensor_copyFloat = C[THZTensor .. '_copyFloat']
   local THZTensor_copyInt = C[THZTensor .. '_copyInt']
   local THZTensor_copyLong = C[THZTensor .. '_copyLong']
   local THZTensor_copyShort = C[THZTensor .. '_copyShort']
   local THZTensor_cross = C[THZTensor .. '_cross']
   local THZTensor_cumprod = C[THZTensor .. '_cumprod']
   local THZTensor_cumsum = C[THZTensor .. '_cumsum']
   local THZTensor_diag = C[THZTensor .. '_diag']
   local THZTensor_dist = C[THZTensor .. '_dist']
   local THZTensor_div = C[THZTensor .. '_div']
   local THZTensor_dot = C[THZTensor .. '_dot']
   local THZTensor_fill = C[THZTensor .. '_fill']
   local THZTensor_free = C[THZTensor .. '_free']
   local THZTensor_geev = C[THZTensor .. '_geev']
   local THZTensor_gels = C[THZTensor .. '_gels']
   local THZTensor_gesv = C[THZTensor .. '_gesv']
   local THZTensor_gesvd = C[THZTensor .. '_gesvd']
   local THZTensor_isContiguous = C[THZTensor .. '_isContiguous']
   local THZTensor_max = C[THZTensor .. '_max']
   local THZTensor_maxall = C[THZTensor .. '_maxall']
   local THZTensor_mean = C[THZTensor .. '_mean']
   local THZTensor_meanall = C[THZTensor .. '_meanall']
   local THZTensor_min = C[THZTensor .. '_min']
   local THZTensor_minall = C[THZTensor .. '_minall']
   local THZTensor_mul = C[THZTensor .. '_mul']
   local THZTensor_nElement = C[THZTensor .. '_nElement']
   local THZTensor_narrow = C[THZTensor .. '_narrow']
   local THZTensor_new = C[THZTensor .. '_new']
   local THZTensor_newClone = C[THZTensor .. '_newClone']
   local THZTensor_newContiguous = C[THZTensor .. '_newContiguous']
   local THZTensor_newNarrow = C[THZTensor .. '_newNarrow']
   local THZTensor_newSelect = C[THZTensor .. '_newSelect']
   local THZTensor_newTranspose = C[THZTensor .. '_newTranspose']
   local THZTensor_newWithSize = C[THZTensor .. '_newWithSize']
   local THZTensor_newWithSize4d = C[THZTensor .. '_newWithSize4d']
   local THZTensor_newWithStorage = C[THZTensor .. '_newWithStorage']
   local THZTensor_newWithTensor = C[THZTensor .. '_newWithTensor']
   local THZTensor_norm = C[THZTensor .. '_norm']
   local THZTensor_normall = C[THZTensor .. '_normall']
   local THZTensor_pow = C[THZTensor .. '_pow']
   local THZTensor_prod = C[THZTensor .. '_prod']
   local THZTensor_reshape = C[THZTensor .. '_reshape']
   local THZTensor_resize = C[THZTensor .. '_resize']
   local THZTensor_resize4d = C[THZTensor .. '_resize4d']
   local THZTensor_resizeAs = C[THZTensor .. '_resizeAs']
   local THZTensor_select = C[THZTensor .. '_select']
   local THZTensor_set = C[THZTensor .. '_set']
   local THZTensor_setStorage = C[THZTensor .. '_setStorage']
   local THZTensor_sort = C[THZTensor .. '_sort']
   local THZTensor_squeeze = C[THZTensor .. '_squeeze']
   local THZTensor_std = C[THZTensor .. '_std']
   local THZTensor_stdall = C[THZTensor .. '_stdall']
   local THZTensor_sum = C[THZTensor .. '_sum']
   local THZTensor_sumall = C[THZTensor .. '_sumall']
   local THZTensor_syev = C[THZTensor .. '_syev']
   local THZTensor_trace = C[THZTensor .. '_trace']
   local THZTensor_transpose = C[THZTensor .. '_transpose']
   local THZTensor_tril = C[THZTensor .. '_tril']
   local THZTensor_triu = C[THZTensor .. '_triu']
   local THZTensor_unfold = C[THZTensor .. '_unfold']
   local THZTensor_var = C[THZTensor .. '_var']
   local THZTensor_varall = C[THZTensor .. '_varall']
   local THZTensor_zero = C[THZTensor .. '_zero']
   local THZStorage = 'THZ' .. Real .. 'Storage'
   local THZStorage_free = C[THZStorage .. '_free']
   local THZStorage_retain = C[THZStorage .. '_retain']

   local ZTensor = {}

   ZTensor.__new = argcheck{
      nonamed=true,
      call =
         function()
            local self = THZTensor_new()
            ffi.gc(self, THZTensor_free)
            return self
         end
   }

   ZTensor.__new = argcheck{
      {name='storage', type=storageType},
      {name='storageOffset', type='number', default=1},
      {name='size', type='table', opt=true},
      {name='stride', type='table', opt=true},
      nonamed=true,
      overload = ZTensor.__new,
      call =
         function(storage, storageOffset, size, stride)
            if size then
               size = torch.LongStorage(size):cdata()
            end
            if stride then
               stride = torch.LongStorage(stride):cdata()
            end
            local self = THZTensor_newWithStorage(storage, storageOffset-1, size, stride)
            ffi.gc(self, THZTensor_free)
            return self
         end
   }

   ZTensor.__new = argcheck{
      {name='dim1', type='number'},
      {name='dim2', type='number', default=0},
      {name='dim3', type='number', default=0},
      {name='dim4', type='number', default=0},
      nonamed=true,
      overload = ZTensor.__new,
      call =
         function(dim1, dim2, dim3, dim4)
            local self = THZTensor_newWithSize4d(dim1, dim2, dim3, dim4)
            ffi.gc(self, THZTensor_free)
            return self
         end
   }

   ZTensor.__new = argcheck{
      {name='size', type='torch.LongStorage'},
      nonamed=true,
      overload = ZTensor.__new,
      call =
         function(size)
            local self = THZTensor_newWithSize(size:cdata(), nil)
            ffi.gc(self, THZTensor_free)
            return self
         end
   }

   ZTensor.__new = argcheck{
      {name='tensor', type=typename},
      nonamed=true,
      overload = ZTensor.__new,
      call =
         function(tensor)
            local self = THZTensor_newWithTensor(tensor)
            ffi.gc(self, THZTensor_free)
            return self
         end
   }

   ZTensor.new = ZTensor.__new

   -- access methods
   local NULL = (not jit) and ffi.C.NULL or nil
   ZTensor.storage = argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            if self.__storage == NULL then
               return nil
            end
            local storage = self.__storage[0]
            THZStorage_retain(storage)
            ffi.gc(storage, THZStorage_free)
            return storage
         end
   }

   ZTensor.storageOffset= argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            return tonumber(self.__storageOffset+1)
         end
   }
   ZTensor.offset = ZTensor.storageOffset

   ZTensor.nDimension = argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            return tonumber(self.__nDimension)
         end
   }
   ZTensor.dim = ZTensor.nDimension

   ZTensor.size = argcheck{
      {name='self', type=typename},
      {name='dim', type='number', opt=true},
      nonamed=true,
      call =
         function(self, dim)
            if dim then
               assert(dim > 0 and dim <= self.__nDimension, 'out of range')
               return tonumber(self.__size[dim-1])
            else
               local dim = tonumber(self.__nDimension)
               local size = torch.LongStorage(dim)
               ffi.copy(size:data(), self.__size, ffi.sizeof('long') * dim)
               return size
            end
         end
   }

   ZTensor.stride = argcheck{
      {name='self', type=typename},
      {name='dim', type='number', opt=true},
      nonamed=true,
      call =
         function(self, dim)
            if dim then
               assert(dim > 0 and dim <= self.__nDimension, 'out of range')
               return tonumber(self.__stride[dim-1])
            else
               local dim = tonumber(self.__nDimension)
               local stride = torch.LongStorage(dim)
               ffi.copy(stride:data(), self.__stride, ffi.sizeof('long') * dim)
               return stride
            end
         end
   }

   ZTensor.data = argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            if self.__storage then
               return self.__storage.__data+self.__storageOffset
            end
         end
   }

   ZTensor.clone = argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            local tensor = THZTensor_newClone(self)
            ffi.gc(tensor, THZTensor_free)
            return tensor
         end
   }

   ZTensor.contiguous = argcheck{
      {name='self', type=typename},
      nonamed=true,
      call =
         function(self)
            local tensor = THZTensor_newContiguous(self)
            ffi.gc(tensor, THZTensor_free)
            return tensor
         end
   }

   ZTensor.set = argcheck{
      {name='self', type=typename},
      {name='src', type=typename},
      nonamed=true,
      call =
         function(self, src)
            THZTensor_set(self, src)
            return self
         end
   }

   ZTensor.set = argcheck{
      {name='self', type=typename},
      {name='storage', type=storageType},
      {name='storageOffset', type='number', default=1},
      {name='size', type='torch.LongStorage', opt=true},
      {name='stride', type='torch.LongStorage', opt=true},
      nonamed=true,
      overload = ZTensor.set,
      call =
         function(self, storage, storageOffset, size, stride)
            local scdata, stdata
            if size then scdata = size:cdata() end
            if stride then stdata = stride:cdata() end
            THZTensor_setStorage(self, storage, storageOffset-1, scdata, stdata)
            return self
         end
   }


   ZTensor.resize = argcheck{
      {name='self', type=typename},
      {name='size', type='table'},
      {name='stride', type='table', opt=true},
      nonamed=true,
      call =
         function(self, size, stride)
            local dim = #size
            assert(not stride or (#stride == dim), 'inconsistent size/stride sizes')
            size = torch.LongStorage(size)
            local stridecdata
            if stride then
               stride = torch.LongStorage(stride)
               stridecdata = stride:cdata()
            end
            THZTensor_resize(self, size:cdata(), stridecdata)
            return self
         end
   }

   ZTensor.resize = argcheck{
      {name='self', type=typename},
      {name='dim1', type='number'},
      {name='dim2', type='number', default=0},
      {name='dim3', type='number', default=0},
      {name='dim4', type='number', default=0},
      nonamed=true,
      overload = ZTensor.resize,
      call =
         function(self, dim1, dim2, dim3, dim4)
            THZTensor_resize4d(self, dim1, dim2, dim3, dim4)
            return self
         end
   }

   ZTensor.resize = argcheck{
      {name='self', type=typename},
      {name='size', type='torch.LongStorage'},
      {name='stride', type='torch.LongStorage', opt=true},
      nonamed=true,
      overload = ZTensor.resize,
      call =
         function(self, size, stride)
            if stride then stride = stride:cdata() end
            THZTensor_resize(self, size:cdata(), stride)
            return self
         end
   }

   ZTensor.resizeAs = argcheck{
      {name='self', type=typename},
      {name='src', type=typename},
      nonamed=true,
      call =
         function(self, src)
            THZTensor_resizeAs(self, src)
            return self
         end
   }

   ZTensor.fill = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="value", type="number"},
      call =
         function(dst, value)
            THZTensor_fill(dst, value)
            return dst
         end
   }

   ZTensor.fill = argcheck{
      nonamed=true,
      {name="self", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      overload=ZTensor.fill,
      call = function(dst, value)
         THZTensor_fill(dst, value)
         return dst
      end
   }

   ZTensor.mean = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return THZTensor_meanall(src)
         end
   }

   ZTensor.mean = argcheck{
      nonamed=true,
      name = "mean",
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      overload=ZTensor.mean,
      call =
         function(dst, src, dim)
            dst = dst or src
            THZTensor_mean(dst, src, dim-1)
            return dst
         end
   }


   ZTensor.std = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return THZTensor_stdall(src)
         end
   }

   ZTensor.std = argcheck{
      nonamed=true,
      name = "std",
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      {name="flag", type="boolean", default=false},
      overload=ZTensor.std,
      call =
         function(dst, src, dim, flag)
            dst = dst or src
            THZTensor_std(dst, src, dim-1, flag and 1 or 0)
            return dst
         end
   }

   ZTensor.var = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return THZTensor_varall(src)
         end
   }

   ZTensor.var = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      {name="flag", type="boolean", default=false},
      overload=ZTensor.var,
      call =
         function(dst, src, dim, flag)
            dst = dst or src
            THZTensor_var(dst, src, dim-1, flag and 1 or 0)
            return dst
         end
   }

   ZTensor.norm = argcheck{
      nonamed=true,
      {name="src", type=typename},
      {name="n", type="number", default=2},
      call = THZTensor_normall
   }

   ZTensor.norm = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="src", type=typename, opt=true},
      {name="n", type="number"},
      {name="dim", type="number"},
      overload = ZTensor.norm,
      call =
         function(dst, src, n, dim)
            dst = dst or src
            THZTensor_norm(dst, src, n, dim-1)
            return dst
         end
   }

   ZTensor.dist = argcheck{
      nonamed=true,
      name = "dist",
      {name="dst", type=typename},
      {name="src", type=typename},
      {name="n", type="number", default=2},
      call = THZTensor_dist
   }

   ZTensor.re = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call = function(src)
         local dst = Tensor.new()
         THZTensor_Real_re(dst:cdata(), src)
         return dst
      end
   }
   ZTensor.re = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="src", type=typename},
      overload=ZTensor.re,
      call = function(dst, src)
         THZTensor_zre(dst, src)
         return dst
      end
   }

   ZTensor.im = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call = function(src)
         local dst = Tensor.new()
         THZTensor_Real_im(dst:cdata(), src)
         return dst
      end
   }
   ZTensor.im = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="src", type=typename},
      overload=ZTensor.im,
      call = function(dst, src)
         THZTensor_zim(dst, src)
         return dst
      end
   }

   -- Add re and im functions to torch.FloatTensor and torch.DoubleTensor
   for _,BaseReal in ipairs{'Float', 'Double'} do
      local basename = 'torch.' .. BaseReal .. 'Tensor'
      local metatable = torch.getmetatable(basename)
      for _,funcname in pairs{'re','im'} do
         local func = C[THZTensor .. '_' .. BaseReal .. '_' .. funcname]

         rawset(metatable, funcname, argcheck{
            {name='dst', type=basename},
            {name='src', type=typename},
            nonamed=true,
            overload = rawget(metatable, funcname),
            call =
               function(dst, src)
                  func(dst:cdata(), src)
                  return dst
               end
         })
      end
   end

   ZTensor.abs = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call = function(src)
         local dst = Tensor.new()
         THZTensor_Real_abs(dst:cdata(), src)
         return dst
      end
   }
   ZTensor.abs = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="src", type=typename},
      overload=ZTensor.abs,
      call = function(dst, src)
         THZTensor_zabs(dst, src)
         return dst
      end
   }

   ZTensor.arg = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call = function(src)
         local dst = Tensor.new()
         THZTensor_Real_arg(dst:cdata(), src)
         return dst
      end
   }
   ZTensor.arg = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      {name="src", type=typename},
      overload=ZTensor.arg,
      call = function(dst, src)
         THZTensor_zarg(dst, src)
         return dst
      end
   }

   for _,name in ipairs{'log', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                        'sinh', 'tan', 'atan', 'tanh', 'sqrt',
                        'conj', 'proj'} do
      local func = C['THZ' .. Real .. 'Tensor_' .. name]
      ZTensor[name] = argcheck{
         nonamed=true,
         name = name,
         {name="dst", type=typename, opt=true},
         {name="src", type=typename},
         call =
            function(dst, src)
               dst = dst or src
               func(dst, src)
               return dst
            end
      }
   end

   ZTensor.pow = argcheck{
      nonamed=true,
      name = "pow",
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number"},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_pow(dst, src)
            return dst
         end
   }

   ZTensor.narrow = argcheck{
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      {name='dim', type='number'},
      {name='idx', type='number'},
      {name='size', type='number'},
      nonamed=true,
      call =
         function(self, src, dim, idx, size)
            if src then
               THZTensor_narrow(self, src, dim-1, idx-1, size)
               return self
            else
               local tensor = THZTensor_newNarrow(self, dim-1, idx-1, size)
               ffi.gc(tensor, THZTensor_free)
               return tensor
            end
         end
   }

   ZTensor.select = argcheck{
      nonamed=true,
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      {name='dim', type='number'},
      {name='idx', type='number'},
      call =
         function(self, src, dim, idx)
            if src then
               THZTensor_select(self, src, dim-1, idx-1)
               return self
            else
               local tensor = THZTensor_newSelect(self, dim-1, idx-1)
               ffi.gc(tensor, THZTensor_free)
               return tensor
            end
         end
   }

   ZTensor.t = argcheck{
      nonamed=true,
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      call =
         function(self, src)
            if src then
               assert(src.__nDimension == 2, 'tensor to be transposed must be 2D')
               THZTensor_transpose(self, src, 0, 1)
               return self
            else
               assert(self.__nDimension == 2, 'tensor to be transposed must be 2D')
               local tensor = THZTensor_newTranspose(self, 0, 1)
               ffi.gc(tensor, THZTensor_free)
               return tensor
            end
         end
   }

   ZTensor.transpose = argcheck{
      nonamed=true,
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      {name='dim1', type='number'},
      {name='dim2', type='number'},
      call =
         function(self, src, dim1, dim2)
            if src then
               THZTensor_transpose(self, src, dim1-1, dim2-1)
               return self
            else
               local tensor = THZTensor_newTranspose(self, dim1-1, dim2-1)
               ffi.gc(tensor, THZTensor_free)
               return tensor
            end
         end
   }

   ZTensor.unfold = argcheck{
      nonamed=true,
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      {name='dim', type='number'},
      {name='size', type='number'},
      {name='step', type='number'},
      call =
         function(self, src, dim, size, step)
            if src then
               THZTensor_unfold(self, src, dim-1, size, step)
               return self
            else
               local tensor = THZTensor_newTranspose(self, src, dim-1, size, step)
               ffi.gc(tensor, THZTensor_free)
               return tensor
            end
         end
   }

   ZTensor.squeeze = argcheck{
      nonamed=true,
      {name='self', type=typename},
      {name='src', type=typename, opt=true},
      call =
         function(self, src)
            local dst = src and self or ZTensor.new()
            src = src or self
            if src.__nDimension == 0 then
               return
            elseif src.__nDimension == 1 and src:size(1) == 1 then
               return src:data()[0]
            else
               THZTensor_squeeze(dst, src)
               return dst
            end
         end
   }

   ZTensor.isContiguous = argcheck{
      nonamed=true,
      {name='self', type=typename},
      call =
         function(self)
            return THZTensor_isContiguous(self) == 1
         end
   }

   ZTensor.nElement = argcheck{
      nonamed=true,
      {name='self', type=typename},
      call =
         function(self)
            return tonumber(THZTensor_nElement(self))
         end
   }

   ZTensor.zero = argcheck{
      nonamed=true,
      {name="dst", type=typename},
      call =
         function(dst)
            THZTensor_zero(dst)
            return dst
         end
   }

   ZTensor.dot = argcheck{
      nonamed=true,
      {name="src1", type=typename},
      {name="src2", type=typename},
      call =
         function(src1, src2)
            return THZTensor_dot(src1, src2)
         end
   }

   ZTensor.min = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return tonumber(THZTensor_minall(src))
         end
   }

   ZTensor.min = argcheck {
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="idx", type="torch.LongTensor", opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      overload=ZTensor.min,
      call =
         function(dst, idx, src, dim)
            dst = dst or src
            idx = idx or torch.LongTensor()
            THZTensor_min(dst, idx, src, dim-1)
            idx:add(1)
            return dst, idx
         end
   }

   ZTensor.max = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return tonumber(THZTensor_maxall(src))
         end
   }

   ZTensor.max = argcheck {
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="idx", type="torch.LongTensor", opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      overload=ZTensor.max,
      call =
         function(dst, idx, src, dim)
            dst = dst or src
            idx = idx or torch.LongTensor()
            THZTensor_max(dst, idx, src, dim-1)
            idx:add(1)
            return dst, idx
         end
   }

   ZTensor.sum = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return THZTensor_sumall(src)
         end
   }

   ZTensor.sum = argcheck {
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      overload=ZTensor.sum,
      call =
         function(dst, src, dim)
            dst = dst or src
            THZTensor_sum(dst, src, dim-1)
            return dst
         end
   }

   ZTensor.add = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number"},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_add(dst, src, value)
            return dst
         end
   }

   ZTensor.add = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_add(dst, src, value)
            return dst
         end
   }

   ZTensor.add = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="value", type="number", default=1},
      {name="src2", type=typename},
      overload=ZTensor.add,
      call =
         function(dst, src1, value, src2)
            dst = dst or src1
            THZTensor_cadd(dst, src1, value, src2)
            return dst
         end
   }

   ZTensor.add = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      {name="src2", type=typename},
      overload=ZTensor.add,
      call =
         function(dst, src1, value, src2)
            dst = dst or src1
            THZTensor_cadd(dst, src1, value, src2)
            return dst
         end
   }

   ZTensor.mul = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number"},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_mul(dst, src, value)
            return dst
         end
   }
   ZTensor.mul = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      overload=ZTensor.mul,
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_mul(dst, src, value)
            return dst
         end
   }

   ZTensor.cmul = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      call =
         function(dst, src1, src2)
            dst = dst or src1
            THZTensor_cmul(dst, src1, src2)
            return dst
         end
   }


   ZTensor.div = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_div(dst, src, value)
            return dst
         end
   }

   ZTensor.div = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number"},
      call =
         function(dst, src, value)
            dst = dst or src
            THZTensor_div(dst, src, value)
            return dst
         end
   }

   ZTensor.cdiv = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      call =
         function(dst, src1, src2)
            dst = dst or src1
            THZTensor_cdiv(dst, src1, src2)
            return dst
         end
   }

   ZTensor.addcmul = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number", default=1},
      {name="src1", type=typename},
      {name="src2", type=typename},
      call =
         function(dst, src, value, src1, src2)
            dst = dst or src
            THZTensor_addcmul(dst, src, value, src1, src2)
            return dst
         end
   }

   ZTensor.addcdiv = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="value", type="number", default=1},
      {name="src1", type=typename},
      {name="src2", type=typename},
      call =
         function(dst, src, value, src1, src2)
            dst = dst or src
            THZTensor_addcdiv(dst, src, value, src1, src2)
            return dst
         end
   }

   ZTensor.numel = ZTensor.nElement

   ZTensor.prod = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      call =
         function(dst, src, dim)
            dst = dst or src
            THZTensor_prod(dst, src, dim-1)
            return dst
         end
   }

   ZTensor.cumsum = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      call =
         function(dst, src, dim)
            dst = dst or src
            THZTensor_cumsum(dst, src, dim-1)
            return dst
         end
   }

   ZTensor.cumprod = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="dim", type="number"},
      call =
         function(dst, src, dim)
            dst = dst or src
            THZTensor_cumprod(dst, src, dim-1)
            return dst
         end
   }

   ZTensor.trace = argcheck{
      nonamed=true,
      {name="src", type=typename},
      call =
         function(src)
            return tonumber(THZTensor_trace(src))
         end
   }

   ZTensor.cross = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="dim", type="number", default=0},
      call =
         function(dst, src1, src2, dim)
            dst = dst or ZTensor.new()
            THZTensor_cross(dst, src1, src2, dim-1)
            return dst
         end
   }

   ZTensor.diag = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="k", type='number', default=0},
      call =
         function(dst, src, k)
            dst = dst or src
            THZTensor_diag(dst, src, k)
            return dst
         end
   }

   ZTensor.reshape = argcheck{
      nonamed=true,
      name = "reshape",
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="size", type='numbers'},
      call =
         function(dst, src, size)
            dst = dst or src
            THZTensor_reshape(dst, src, size)
            return dst
         end
   }

   ZTensor.sort = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="idx", type='torch.LongTensor', opt=true},
      {name="src", type=typename},
      {name="dim", type='number', opt=true},
      {name="descend", type='boolean', default=false},
      call =
         function(dst, idx, src, dim, descend)
            dst = dst or src
            idx = idx or torch.LongTensor()
            dim = dim or src:nDimension()
            THZTensor_sort(dst, idx, src, dim-1, descend and 1 or 0)
            idx:add(1)
            return dst, idx
         end
   }

   ZTensor.tril = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="k", type='number', default=0},
      call =
         function(dst, src, k)
            dst = dst or src
            THZTensor_tril(dst, src, k)
            return dst
         end
   }

   ZTensor.triu = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src", type=typename},
      {name="k", type='number', default=0},
      call =
         function(dst, src, k)
            dst = dst or src
            THZTensor_triu(dst, src, k)
            return dst
         end
   }

   ZTensor.cat = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="dim", type='number', opt=true},
      call =
         function(dst, src1, src2, dim)
            dst = dst or src1
            dim = dim or src1:nDimension()
            THZTensor_cat(dst, src1, src2, dim-1)
            return dst
         end
   }

   -- comparison
   for _,name in ipairs{'lt','gt','le','ge','eq','ne'} do
      local func_val = C[THZTensor .. '_' .. name .. 'Value']
      local func_tens = C[THZTensor .. '_' .. name .. 'Tensor']

      ZTensor[name] = argcheck{
         nonamed=true,
         {name="dst", type='torch.ByteTensor', opt=true},
         {name="src", type=typename},
         {name="value", type='number'},
         call =
            function(dst, src, value)
               local res = dst or torch.ByteTensor()
               func_val(res, src, value)
               return res
            end
      }

      ZTensor[name] = argcheck{
         nonamed=true,
         {name="dst", type='torch.ByteTensor', opt=true},
         {name="src1", type=typename},
         {name="src2", type=typename},
         overload=ZTensor[name],
         call =
            function(dst, src1, src2)
               local res = dst or torch.ByteTensor()
               func_tens(res, src1, src2)
               return res
            end
      }
   end

   for _, f in ipairs{
      {name="mv", addname="addmv", arg1="mat", arg2="vec"},
      {name="mm", addname="addmm", arg1="mat", arg2="mat"},
      {name="ger", addname="addr", arg1="vec1", arg2="vec2"},
      {name="geru", addname="addru", arg1="vec1", arg2="vec2"}} do

      local func = C[THZTensor .. "_" .. f.addname]

      ZTensor[f.name] = argcheck{
         nonamed=true,
         {name=f.arg1, type=typename},
         {name=f.arg2, type=typename},
         call =
            function(arg1, arg2)
               local res
               if f.name == 'mv' then
                  res = ZTensor.new(arg1:size(1)):zero()
               elseif f.name == 'mm' then
                  res = ZTensor.new(arg1:size(1), arg2:size(2)):zero()
               elseif f.name == 'ger' or f.name == 'geru' then
                  res = ZTensor.new(arg1:size(1), arg2:size(1)):zero()
               end
               func(res, 0, res, 1, arg1, arg2)
               return res
            end
      }

      ZTensor[f.addname] = argcheck{
         nonamed=true,
         {name="dst", type=typename, opt=true},
         {name="beta", type='number', default=1},
         {name="src", type=typename},
         {name="alpha", type='number', default=1},
         {name="mat", type=typename}, -- could check dim
         {name="vec", type=typename},
         call =
            function(dst, beta, src, alpha, mat, vec)
               dst = dst or src
               func(dst, beta, src, alpha, mat, vec)
               return dst
            end
      }
      ZTensor[f.addname] = argcheck{
         nonamed=true,
         {name="src", type=typename},
         {name="beta", type='number'},
         {name="alpha", type='number'},
         {name="mat", type=typename}, -- could check dim
         {name="vec", type=typename},
         overload=ZTensor[f.addname],
         call =
            function(src, beta, alpha, mat, vec)
               func(src, beta, src, alpha, mat, vec)
               return src
            end
      }
   end

   ZTensor.conv2 = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="opt", type="string", default='V'},
      call =
         function(dst, src1, src2, opt)
            assert(opt == 'F' or opt == 'V', 'option must be F or V')
            dst = dst or src1
            if src1.__nDimension == 2 and src2.__nDimension == 2 then
               THZTensor_conv2Dmul(dst, 0, 1, src1, src2, 1, 1, opt, 'C')
            elseif src1.__nDimension == 3 and src2.__nDimension == 3 then
               THZTensor_conv2Dcmul(dst, 0, 1, src1, src2, 1, 1, opt, 'C')
            elseif src1.__nDimension == 3 and src2.__nDimension == 4 then
               THZTensor_conv2Dmv(dst, 0, 1, src1, src2, 1, 1, opt, 'C')
            else
               error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
            end
            return dst
         end
   }

   ZTensor.xcorr2 = argcheck{
      nonamed=true,
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="opt", type="string", default='V'},
      call =
         function(dst, src1, src2, opt)
            assert(opt == 'F' or opt == 'V', 'option must be F or V')
            dst = dst or src1
            if src1.__nDimension == 2 and src2.__nDimension == 2 then
               THZTensor_conv2Dmul(dst, 0, 1, src1, src2, 1, 1, opt, 'X')
            elseif src1.__nDimension == 3 and src2.__nDimension == 3 then
               THZTensor_conv2Dcmul(dst, 0, 1, src1, src2, 1, 1, opt, 'X')
            elseif src1.__nDimension == 3 and src2.__nDimension == 4 then
               THZTensor_conv2Dmv(dst, 0, 1, src1, src2, 1, 1, opt, 'X')
            else
               error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
            end
            return dst
         end
   }

   ZTensor.conv3 = argcheck{
      nonamed=true,
      name = "conv3",
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="opt", type="string", default='V'},
      call =
         function(dst, src1, src2, opt)
            assert(opt == 'F' or opt == 'V', 'option must be F or V')
            dst = dst or src1
            if src1.__nDimension == 3 and src2.__nDimension == 3 then
               THZTensor_conv3Dmul(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
            elseif src1.__nDimension == 4 and src2.__nDimension == 4 then
               THZTensor_conv3Dcmul(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
            elseif src1.__nDimension == 4 and src2.__nDimension == 5 then
               THZTensor_conv3Dmv(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
            else
               error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
            end
            return dst
         end
   }

   ZTensor.xcorr3 = argcheck{
      nonamed=true,
      name = "xcorr3",
      {name="dst", type=typename, opt=true},
      {name="src1", type=typename},
      {name="src2", type=typename},
      {name="opt", type="string", default='V'},
      call =
         function(dst, src1, src2, opt)
            assert(opt == 'F' or opt == 'V', 'option must be F or V')
            dst = dst or src1
            if src1.__nDimension == 3 and src2.__nDimension == 3 then
               THZTensor_conv3Dmul(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
            elseif src1.__nDimension == 4 and src2.__nDimension == 4 then
               THZTensor_conv3Dcmul(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
            elseif src1.__nDimension == 4 and src2.__nDimension == 5 then
               THZTensor_conv3Dmv(dst, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
            else
               error('invalid source dimensions (expected: 3/3 or 4/4 or 4/5')
            end
            return res
         end
   }

   ZTensor.gesv = argcheck{
      nonamed=true,
      {name="B", type=typename},
      {name="A", type=typename},
      call =
         function(B, A)
            local X = ZTensor.new()
            local LU = ZTensor.new()
            THZTensor_gesv(X, LU, B, A)
            return X, LU
         end
   }

   ZTensor.gesv = argcheck{
      nonamed=true,
      {name="X", type=typename},
      {name="LU", type=typename},
      {name="B", type=typename},
      {name="A", type=typename},
      overload=ZTensor.gesv,
      call =
         function(X, LU, B, A)
            THZTensor_gesv(X, LU, B, A)
            return X, LU
         end
   }

   ZTensor.gels = argcheck{
      nonamed=true,
      {name="B", type=typename},
      {name="A", type=typename},
      call =
         function(B, A)
            local X = ZTensor.new()
            local LU = ZTensor.new()
            THZTensor_gels(X, LU, B, A)
            return X, LU
         end
   }

   ZTensor.gels = argcheck{
      nonamed=true,
      {name="X", type=typename},
      {name="LU", type=typename},
      {name="B", type=typename},
      {name="A", type=typename},
      overload=ZTensor.gels,
      call =
         function(X, LU, B, A)
            THZTensor_gels(X, LU, B, A)
            return X, LU
         end
   }

   ZTensor.symeig = argcheck{
      nonamed=true,
      {name="A", type=typename},
      {name="opteig", type="string", default='N'},
      {name="opttriang", type="string", default='U'},
      call =
         function(A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            assert(opttriang == 'L' or opttriang == 'U', '  : L or U expected')
            local E = ZTensor.new()
            local V = ZTensor.new()
            THZTensor_syev(E, V, A, opteig, opttriang)
            return E, V
         end
   }

   ZTensor.symeig = argcheck{
      nonamed=true,
      {name="E", type=typename},
      {name="V", type=typename},
      {name="A", type=typename},
      {name="opteig", type="string", default='N'},
      {name="opttriang", type="string", default='U'},
      overload=ZTensor.symeig,
      call =
         function(E, V, A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            assert(opttriang == 'L' or opttriang == 'U', 'opttriang: L or U expected')
            THZTensor_syev(E, V, A, opteig, opttriang)
            return E, V
         end
   }

   ZTensor.eig = argcheck{
      nonamed=true,
      name = "eig",
      {name="A", type=typename},
      {name="opteig", type="string", default='N'},
      call =
         function(A, opteig)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            local E = ZTensor.new()
            local V = ZTensor.new()
            THZTensor_geev(E, V, A, opteig)
            return E, V
         end
   }

   ZTensor.eig = argcheck{
      nonamed=true,
      name = "eig",
      {name="E", type=typename},
      {name="V", type=typename},
      {name="A", type=typename},
      {name="opteig", type="string", default='N'},
      overload=ZTensor.eig,
      call =
         function(E, V, A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            THZTensor_geev(E, V, A, opteig)
            return E, V
         end
   }

   ZTensor.svd = argcheck{
      nonamed=true,
      {name="A", type=typename},
      {name="opteig", type="string", default='S'},
      call =
         function(A, opteig)
            assert(opteig == 'S' or opteig == 'A', 'opteig: S or A expected')
            local U = ZTensor.new()
            local S = ZTensor.new()
            local V = ZTensor.new()
            THZTensor_gesvd(U, S, V, A, opteig)
            return U, S, V
         end
   }

   ZTensor.svd = argcheck{
      nonamed=true,
      {name="U", type=typename},
      {name="S", type=typename},
      {name="V", type=typename},
      {name="A", type=typename},
      {name="opteig", type="string", default='S'},
      overload=ZTensor.svd,
      call =
         function(U, S, V, A, opteig)
            assert(opteig == 'S' or opteig == 'A', 'opteig: S or A expected')
            THZTensor_gesvd(U, S, V, A, opteig)
            return U, S, V
         end
   }

   for _, name in ipairs{'inverse', 'potri', 'potrf'} do
      local cname = name == 'inverse' and 'getri' or name
      local func = C[THZTensor .. '_' .. cname]
      ZTensor[name] = argcheck{
         nonamed=true,
         {name="dst", type=typename, opt=true},
         {name="src", type=typename},
         call =
            function(dst, src)
               dst = dst or src
               func(dst, src)
               return dst
            end
      }

   end

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type=typename},
      call =
         function(dst, src)
            THZTensor_copy(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.ByteTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyByte(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.CharTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyChar(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.ShortTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyShort(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.IntTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyInt(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.LongTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyLong(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.FloatTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyFloat(dst, src)
            return dst
         end
   }

   ZTensor.copy = argcheck{
      nonamed=true,
      name = "copy",
      {name="dst", type=typename},
      {name="src", type='torch.DoubleTensor'},
      overload=ZTensor.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZTensor_copyDouble(dst, src)
            return dst
         end
   }

   ZTensor.__factory =
      function(file)
         return ZTensor.__new()
      end

   function ZTensor:__write(file)
      file:writeObject(self:size())
      file:writeObject(self:stride())
      file:writeLong(self:storageOffset())
      file:writeObject(self:storage())
   end

   function ZTensor:__read(file, version)
      local size = file:readObject()
      local stride = file:readObject()
      local offset = file:readLong()-1
      local storage = file:readObject()
      THZTensor_setStorage(self, storage, offset, size:cdata(), stride:cdata())
   end

   -- define index and newindex for dispatch
   function ZTensor:__newindex(k, v)
      local type_k = torch.type(k)
      local type_v = torch.type(v)
      assert(self, 'Need to call as method')
      if type_k == 'number' then
         if type_v == 'number' or ztorch.isComplex(v) then -- set single value
            if self.__nDimension == 1 then
               assert(k > 0 and k <= tonumber(self.__size[0]), 'out of range')
               self.__storage.__data[self.__storageOffset+(k-1)*self.__stride[0]] = v
            elseif self.__nDimension > 1 then
               local t = THZTensor_newWithTensor(self)
               THZTensor_narrow(t, nil, 0, k-1, 1)
               THZTensor_fill(t, v)
               THZTensor_free(t)
            else
               error('empty tensor')
            end
         elseif type_v == typename then
            local t = self:narrow(1, k, 1)
            t:copy(v)
         end
      elseif type(k) == 'table' then
         assert(#k <= self.__nDimension, 'too many indices provided')
         local t = self
         for dim,idx in ipairs(k) do
            if dim == #k then
               t[idx] = v
            else
               t = t[idx]
            end
         end
      else
         rawset(ZTensor, k, v)
      end
   end

   function ZTensor:__index(k)
      assert(self, 'Need to call as method')
      if type(k) == 'number' then
         if self.__nDimension == 1 then
            assert(k > 0 and k <= tonumber(self.__size[0]), 'out of range')
            return self.__storage.__data[(k-1)*self.__stride[0]+self.__storageOffset]
         elseif self.__nDimension > 1 then
            assert(k > 0 and k <= tonumber(self.__size[0]), 'out of range')
            return self:select(1, k)
         else
            error('empty tensor')
         end
      elseif type(k) == 'table' then
         assert(#k <= self.__nDimension, 'too many indices provided')
         local t = self
         for _,idx in ipairs(k) do
            t = t[idx]
         end
         return t
      else
         return rawget(ZTensor, k)
      end
   end

   function ZTensor:__pairs()
      return pairs(ZTensor)
   end

   ZTensor.__tostring = display.tensor

   function ZTensor:__len()
      return self:size()
   end

   function ZTensor.__add(t1, t2)
      local type_t1 = torch.type(t1)
      local type_t2 = torch.type(t2)

      local r = ZTensor.new()
      if type_t1 == typename and (type_t2 == 'number' or ztorch.isComplex(t2)) then
         r:resizeAs(t1)
         r:copy(t1)
         r:add(t2)
      elseif type_t1 == typename and type_t2 == typename then
         r:resizeAs(t1)
         r:copy(t1)
         r:add(t2)
      else
         error('two tensors or one tensor and one number expected')
      end

      return r
   end

   function ZTensor.__sub(t1, t2)
      local type_t1 = torch.type(t1)
      local type_t2 = torch.type(t2)

      local r = ZTensor.new()
      if type_t1 == typename and (type_t2 == 'number' or ztorch.isComplex(t2)) then
         r:resizeAs(t1)
         r:copy(t1)
         r:add(-t2)
      elseif type_t1 == typename and type_t2 == typename then
         r:resizeAs(t1)
         r:copy(t1)
         r:add(-1, t2)
      else
         error('two tensors or one tensor and one number expected')
      end

      return r
   end

   function ZTensor.__unm(self)
      local r = ZTensor.new()
      r:resizeAs(self)
      r:zero()
      r:add(-1, self)
      return r
   end

   function ZTensor.__mul(t1, t2)
      local type_t1 = torch.type(t1)
      local type_t2 = torch.type(t2)

      local r = ZTensor.new()
      if type_t1 == typename and (type_t2 == 'number' or ztorch.isComplex(t2)) then
         r:resizeAs(t1)
         r:zero()
         r:add(t2, t1)
      elseif type_t1 == typename and type_t2 == typename then
         if t1.__nDimension == 1 and t2.__nDimension == 1 then
            return t1:dot(t2)
         elseif t1.__nDimension == 2 and t2.__nDimension == 1 then
            return t1:mv(t2)
         elseif t1.__nDimension == 2 and t2.__nDimension == 2 then
            return t1:mm(t2)
         else
            error(string.format('multiplication between %dD and %dD tensors not yet supported',
                                t1.__nDimension, t2.__nDimension))
         end
      else
         error('two tensors or one tensor and one number expected')
      end

      return r
   end

   function ZTensor.__div(t1, t2)
      local type_t1 = torch.type(t1)
      local type_t2 = torch.type(t2)

      assert(type_t2 == 'number' or ztorch.isComplex(t2), 'number (real or complex) expected')

      local r = ZTensor.new()
      r:resizeAs(t1)
      r:copy(t1)
      r:mul(1/t2)

      return r
   end

   ---------------------------------------------------------------------------------------
   -- RNG functions
   ZTensor.normal = argcheck{
      nonamed=true,
      name = "normal",
      {name="src", type=typename},
      {name="mean", type='number', default=0},
      {name="stdv", type='number', default=1},
      call =
         function(src, mean, stdv)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):normal(mean, stdv)
            src:copy(f)
            return src
         end
   }

   ZTensor.uniform = argcheck{
      nonamed=true,
      name = "uniform",
      {name="src", type=typename},
      {name="mean", type='number', default=0},
      {name="stdv", type='number', default=1},
      call =
         function(src, mean, stdv)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):uniform(mean, stdv)
            src:copy(f)
            return src
         end
   }

   ZTensor.cauchy = argcheck{
      nonamed=true,
      name = "cauchy",
      {name="src", type=typename},
      {name="mean", type='number', default=0},
      {name="stdv", type='number', default=1},
      call =
         function(src, mean, stdv)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):cauchy(mean, stdv)
            src:copy(f)
            return src
         end
   }

   ZTensor.logNormal = argcheck{
      nonamed=true,
      name = "logNormal",
      {name="src", type=typename},
      {name="mean", type='number', default=1},
      {name="stdv", type='number', default=2},
      call =
         function(src, mean, stdv)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):logNormal(mean, stdv)
            src:copy(f)
            return src
         end
   }

   ZTensor.bernoulli = argcheck{
      nonamed=true,
      name = "bernoulli",
      {name="src", type=typename},
      {name="p", type='number', default=0.5},
      call =
         function(src, p)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):bernoulli(p)
            src:copy(f)
            return src
         end
   }

   ZTensor.geometric = argcheck{
      nonamed=true,
      name = "geometric",
      {name="src", type=typename},
      {name="p", type='number', default=0.5},
      call =
         function(src, p)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):geometric(p)
            src:copy(f)
            return src
         end
   }

   ZTensor.exponential = argcheck{
      nonamed=true,
      name = "exponential",
      {name="src", type=typename},
      {name="p", type='number'},
      call =
         function(src, p)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):exponential(p)
            src:copy(f)
            return src
         end
   }

   ZTensor.random = argcheck{
      nonamed=true,
      name = "random",
      {name="src", type=typename},
      call =
         function(src)
            local size = src:size():totable()
            size[#size+1] = 2
            local size2 = torch.LongStorage(size)
            local f = Tensor.new(size2):random()
            src:copy(f)
            return src
         end
   }

   ---------------------------------------------------------------------------------------
   for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor'} do
      ZTensor[func] = torch[func]
   end

   ZTensor.__version = 0
   ZTensor.__typename = typename
   torch.metatype(typename, ZTensor, THZTensor .. '&')
   ffi.metatype(THZTensor, ZTensor)

   -- constructor metatable
   local ZTensor_ctr = {}
   setmetatable(ZTensor_ctr, {
                   __call =
                      function(self, ...)
                         return ZTensor.__new(...)
                      end,
                   __index = ZTensor,
                   __newindex = ZTensor,
                   __len = ZTensor,
                   __typename = typename,
   })
   torch['Z' .. Real .. 'Tensor'] = ZTensor_ctr

end
