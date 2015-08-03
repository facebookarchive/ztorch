--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.

require 'torch'

local ffi = require 'ffi'
local argcheck = require 'argcheck'
local C = require 'ztorch.THZ'

local ztorch = require 'ztorch.env'
ztorch.complex = require 'ztorch.complex'
ztorch.fcomplex = require 'ztorch.fcomplex'
function ztorch.isComplex(v)
   return ffi.istype(ztorch.complex.type, v) or ffi.istype(ztorch.fcomplex.type, v)
end
function ztorch.im(im)
   return ffi.new('complex', 0, im)
end

local argcheckenv = require 'argcheck.env'
function argcheckenv.istype(obj, typename)
   local tname = torch.typename(obj)
   return tname and tname == typename or type(obj) == typename
end

require 'ztorch.Storage'
require 'ztorch.Tensor'

ztorch.re = argcheck{
   {name='value', type='number'},
   nonamed=true,
   call =
      function(value)
         return ffi.new('complex', value, 0)
      end
}
ztorch.im = argcheck{
   {name='value', type='number'},
   nonamed=true,
   call =
      function(value)
         return ffi.new('complex', 0, value)
      end
}

-- HACK: until we get torch.isTypeOf to work with complex tensors and storages
function torch.isTensor(obj)
   local typename = torch.typename(obj)
   if typename and typename:find('torch.*Tensor') then
      return true
   end
   return false
end
function torch.isStorage(obj)
   local typename = torch.typename(obj)
   if typename and typename:find('torch.*Storage') then
      return true
   end
   return false
end

return ztorch
