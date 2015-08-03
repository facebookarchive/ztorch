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
local complex = require 'ztorch.complex'
local ffi = require 'ffi'
local display = require 'ztorch.display'

for _,Real in ipairs{'Float', 'Double'} do
   local ctype = Real:lower() .. ' _Complex'
   local typename = 'torch.Z' .. Real .. 'Storage'
   local THZStorage = 'THZ' .. Real .. 'Storage'
   local THZStorage_newWithSize = C[THZStorage .. '_newWithSize']
   local THZStorage_newWithAllocator = C[THZStorage .. '_newWithAllocator']
   local THZStorage_newWithMapping = C[THZStorage .. '_newWithMapping']
   local THZStorage_free = C[THZStorage .. '_free']
   local THZStorage_fill = C[THZStorage .. '_fill']
   local THZStorage_resize = C[THZStorage .. '_resize']
   local THZStorage_copyZFloat = C[THZStorage .. '_copyZFloat']
   local THZStorage_copyZDouble = C[THZStorage .. '_copyZDouble']
   local THZStorage_copyByte = C[THZStorage .. '_copyByte']
   local THZStorage_copyChar = C[THZStorage .. '_copyChar']
   local THZStorage_copyShort = C[THZStorage .. '_copyShort']
   local THZStorage_copyInt = C[THZStorage .. '_copyInt']
   local THZStorage_copyLong = C[THZStorage .. '_copyLong']
   local THZStorage_copyFloat = C[THZStorage .. '_copyFloat']
   local THZStorage_copyDouble = C[THZStorage .. '_copyDouble']

   local ZStorage = {}

   ZStorage.__new = argcheck{
      {name="size", type="number", default=0},
      nonamed = true,
      call =
         function(size)
            local self = THZStorage_newWithSize(size)
            ffi.gc(self, THZStorage_free)
            return self
         end
   }

   ZStorage.__new = argcheck{
      {name="allocator", type="cdata"},
      {name="size", type="number", default=0},
      overload = ZStorage.__new,
      nonamed = true,
      call =
         function(allocator, size)
            local self = THZStorage_newWithAllocator(size, allocator, nil)
            ffi.gc(self, THZStorage_free)
            return self
         end
   }

   ZStorage.__new = argcheck{
      {name="table", type="table"},
      overload = ZStorage.__new,
      nonamed = true,
      call =
         function(tbl)
            local size = #tbl
            local self = THZStorage_newWithSize(size)
            ffi.gc(self, THZStorage_free)
            for i=1,size do
               self.data[i-1] = tbl[i]
            end
            return self
         end
   }

   ZStorage.__new = argcheck{
      {name="filename", type="string"},
      {name="shared", type="boolean", default=false},
      {name="size", type="number", default=0},
      overload = ZStorage.__new,
      nonamed = true,
      call =
         function(filename, shared, size)
            local self = THZStorage_newWithMapping(filename, size,
                                                   shared and 1 or 0)
            ffi.gc(self, THZStorage_free)
            return self
         end
   }

   ZStorage.new = ZStorage.__new

   ZStorage.fill = argcheck{
      {name="self", type=typename},
      {name="value", type="number"},
      call = THZStorage_fill
   }

   ZStorage.fill = argcheck{
      {name="self", type=typename},
      {name="value", type="cdata", check=ztorch.isComplex},
      overload=ZStorage.fill,
      call = THZStorage_fill
   }

   ZStorage.size = argcheck{
      {name="self", type=typename},
      call =
         function(self)
            return tonumber(self.__size)
         end
   }

   ZStorage.resize = argcheck{
      {name="self", type=typename},
      {name="size", type="number"},
      call =
         function(self, size)
            THZStorage_resize(self, size)
            return self
         end
   }

   ZStorage.rawCopy = argcheck{
      {name="self", type=typename},
      {name="data", type="cdata"},
      call =
         function(self, data)
            ffi.copy(self.__data, data, ffi.sizeof(ctype)*self.__size)
            return self
         end
   }

   ZStorage.copy = argcheck{
      {name="self", type=typename},
      {name="src", type="torch.ZFloatStorage"},
      call =
         function(dst, src)
            THZStorage_copyZFloat(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      {name="self", type=typename},
      {name="src", type="torch.ZDoubleStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            THZStorage_copyZDouble(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.ByteStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyByte(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.CharStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyChar(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.ShortStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyShort(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.IntStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyInt(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.LongStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyLong(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.FloatStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyFloat(dst, src)
            return dst
         end
   }

   ZStorage.copy = argcheck{
      name = "copy",
      {name="dst", type=typename},
      {name="src", type="torch.DoubleStorage"},
      overload=ZStorage.copy,
      call =
         function(dst, src)
            src=src:cdata()
            THZStorage_copyDouble(dst, src)
            return dst
         end
   }

   ZStorage.totable = argcheck{
      {name="self", type=typename},
      call =
         function(self)
            local tbl = {}
            for i=1,tonumber(self.__size) do
               tbl[i] = {self.__data[i-1].re, self.__data[i-1].im}
            end
            return tbl
         end
   }

   -- define index and newindex for dispatch
   function ZStorage:__index(k)
      assert(self, 'Need to call as method')
      if type(k) == 'number' then
         if k > 0 and k <= tonumber(self.__size) then
            return self.__data[k-1]
         else
            error('index out of bounds')
         end
      else
         return rawget(ZStorage, k)
      end
   end

   function ZStorage:__newindex(k, v)
      assert(self, 'Need to call as method')
      if type(k) == 'number' then
         if k > 0 and k <= tonumber(self.__size) then
            self.__data[k-1] = v
         else
            error('index out of bounds')
         end
      else
         rawset(ZStorage, k, v)
      end
   end

   function ZStorage:__len()
      return self:size()
   end


   function ZStorage:__pairs()
      return pairs(ZStorage)
   end

   ZStorage.__tostring = display.storage

   ZStorage.__factory =
      function(file)
         return ZStorage.__new()
      end

   if Real == 'Float' then
      function ZStorage:__write(file)
         file:writeLong(self:size())
         for i=1,self:size() do
            local d = self[i]
            file:writeFloat(d.re)
            file:writeFloat(d.im)
         end
      end

      function ZStorage:__read(file)
         local size = file:readLong()
         self:resize(size)
         for i=1,self:size() do
            local re,im
            re = file:readFloat()
            im = file:readFloat()
            self[i] = ffi.new('complex', re, im)
         end
      end
   else
      function ZStorage:__write(file)
         file:writeLong(self:size())
         for i=1,self:size() do
            local d = self[i]
            file:writeDouble(d.re)
            file:writeDouble(d.im)
         end
      end

      function ZStorage:__read(file)
         local size = file:readLong()
         self:resize(size)
         for i=1,self:size() do
            local re,im
            re = file:readDouble()
            im = file:readDouble()
            self[i] = ffi.new('complex', re, im)
         end
      end
   end

   ZStorage.__version = 0
   torch.metatype(typename, ZStorage, THZStorage .. '&')
   ffi.metatype(THZStorage, ZStorage)

   -- constructor metatable
   local ZStorage_ctr = {}
   setmetatable(ZStorage_ctr, {
                   __call =
                      function(self, ...)
                         return ZStorage.__new(...)
                      end,
                   __index = ZStorage,
                   __newindex = ZStorage,
                   __len = ZStorage,
   })
   torch['Z' .. Real .. 'Storage'] = ZStorage_ctr

end
