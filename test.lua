--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.

local ztorch = require 'ztorch'
local cpx = ztorch.complex
local fcpx = ztorch.fcomplex
local z = ztorch

torch.manualSeed(1)
local mytester
local ztest = {}
local precision = 1e-4

function ztest.storageInit()
   local a=torch.ZFloatStorage()
   mytester:assert(torch.typename(a) == 'torch.ZFloatStorage')
end

function ztest.storageFill()
   local a=torch.ZFloatStorage(3)
   a:fill(2+z.im(3))
   for i=1,a:size() do
      mytester:assert(a[i] == 2+z.im(3), 'fill does not equate')
   end
end

function ztest.storageToTable()
   local a=torch.ZFloatStorage(3)
   a:fill(2+z.im(3))
   local t = a:totable()
   for i=1,a:size() do
      mytester:assert(t[i][1] == a[i].re)
      mytester:assert(t[i][2] == a[i].im)
   end
end

function ztest.storageNewIndex()
   local a=torch.ZFloatStorage(3)
   for i=1,a:size() do
      local n = torch.normal()
      local m = torch.normal()
      a[i] = n+m*z.im(1)
      mytester:assert(math.abs((a[i].re - n)) <= precision)
      mytester:assert(math.abs((a[i].im - m)) <= precision)
   end
end

function ztest.storageSerialize()
   local a=torch.ZFloatStorage(3)
   for i=1,a:size() do
      local n = torch.normal()
      local m = torch.normal()
      a[i] = n+m*z.im(1)

   end
   torch.save('tmp.t7',a)
   local b=torch.load('tmp.t7')
   for i=1,a:size() do
      mytester:assert(fcpx.abs(a[i] - b[i]) < precision)
   end
   os.execute('rm tmp.t7')
end

function ztest.storageSizeAndResize()
   local sz = math.floor(torch.uniform(2,100))
   local a=torch.ZFloatStorage(sz)
   mytester:assert(a:size() == sz)
   mytester:assert(#a == sz)

   local sz2 = math.floor(torch.uniform(2,100))
   a:resize(sz2)
   mytester:assert(a:size() == sz2)
   mytester:assert(#a == sz2)
end

-------------------------------------------------------
-- Tensor tests
------------------------------------------------------
-- create a tensor of randomized dimensions and sizes
local function createTensor()
   local ndim = torch.random(1,5)
   local sizes = {}
   for i=1,ndim do sizes[#sizes+1] = torch.random(1,5) end
   sizes = torch.LongStorage(sizes)
   local t = torch.ZFloatTensor(sizes)
   return t
end

function ztest.tensorInit()
   local a=torch.ZFloatTensor()
   mytester:assert(torch.typename(a) == 'torch.ZFloatTensor')

   local a=torch.ZFloatTensor(1)
   mytester:assert(torch.typename(a) == 'torch.ZFloatTensor')
   mytester:assert(torch.typename(a:storage()) == 'torch.ZFloatStorage')
   mytester:assert(a:storageOffset() == 1)

   local a=createTensor()
   mytester:assert(torch.typename(a) == 'torch.ZFloatTensor')
   mytester:assert(torch.typename(a:storage()) == 'torch.ZFloatStorage')
   mytester:assert(a:storageOffset() == 1)

end

function ztest.tensorFill()
   local a=torch.ZFloatTensor(3,4,5)
   a:fill(2+z.im(3))
   for i=1,a:size(1) do
      for j=1,a:size(2) do
         for k=1,a:size(3) do
            mytester:assert(a[i][j][k] == 2+z.im(3), 'fill does not equate')
         end
      end
   end
end

function ztest.tensorNewIndex()
   local a=torch.ZFloatTensor(3,4)
   for i=1,a:size(1) do
      for j=1,a:size(2) do
         local n = torch.normal()
         local m = torch.normal()
         a[i][j] = n+m*z.im(1)
         mytester:assert(math.abs((a[i][j].re - n)) <= precision)
         mytester:assert(math.abs((a[i][j].im - m)) <= precision)
      end
   end
end

function ztest.tensorIndexTable()
   local a=torch.ZFloatTensor(3,4,5):zero()
   a[1][2][3] = 2+z.im(3)
   mytester:assert(a[{1,2,3}] == 2+z.im(3), 'table newindex incorrect')

   a[2][3] = 1+z.im(2)
   mytester:assert((a[{2,3}] - a[2][3]):abs():max() < 1e-6, 'table index incorrect')
   mytester:assert(a[{2,3,1}] == 1+z.im(2), 'table index incorrect')
end

function ztest.tensorSerialize()
   local a=torch.ZFloatTensor(3,4,5)
   for i=1,a:size(1) do
      for j=1,a:size(2) do
         for k=1,a:size(3) do
            local n = torch.normal()
            local m = torch.normal()
            a[i][j][k] = n+m*z.im(1)
         end
      end
   end
   torch.save('tmp.t7',a)
   local b=torch.load('tmp.t7')
   for i=1,a:size(1) do
      for j=1,a:size(2) do
         for k=1,a:size(3) do
            mytester:assert(fcpx.abs(a[i][j][k] - b[i][j][k]) < precision)
         end
      end
   end
   os.execute('rm tmp.t7')
end

function ztest.tensorSizeAndResizeAndContiguous()
   local sz = math.floor(torch.uniform(2,100))
   local sz2 = math.floor(torch.uniform(2,100))
   local a=torch.ZFloatTensor(sz,sz2)
   mytester:assert(a:size(1) == sz)
   mytester:assert(a:size(2) == sz2)
   mytester:assert((#a)[1] == sz)
   mytester:assert((#a)[2] == sz2)
   mytester:assert(a:nDimension() == 2)
   mytester:assert(a:dim() == 2)
   mytester:assert(a:isContiguous())

   a=a:t()
   mytester:assert(not a:isContiguous())
   local b=a:contiguous()
   mytester:assert(b:isContiguous())

   local b=a:clone()
   mytester:assert(b:isContiguous())
   -- TODO: do (b-a):abs() assertions

   local sz2 = math.floor(torch.uniform(2,100))
   a:resize(sz2)
   mytester:assert(a:size(1) == sz2)
   mytester:assert((#a)[1] == sz2)

   local c=torch.ZFloatTensor()
   c:resize(torch.LongStorage({20}))
   mytester:assert(c:size(1) == 20)
end

-- operations: +,-
function ztest.sub()
   local s1 = 1 + z.im(3)
   local s2 = 2 + z.im(4)
   -- 1D Tensor + Tensor
   local a = torch.ZFloatTensor(10):fill(s1)
   local b = torch.ZFloatTensor(10):fill(s2)
   local c = a - b
   for i = 1, 10 do
      mytester:assert(c[i] == (s1 - s2))
   end

   -- 1D Tensor + number
   local c = a - s2
   for i = 1, 10 do
      mytester:assert(c[i] == (s1 - s2))
   end


end

function ztest.add()
   local s1 = 1 + z.im(3)
   local s2 = 2 + z.im(4)
   local a = torch.ZFloatTensor(10):fill(s1)
   local b = torch.ZFloatTensor(10):fill(s2)
   -- 1D Tensor + Tensor
   local c = a + b
   for i = 1, 10 do
      mytester:assert(c[i] == (s1 + s2))
   end

   -- 1D Tensor + number
   local c = a + s2
   for i = 1, 10 do
      mytester:assert(c[i] == (s1 + s2))
   end

end

-- conjugate
function ztest.conj()
   local t = createTensor():normal()
   local t2 = t:view(-1)
   local t2conj = t2:clone():conj()
   for i=1,t2:size(1) do
      assert(t2conj[i].re == fcpx.conj(t2[i]).re)
      assert(t2conj[i].im == fcpx.conj(t2[i]).im)
   end
end

-- * (component wise and dot product)
function ztest.dot()
   local sz = torch.random(1,30)
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local res = t * t2
   -- (1+2i)*conj((2-3i)) = 2 - 6 +3i + 4i = -4 + 7i
   assert(res.re == -4 * sz, 'expected ' .. -4*sz .. ' but found: ' .. res.re)
   assert(res.im == 7 * sz, 'expected ' .. 7*sz .. ' but found: ' .. res.im)

   local res = t:dot(t2)
   assert(res.re == -4 * sz, 'expected ' .. -4*sz .. ' but found: ' .. res.re)
   assert(res.im == 7 * sz, 'expected ' .. 7*sz .. ' but found: ' .. res.im)
end

function ztest.mv()
   local t = torch.ZFloatTensor(3, 2):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(2):fill(2-z.im(3))
   local res = t * t2
   local sz = t2:size(1)
   -- (1+2i)*(2-3i) = 2 + 6 - 3i + 4i = 8 + i
   for i=1,res:size(1) do
      assert(res[i].re == 8 * sz, 'expected ' .. 8*sz .. ' but found: ' .. res[i].re)
      assert(res[i].im == 1 * sz, 'expected ' .. 1*sz .. ' but found: ' .. res[i].im)
   end
end

function ztest.mm()
   local sz = 3
   local sz2 = 2
   local sz3 = 4
   local t = torch.ZFloatTensor(sz, sz2):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz2, sz3):fill(2-z.im(3))
   local res = t * t2
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == 8 * sz2, 'expected ' .. 8*sz2 .. ' but found: ' .. res[i].re)
      assert(res[i].im == 1 * sz2, 'expected ' .. 1*sz2 .. ' but found: ' .. res[i].im)
   end
end

function ztest.ger()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local res1 = (1+z.im(2))*z.complex.conj(2-z.im(3))
   local res = t:ger(t2)
   res = res:view(-1)
   mytester:asserteq(res:size(1), sz*sz, 'incorrect size')
   for i=1,res:size(1) do
      mytester:asserteq(res[i].re, res1.re, 're incorrect at ' .. tostring(i))
      mytester:asserteq(res[i].im, res1.im, 'im incorrect at ' .. tostring(i))
   end
end

function ztest.addmv()
   local t = torch.ZFloatTensor(3, 2):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(2):fill(2-z.im(3))
   local res = torch.ZFloatTensor(3):fill(5+z.im(3))
   res:addmv(1, t, t2)
   local sz = t2:size(1)
   -- 5+3i + (1+2i)*(2-3i) = 2 + 6 - 3i + 4i = 8 + i + 5 + 3i
   for i=1,res:size(1) do
      assert(res[i].re == 5 + 8 * sz, 'expected ' .. 5 + 8*sz .. ' but found: ' .. res[i].re)
      assert(res[i].im == 3 + 1 * sz, 'expected ' .. 3 + 1*sz .. ' but found: ' .. res[i].im)
   end
end

function ztest.addmm()
   local sz = 3
   local sz2 = 2
   local sz3 = 4
   local scale = 3
   local t = torch.ZFloatTensor(sz, sz2):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz2, sz3):fill(2-z.im(3))
   local res = torch.ZFloatTensor(sz, sz3):fill(5+z.im(3))
   res:addmm(scale, t, t2)
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == 5 + scale * 8 * sz2, 'expected ' .. 5+8*sz2 .. ' but found: ' .. res[i].re)
      assert(res[i].im == 3 + scale * 1 * sz2, 'expected ' .. 3+1*sz2 .. ' but found: ' .. res[i].im)
   end
end

function ztest.addmm_scale_dst()
   local sz = 3
   local sz2 = 2
   local sz3 = 4
   local scale = 3
   local t = torch.ZFloatTensor(sz, sz2):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz2, sz3):fill(2-z.im(3))
   local res = torch.ZFloatTensor(sz, sz3):fill(5+z.im(3))
   res:addmm(scale, res, t, t2)
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == scale * 5 + 8 * sz2, 'expected ' .. 5+8*sz2 .. ' but found: ' .. res[i].re)
      assert(res[i].im == scale * 3 + 1 * sz2, 'expected ' .. 3+1*sz2 .. ' but found: ' .. res[i].im)
   end
end

function ztest.addru()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local res = torch.ZFloatTensor(sz,sz):fill(5+z.im(3))
   res:addru(2.0, t, t2)
   local res1 = 2*(1+z.im(2))*(2-z.im(3)) + 5+z.im(3)
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == res1.re, 'expected ' .. res1.re .. ' but found: ' .. res[i].re)
      assert(res[i].im == res1.im, 'expected ' .. res1.im .. ' but found: ' .. res[i].im)
   end
end

function ztest.addr()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local res = torch.ZFloatTensor(sz,sz):fill(5+z.im(3))
   res:addr(2.0, t, t2)
   local res1 = 2*(1+z.im(2))*z.complex.conj(2-z.im(3)) + 5+z.im(3)
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == res1.re, 'expected ' .. res1.re .. ' but found: ' .. res[i].re)
      assert(res[i].im == res1.im, 'expected ' .. res1.im .. ' but found: ' .. res[i].im)
   end
end

function ztest.addr_four_args()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local res = torch.ZFloatTensor(sz,sz):fill(5+z.im(3))
   res:addr(3, 2, t, t2)
   local res1 = 2*(1+z.im(2))*z.complex.conj(2-z.im(3)) + 3*(5+z.im(3))
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == res1.re, 'expected ' .. res1.re .. ' but found: ' .. res[i].re)
      assert(res[i].im == res1.im, 'expected ' .. res1.im .. ' but found: ' .. res[i].im)
   end
end

function ztest.addr_five_args()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local mat = torch.ZFloatTensor(sz,sz):fill(5+z.im(3))
   local res = torch.ZFloatTensor(sz,sz)
   res:addr(3, mat, 2, t, t2)
   local res1 = 2*(1+z.im(2))*z.complex.conj(2-z.im(3)) + 3*(5+z.im(3))
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == res1.re, 'expected ' .. res1.re .. ' but found: ' .. res[i].re)
      assert(res[i].im == res1.im, 'expected ' .. res1.im .. ' but found: ' .. res[i].im)
   end
end

function ztest.addr_src()
   local sz = 4
   local t = torch.ZFloatTensor(sz):fill(1+z.im(2))
   local t2 = torch.ZFloatTensor(sz):fill(2-z.im(3))
   local src = torch.ZFloatTensor(sz,sz):fill(5+z.im(3))
   local res = torch.ZFloatTensor(sz,sz)
   res:addr(src, 2, t, t2)
   local res1 = 2*(1+z.im(2))*z.complex.conj(2-z.im(3)) + (5+z.im(3))
   res = res:view(-1)
   for i=1,res:size(1) do
      assert(res[i].re == res1.re, 'expected ' .. res1.re .. ' but found: ' .. res[i].re)
      assert(res[i].im == res1.im, 'expected ' .. res1.im .. ' but found: ' .. res[i].im)
   end
end

function ztest.re()
   local t = torch.ZFloatTensor(2)
   t[1] = 2 + z.im(3)
   t[2] = 5 - z.im(7)

   local res = t:re()
   mytester:assert(res[1] == 2, 'expected 2 but found ' .. res[1])
   mytester:assert(res[2] == 5, 'expected 5 but found ' .. res[2])

   res = torch.Tensor(2)
   res:re(t)
   mytester:assert(res[1] == 2, 'expected 2 but found ' .. res[1])
   mytester:assert(res[2] == 5, 'expected 5 but found ' .. res[2])

   res = torch.ZFloatTensor(2)
   res:re(t)
   mytester:assert(res[1].re == 2, 'expected 2 but found ' .. res[1].re)
   mytester:assert(res[2].re == 5, 'expected 5 but found ' .. res[2].re)
   mytester:assert(res[1].im == 0, 'expected 0 but found ' .. res[1].im)
   mytester:assert(res[2].im == 0, 'expected 0 but found ' .. res[2].im)
end

function ztest.im()
   local t = torch.ZFloatTensor(2)
   t[1] = 2 + z.im(3)
   t[2] = 5 - z.im(7)

   local res = t:im()
   mytester:assert(res[1] == 3, 'expected 3 but found ' .. res[1])
   mytester:assert(res[2] == -7, 'expected -7 but found ' .. res[2])

   res = torch.Tensor(2)
   res:im(t)
   mytester:assert(res[1] == 3, 'expected 3 but found ' .. res[1])
   mytester:assert(res[2] == -7, 'expected -7 but found ' .. res[2])

   res = torch.ZFloatTensor(2)
   res:im(t)
   mytester:assert(res[1].re == 3, 'expected 3 but found ' .. res[1].re)
   mytester:assert(res[2].re == -7, 'expected -7 but found ' .. res[2].re)
   mytester:assert(res[1].im == 0, 'expected 0 but found ' .. res[1].im)
   mytester:assert(res[2].im == 0, 'expected 0 but found ' .. res[2].im)
end

function ztest.abs()
   local t = torch.ZFloatTensor(2)
   t[1] = 2 + z.im(3)
   t[2] = 5 - z.im(7)

   local res = t:abs()
   mytester:assertlt((res[1] - math.sqrt(13)), 1e-4, 'expected sqrt(13) but found ' .. res[1])
   mytester:assertlt((res[2] - math.sqrt(74)), 1e-4, 'expected sqrt(74) but found ' .. res[2])
end

function ztest.nnLinear()
   require 'nn'
   local m = nn.Linear(10,20):type('torch.ZFloatTensor')
   m:reset()
   local input = torch.ZFloatTensor(10):normal()
   local output = m:forward(input)
   local gradients = torch.ZFloatTensor(20):normal()
   m:backward(input, gradients)
   -- print(o)
end

-- FB: hooks to work with our test runner
pcall(function ()
        require 'fb.luaunit'
        require 'fbtorch'
end)

mytester = torch.Tester()
mytester:add(ztest)
mytester:run()
