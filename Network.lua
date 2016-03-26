Network = {}
Network.__index = Network

function Network.new(sizes)
  local net = {}
  setmetatable(net, Network)
  
  math.randomseed(os.time())
  
  net.sizes = sizes
  net.num_layers = #sizes
  net.layers = {}
  for l = 1,net.num_layers do
    local layer = {}
    net.layers[l] = layer
    
    layer.biases = {}
    layer.weights = {}
    for u = 1, net.sizes[l] do
      -- biases and incoming weights, only from second layer onwards
      if l > 1 then
        layer.biases[u] = gauss(0, 1)
        layer.weights[u] = {}
      
        for u_in = 1,net.sizes[l-1] do
          layer.weights[u][u_in] = gauss(0, 1)
        end
      end
    end
  end
  
  return net
end

function Network:classify(input)
  local output = self:compute(input)
  local maxindex = 1
  for i = 1,#output do
    if output[i] > output[maxindex] then
      maxindex = i
    end
  end
  return maxindex
end

function Network:compute(input)
  assert(#input == self.sizes[1])
  
  local a = {input}
  local z = {{}}
  for l = 2,self.num_layers do
    z[l] = mat_mult_mv(self.layers[l].weights, a[l-1])
    a[l] = map(sigmoid, z[l]) -- TODO: customize activation funcs?
  end
  return a[self.num_layers], a, z
end

function Network:backprop(input, y)
  local _, a, z = self:compute(input)
  local nabla_b, nabla_w = {{}}, {{}} -- first layer is input layer, does not have w/b
  local err
  for l = self.num_layers,2,-1 do
    nabla_b[l] = zero_mat_from_shape(self.layers[l].biases)
    nabla_w[l] = zero_mat_from_shape(self.layers[l].weights)
    
    if l == self.num_layers then
      err = error_L(z[l], y)
    else
      err = self:error_l(l, err, z[l])
    end
    nabla_b[l] = err
    nabla_w[l] = map(function(e) return v_mult_s(a[l-1], e) end, err)
  end
  print("nabla_b: " .. v_to_string(nabla_b))
  print("nabla_w: " .. v_to_string(nabla_w))
  
  return nabla_b, nabla_w
end

-- training_data and test_data contain tuples of the form { x = {...}, y = {...}}
function Network:SGD(training_data, epochs, mini_batch_size, eta, test_data)
  local n = #training_data
  math.randomseed(os.time())
  local mini_batch
  for i = 1,epochs do
    print ("Training epoch " .. i)
    shuffle(training_data)
    for m = 1,n,mini_batch_size do
      mini_batch = {}
      for b = 0,mini_batch_size do
        if (m+b > n) then break end
        mini_batch[b+1] = training_data[m+b]
      end
      self:update_mini_batch(mini_batch, eta)
    end
    
    if test_data then
      -- TODO test
    end
  end
end

function Network:update_mini_batch(mini_batch, eta)
  -- TODO
  -- for input in mini_batch: backprop and get nabla_w/b. Accumulate them. Update weights
  for _,input in mini_batch do
    self:backprop(input.x, input.y)
  end
end

function Network:error_l(l, error_next, z_l)
  assert(l < self.num_layers, "l can't be the last layer!")
  
  return hadamard(mat_mult_mv(transpose(self.layers[l+1].weights), error_next), map(sigmoid_prime, z_l))
end

function error_L(z, y)
  return map(error, z, y)
end

function error(zj, yj)
  return cost(zj, yj) * sigmoid_prime(zj)
end

function cost(zj, yj)
  return (sigmoid(zj) - yj)
end

-- Math utility functions

function sigmoid(x)
  return (1 / (1 + math.exp(-x)))
end

function sigmoid_prime(x)
  return sigmoid(x) * (1 - sigmoid(x))
end

function dot(u,v)
  assert(#u == #v)
  return sum(hadamard(u,v))
end

function length(v)
  return math.sqrt(dot(v,v))
end

function hadamard(u,v)
  assert(#u == #v)
  local res = {}
  for i = 1,#u do
    res[i] = (u[i] * v[i])
  end
  return res
end

function v_mult_s(v, s)
  return map(function(e) return e*s end, v)
end

function mat_mult_mv(mat, v)
  assert(#mat > 0 and #mat[1] == #v)
  local res = {}
  for i = 1,#mat do
    res[i] = sum(hadamard(mat[i], v))
  end
  return res
end

function mat_add(m, n)
  assert(#m > 0 and #m == #n and #m[1] == #n[1])
  local res = {}
  for i = 1, #m do
    res[i] = {}
    for j = 1,#n do
      res[i][j] = m[i][j] + n[i][j]
    end
  end
  return res
end

function transpose(mat)
  local dimj = #mat
  local dimi = #mat[1]
  local res = {}
    for i = 1,dimi do
      res[i] = {}
      for j = 1,dimj do
        res[i][j] = mat[j][i]
      end
    end
  return res
end

function sum(v)
  return reduce((function(a,b) return a+b end), v)
end

function map(func, arr, arr2)
  if arr2 then assert(#arr == #arr2) end
  local res = {}
  local v2
  for i,v in pairs(arr) do
    if arr2 then v2 = arr2[i] end
    res[i] = func(v, v2)
  end
  return res
end

function reduce(func, arr)
  local res = arr[1] or 0
  for i = 2,#arr do
    res = func(res, arr[i])
  end
  return res
end

function zero_mat_from_shape(mat)
  if type(mat) == "number" then
    return 0
  else
    local res = {}
    for i,v in ipairs(mat) do
      res[i] = zero_mat_from_shape(v)
    end
    return res
  end
end

-- Normal distributed random from http://osa1.net/posts/2012-12-19-different-distributions-from-uniform.html
function box_muller()
  return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) / 2
end

function gauss(mean, width)
  return mean + box_muller() * width
end
  

-- Utility functions

function shuffle(t)
  local it = #t
  local j
  for i = it, 2, -1 do
    j = math.random(i)
    t[i],t[j] = t[j],t[i]
  end
end

function v_to_string(v,sep)
  local sep = sep or ","
  local s = "("
  local i,e
  for i,e in pairs(v) do
    if type(e) == "table" then
      s = s .. v_to_string(e)
    else
      s = s .. e
    end
    s = s .. sep
  end
  s = s .. ")"
  return s
end

