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

function Network:compute(input)
  assert(#input == self.sizes[1])
  
  local a = {input}
  for l = 2,self.num_layers do
    local raw_a = mat_mult_mv(net.layers[l].weights, a[l-1])
    a[l] = map(sigmoid, raw_a) -- TODO: customize activation funcs?
  end
  return a[self.num_layers]
end

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
  return sum(mat_mult_vv(u,v))
end

function length(v)
  return math.sqrt(dot(v,v))
end

function mat_mult_vv(u,v)
  assert(#u == #v)
  local res = {}
  for i = 1,#u do
    res[i] = (u[i] * v[i])
  end
  return res
end

function mat_mult_mv(mat, v)
  assert(#mat > 0 and #mat[1] == #v)
  local res = {}
  for i = 1,#mat do
    res[i] = sum(mat_mult_vv(mat[i], v))
  end
  return res
end

function sum(v)
  return reduce((function(a,b) return a+b end), v)
end

function map(func, arr)
  local res = {}
  for i,v in pairs(arr) do
    res[i] = func(v)
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

function zero_mat(dims)
  local mat = {}
  for i,v in ipairs(dims) do
    mat[i] = {}
    for z = 1,v do
      mat[i][z] = 0
    end
  end
  return mat
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
  sep = sep or ","
  local s = "("
  for i = 1,#v do
    if type(v[i]) == "table" then
      s = s .. v_to_string(v[i])
    else
      s = s .. v[i]
    end
    if i < #v then s = s .. sep end
  end
  s = s .. ")"
  return s
end

