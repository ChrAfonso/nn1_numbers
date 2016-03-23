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
        layer.biases[u] = math.random()
        layer.weights[u] = {}
      
        for u_in = 1,net.sizes[l-1] do
          layer.weights[u][u_in] = math.random()
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
--    print ("Activation of layer " .. (l-1) .. ": " .. v_to_string(a[l-1]))
--    print ("Trying to compute activation for layer " .. l .. "...")
    a[l] = mat_mult_mv(net.layers[l].weights, a[l-1])
  end
  return a[self.num_layers]
end


-- Math utility functions

function sigmoid(x)
  return (1 / (1 + math.exp(-x)))
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
  local res = 0
  for _,i in ipairs(v) do
    res = res + i
  end
  return res
end


-- Print utility functions

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

