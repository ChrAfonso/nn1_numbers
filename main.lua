require "Network"

-- Tests

function testMath()
  local arr1 = {1,2,3}
  local arr2 = {4,5,6}
  local arr3 = map(function(a) return a*a end, arr1)
  assert(v_to_string(arr3) == "(1,4,9,)")
  local arr4 = hadamard(arr2, arr3)
  assert(v_to_string(arr4) == "(4,20,54,)")
  local sum4 = sum(arr4)
  assert(sum4 == 78)
  local tostring = reduce(function(a,b) return a .. " " .. b end, arr4)
  assert(tostring == "4 20 54")
  
  print ("Testing matrix operations...")
  local mats = {}
  mats[1] = {}
  mats[2] = {0}
  mats[3] = {0, 1, 2}
  mats[4] = {0, {1, 2, 3}, {4, 5}}
  mats[5] = {0, {1, 2, 3}, {{4, 5}, {6, 7}}}
  
  for i = 1, #mats do
    print("Matrix " .. i .. ": ")
    print(v_to_string(mats[i]))
    
    print("Matrix " .. i .. " desparsed: ")
    print(v_to_string(desparse(mats[i])))

    print("Matrix " .. i .. " transposed: ")
    print(v_to_string(transpose(mats[i])))
  end

  print("Testing matrix zero-duplication...")
  for i = 1, #mats do
    print("Matrix " .. i .. ": ")
    print(v_to_string(zero_mat_from_shape(mats[i])))
  end
end

function testNetworkSetup(net, sizes)
  print ("Network layers: " .. #net.sizes)
  assert(#net.sizes == #sizes, "Net layer count wrong (should be #sizes, is: " .. #net.sizes .. ")")
  for i,s in ipairs(net.sizes) do
    print ("Units in layer " .. i .. ": " .. s)
    assert(s == sizes[i], "Net size " .. i .. " wrong (should be: " .. sizes[i] .. ", is: " .. s .. ")")
  end
end

function testCompute(net, input)
  local output = net:compute(input)
  assert(#output == net.sizes[net.num_layers])

  print ("Input: " .. v_to_string(input))
  print ("Output: " .. v_to_string(output))
  print ("")

  net:backprop(input, {1,1,1})
end

function simpleTestSGD()
  training = {
    {x={0}, y={0}},
    {x={0}, y={0}},
    {x={1}, y={1}},
    {x={1}, y={1}}
  }

  net = Network.new({1,1})
  print("=== Testing simple SGD ===")
  net:SGD(training, 1000, 1, 0.03, training)
end

function testSGD()
  training = {
    {x={0, 0, 0, 0},y={1,0,0,0,0,0,0,0,0,0}},
    {x={0, 0, 0, 1},y={0,1,0,0,0,0,0,0,0,0}},
    {x={0, 0, 1, 0},y={0,0,1,0,0,0,0,0,0,0}},
    {x={0, 0, 1, 1},y={0,0,0,1,0,0,0,0,0,0}},
    {x={0, 1, 0, 0},y={0,0,0,0,1,0,0,0,0,0}},
    {x={0, 1, 0, 1},y={0,0,0,0,0,1,0,0,0,0}},
    {x={0, 1, 1, 0},y={0,0,0,0,0,0,1,0,0,0}},
    {x={0, 1, 1, 1},y={0,0,0,0,0,0,0,1,0,0}},
    {x={1, 0, 0, 0},y={0,0,0,0,0,0,0,0,1,0}},
    {x={1, 0, 0, 1},y={0,0,0,0,0,0,0,0,0,1}},

    {x={1, 0, 1, 0},y={1,0,0,0,0,0,0,0,0,0}},
    {x={1, 0, 1, 1},y={0,1,0,0,0,0,0,0,0,0}},
    {x={1, 1, 0, 0},y={0,0,1,0,0,0,0,0,0,0}},
    {x={1, 1, 0, 1},y={0,0,0,1,0,0,0,0,0,0}},
    {x={1, 1, 1, 0},y={0,0,0,0,1,0,0,0,0,0}},
    {x={1, 1, 1, 1},y={0,0,0,0,0,1,0,0,0,0}}
  }

  -- debug: overfitted
  test = {
    {x={0, 0, 0, 0},y={1,0,0,0,0,0,0,0,0,0}},
    {x={0, 0, 0, 1},y={0,1,0,0,0,0,0,0,0,0}},
    {x={0, 0, 1, 0},y={0,0,1,0,0,0,0,0,0,0}},
    {x={0, 0, 1, 1},y={0,0,0,1,0,0,0,0,0,0}},
    {x={0, 1, 0, 0},y={0,0,0,0,1,0,0,0,0,0}},
    {x={0, 1, 0, 1},y={0,0,0,0,0,1,0,0,0,0}},
    {x={0, 1, 1, 0},y={0,0,0,0,0,0,1,0,0,0}},
    {x={0, 1, 1, 1},y={0,0,0,0,0,0,0,1,0,0}},
    {x={1, 0, 0, 0},y={0,0,0,0,0,0,0,0,1,0}},
    {x={1, 0, 0, 1},y={0,0,0,0,0,0,0,0,0,1}},
    
    {x={1, 0, 1, 0},y={1,0,0,0,0,0,0,0,0,0}},
    {x={1, 0, 1, 1},y={0,1,0,0,0,0,0,0,0,0}},
    {x={1, 1, 0, 0},y={0,0,1,0,0,0,0,0,0,0}},
    {x={1, 1, 0, 1},y={0,0,0,1,0,0,0,0,0,0}},
    {x={1, 1, 1, 0},y={0,0,0,0,1,0,0,0,0,0}},
    {x={1, 1, 1, 1},y={0,0,0,0,0,1,0,0,0,0}}
  }

  net = Network.new({4, 30, 10})
  print("=== Testing SGD ===")
  net:SGD(training, 500, 5, 0.01, test)
end

sizes = {2,4,3}
net = Network.new(sizes)
print "=== Starting Network tests ==="
testMath()
testNetworkSetup(net, sizes)
testCompute(net, {1,1})
testCompute(net, {2,2})
testCompute(net, {0,0})

simpleTestSGD()
--testSGD()
