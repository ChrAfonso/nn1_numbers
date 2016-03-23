require "Network"

-- Test

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
  assert(#output == #net.sizes[net.num_layers])

  print ("Input: " .. v_to_string(input))
  print ("Output: " .. v_to_string(output))
end

net = Network.new{2,4,3}
testNetworkSetup(net, {2,4,3})
testCompute(net, {1,1})
testCompute(net, {2,2})
testCompute(net, {0,0})
