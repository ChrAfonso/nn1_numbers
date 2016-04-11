MNIST = {}

function MNIST.load(filename)
  print("MNIST file loading started...")
  local file = io.open(filename, "rb")
  io.input(file)
  
  local training_samples = 50000
  local validation_samples = 10000
  local test_samples = 10000

  -- TODO read
  local MAGIC_NUMBER = bytes(file:read(4))
  local NUM_ITEMS = bytes(file:read(4))
  local num_items = tonumber(NUM_ITEMS, 16)
  print("Magic number: " .. MAGIC_NUMBER)
  print("Num of items: " .. NUM_ITEMS .. " (" .. (num_items or "_") .. ")")

  local training_labels = {}
  for i = 1,training_samples do
    training_labels[i] = file:read(1)
    --print("  Training label " .. i .. ": " .. bytes(training_labels[i]))
  end
  print("Read in " .. training_samples .. " training samples")
  
  local validation_labels = {}
  for i = training_samples+1,num_items do
    validation_labels[i] = file:read(1)
  end
  print("Read in " .. validation_samples .. " validation labels")

  file:close()
  return -- TODO
end

-- util
function bytes(str, sep)
  sep = sep or ""
  local ret = ""
  for i=1,#str do
    ret = ret .. string.format("%02X", str:byte(i))
    if i < #str then ret = ret .. sep end
  end
  return ret
end
