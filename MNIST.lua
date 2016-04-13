MNIST = {}

function MNIST.load(training_label_file, training_image_file, test_label_file, test_image_file)
  local training_labels = MNIST.load_file(training_label_file, "training", 50000)
  assert(#training_labels == 50000)
  local validation_labels = MNIST.load_file(training_label_file, "validation", 10000, 50000)
  assert(#validation_labels == 10000)
  local test_labels = MNIST.load_file(test_label_file, "test", 10000)
  assert(#test_labels == 10000)
end

function MNIST.load_file(filename, type, num, offset)
  offset = offset or 0

  print("MNIST file loading started...")
  local file = io.open(filename, "rb")
  io.input(file)

  -- read labels
  local MAGIC_NUMBER = bytes(file:read(4))
  local NUM_ITEMS = bytes(file:read(4))
  local magic_number = tonumber(MAGIC_NUMBER, 16)
  local num_items = tonumber(NUM_ITEMS, 16)
  print("Magic number: " .. MAGIC_NUMBER .. " (" .. (magic_number or "_") .. ")")
  print("Num of items: " .. NUM_ITEMS .. " (" .. (num_items or "_") .. ")")

  -- TODO generalize
  local labels = {}
  for i = 1,offset+num do
    if i >= offset then
      labels[i-offset] = file:read(1)
      --print("  label " .. i .. ": " .. bytes(training_labels[i]))
    end
  end
  print("Read in " .. num .. " " .. type .. " labels")

  file:close()
  return labels
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
