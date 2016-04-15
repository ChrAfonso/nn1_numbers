MNIST = {}

function MNIST.load(training_label_file, training_image_file, test_label_file, test_image_file)
  local training_labels = MNIST.load_file(training_label_file, "training", 50000)
  assert(#training_labels == 50000)
  local validation_labels = MNIST.load_file(training_label_file, "validation", 10000, 50000)
  assert(#validation_labels == 10000)
  local test_labels = MNIST.load_file(test_label_file, "test", 10000)
  assert(#test_labels == 10000)
  
  local training_images = MNIST.load_file(training_image_file, "training", 50000)
  assert(#training_images == 50000)
  local validation_images = MNIST.load_file(training_image_file, "validation", 10000, 50000)
  assert(#validation_images == 10000)
  local test_images = MNIST.load_file(test_image_file, "test", 10000)
  assert(#test_images == 10000)

  return training_labels, validation_labels, test_labels, training_images, validation_images, test_images
end

function MNIST.load_file(filename, datatype, num, offset)
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
  
  if magic_number == 2049 then
    return read_labels(file, offset, num, datatype)
  elseif magic_number == 2051 then
    return read_images(file, offset, num, datatype)
  else
    print("Invalid magic number " .. magic_number);
    return
  end
end

function read_labels(file, offset, num, datatype)
  local labels = {}
  for i = 1,offset+num do
    if i >= offset then
      if datatype == "training" then
        labels[i-offset] = encode_label(file:read(1))
      else
        labels[i-offset] = file:read(1):byte(1) + 1
        --print("  label " .. i .. ": " .. labels[i-offset])
      end
    end
  end
  print("Read in " .. num .. " " .. datatype .. " labels")

  file:close()
  return labels
end

function encode_label(s)
  local label = s:byte(1)
  local ret = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  ret[label+1] = 1
  return ret
end

function read_images(file, offset, num, datatype)
  local ROWS = bytes(file:read(4))
  local rows = tonumber(ROWS, 16)
  local COLS = bytes(file:read(4))
  local cols = tonumber(COLS, 16)
  print("  ROWS: " .. ROWS .. " (" .. (rows or "_") .. ")")
  print("  COLS: " .. COLS .. " (" .. (cols or "_") .. ")")
  if not rows or not cols then
    print("Invalid dimensions.")
    return nil
  end

  local start_time = os.clock()
  local images = {}
  for i = 1,offset+num do
    if i >= offset then
      images[i-offset] = read_image_fast(file, rows, cols)
    end
  end
  local end_time = os.clock()
  local time_diff = end_time - start_time
  print("Read in " .. num .. " " .. datatype .. " images (Took " .. time_diff .. ")")

  file:close()
  return images
end

function read_image(file, rows, cols)
  local image = { rows = rows, cols = cols }
  for i = 1,rows*cols do
    image[i] = file:read(1):byte(1)
  end
  return image
end

function read_image_fast(file, rows, cols)
  local image = { rows = rows, cols = cols }
  local str = file:read(rows*cols)
  for i = 1,rows*cols do
    image[i] = str:byte(i)
  end
  return image
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
