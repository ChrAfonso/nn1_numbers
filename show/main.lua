package.path = package.path .. '../?.lua'

require "MNIST"

function love.load()
  i = 1
  _, _, _, t_images, _, _ = MNIST.load("../train-labels.idx1-ubyte","../train-images.idx3-ubyte","../t10k-labels.idx1-ubyte","../t10k-images.idx3-ubyte")
end

function love.draw()
  local image = t_images[i]
  for y = 1,image.cols do
    for x = 1,image.rows do
      local c = 255 - image[(y-1)*image.cols + x]
      love.graphics.setColor(c, c, c)
      love.graphics.points(x, y)
    end
  end

  i = i + 1
  if i > #t_images then i = 1 end
end
