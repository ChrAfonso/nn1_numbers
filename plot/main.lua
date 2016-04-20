function love.draw()
  love.graphics.clear()

  local file = io.open("../correct_rates.csv", "r")
  io.input(file)
  local x = 0
  local yoff = 200
  local lasty = yoff
  for line in io.lines() do
    y = yoff - line * 10
    love.graphics.line(x,lasty,x+1,y)

    x = x + 1
    lasty = y
  end
  io.close(file)
end
