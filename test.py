map = {}
map[1] = 1
map[1] = map.get(1, 1) + 1

print(map)