from math import pi

for i in range(20):
    line = input()
    temp = line[1:-1].split(", ")
    temp2 = []
    for j in temp:
        angle = (float(j) * 180) / (2 * pi)
        print(int(angle), end=" ")
    print()