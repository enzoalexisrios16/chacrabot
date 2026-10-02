import random
c=0
for i in range(10000):
    x = random.randint(1,10)
    if x ==10:
        c=c+1
print(c)