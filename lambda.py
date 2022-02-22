import dis

def func(x):
    return lambda y: (x + y + 1)

def func1(x):
    return lambda y : (func(x)(x)+y+1)

print(func1(10)(2))

dis.dis(func1)