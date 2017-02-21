def not_bit(x1):
    if x1 == 0:
        return 1
    else:
        return 0


def a(x1):
    return not_bit(x1)


def b(x1, x2):
    return x1 & x2


def c(x1, x2):
    return x1 | x2


def network(x1, x2):
    bVal = b(x1, not_bit(x2))
    andVal = x2 & a(x1)
    cVal = c(bVal, andVal)
    return cVal


#print (a(0))
#print (b(0,0))
#print (c(0,0))
print (network(0, 0))
print (network(1, 0))
print (network(0, 1))
print (network(1, 1))