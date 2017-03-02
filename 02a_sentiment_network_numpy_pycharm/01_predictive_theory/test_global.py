value = 42

def doit():
    print(value)
    #UnboundLocalError: local variable 'value' referenced before assignment
    value = 0

doit()
print(value)