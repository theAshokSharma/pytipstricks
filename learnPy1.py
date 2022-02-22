grocery_list = []
def add_items(*args):
    grocery_list.extend(args)
    
add_items('apples', 'bananas','sugar', 'flower')

print(grocery_list)
# ['apples', 'bananas', 'sugar']