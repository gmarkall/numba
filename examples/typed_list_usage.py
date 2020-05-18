# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.


def ex_inferred_list_jit():
    # magictoken.ex_inferred_list_jit.begin
    from numba import njit
    from numba.typed import List

    @njit
    def foo():
        # Instantiate a typed-list
        l = List()
        # Append a value to it, this will set the type to int32/int64
        # (depending on platform)
        l.append(42)
        # The usual list operations, getitem, pop and length are supported
        print(l[0])   # 42
        l[0] = 23
        print(l[0])   # 23
        print(len(l)) # 1
        l.pop()
        print(len(l)) # 0
        return l

    mylist = foo()

    # magictoken.ex_inferred_list_jit.end

    # Return mylist outside the example to avoid flake8 error
    return mylist


def ex_inferred_list():
    # magictoken.ex_inferred_list.begin
    from numba import njit
    from numba.typed import List

    @njit
    def foo(mylist):
        for i in range(10, 20):
            mylist.append()
        return mylist

    # Instantiate a typed-list, outside of a jit context
    l = List()
    # Append a value to it, this will set the type to int32/int64
    # (depending on platform)
    l.append(42)
    # The usual list operations, getitem, pop and length are supported
    print(l[0])   # 42
    l[0] = 23
    print(l[0])   # 23
    print(len(l)) # 1
    l.pop()
    print(len(l)) # 0

    # And you can use the typed-list as an argument for a jit compiled function

    l = foo(l)
    print(len(l)) # 10

    # magictoken.ex_inferred_list.end


def ex_nested_list():
    # magictoken.ex_nested_list.begin
    from numba.typed import List

    # typed-lists can be nested in typed-lists
    mylist = List()
    for i in range(10):
        l = List()
        for i in range(10):
            l.append(i)
        mylist.append(l)
    # mylist is now a list of 10 lists, each containing 10 integers
    print(mylist)

    # magictoken.ex_nested_list.end


def ex_empty_list():
    # magictoken.ex_empty_list.begin
    from numba.typed import List
    from numba import types, njit

    @njit
    def main(n):
        l = List()
        for _ in range(n):
            sublist = List.empty_list(types.int64)
            l.append(sublist)
        return l

    # Produces a list of five empty typed lists of int64 type
    print(main(5))

    # magictoken.ex_empty_list.end
