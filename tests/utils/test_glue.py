from openfed.utils.glue import glue


def test_glue():
    class A():
        def __init__(self):
            self.name = dict(a_name="a_name")

        def get_a(self):
            return 'a'

        def put(self):
            return 'a'

        def get(self):
            return dict(a_get="a_get")

    class B():
        def __init__(self):
            self.name = dict(b_name='b_name')

        def get_b(self):
            return 'b'

        def put(self):
            return 'b'

        def get(self):
            return dict(b_get="b_get")

    a = A()
    b = B()

    c = glue(a, b, parall_func_list=['get'])

    print(c.get_a())
    print(c.get_b())
    print(c.name)
    print(c.get())
    print(c.put())
