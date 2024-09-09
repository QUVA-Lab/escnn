import unittest
from unittest import TestCase

from escnn.singleton import Singleton, SingletonABC, SingletonError
from abc import abstractmethod
import pickle
import copy


class TestSingleton(TestCase):

    def test_no_init(self):

        class A(Singleton):
            pass

        a1 = A()
        a2 = A()
        self.assertIs(a1, a2)

    def test_no_args(self):

        class A(Singleton):

            def __init__(self):
                pass

        a1 = A()
        a2 = A()
        self.assertIs(a1, a2)

    def test_required_args(self):

        class A(Singleton):

            def __init__(self, x, y):
                self.x = x
                self.y = y

        a = A(1, 2)

        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)

        self.assertSame(a, A(1, 2))
        self.assertSame(a, A(1, y=2))
        self.assertSame(a, A(x=1, y=2))
        self.assertSame(a, A(y=2, x=1))

        self.assertNotSame(a, A(2, 2))
        self.assertNotSame(a, A(2, y=2))
        self.assertNotSame(a, A(x=2, y=2))

        self.assertNotSame(a, A(1, 3))
        self.assertNotSame(a, A(1, y=3))
        self.assertNotSame(a, A(x=1, y=3))

    def test_optional_args(self):

        class A(Singleton):

            def __init__(self, x=1, y=2):
                self.x = x
                self.y = y

        a = A()

        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)

        self.assertSame(a, A())
        self.assertSame(a, A(1))
        self.assertSame(a, A(x=1))
        self.assertSame(a, A(y=2))
        self.assertSame(a, A(1, 2))
        self.assertSame(a, A(1, y=2))
        self.assertSame(a, A(x=1, y=2))

        self.assertNotSame(a, A(2))
        self.assertNotSame(a, A(x=2))
        self.assertNotSame(a, A(y=3))
        self.assertNotSame(a, A(2, 2))
        self.assertNotSame(a, A(1, 3))
        self.assertNotSame(a, A(2, y=2))
        self.assertNotSame(a, A(1, y=3))
        self.assertNotSame(a, A(x=2, y=2))
        self.assertNotSame(a, A(x=1, y=3))

    def test_var_positional_args(self):

        with self.assertRaises(SingletonError) as err:

            class A(Singleton):
                def __init__(self, *x):
                    pass

        self.assertIn(
                "A.__init__(*x): cannot use variable positional arguments",
                str(err.exception),
        )

    def test_var_keyword_args(self):

        with self.assertRaises(SingletonError) as err:

            class A(Singleton):
                def __init__(self, **x):
                    pass

        self.assertIn(
                "A.__init__(**x): cannot use variable keyword arguments",
                str(err.exception),
        )

    def test_keyword_only_args(self):

        with self.assertRaises(SingletonError) as err:

            class A(Singleton):
                def __init__(self, *, x):
                    pass
            
        self.assertIn(
                "A.__init__(*, x): cannot use keyword-only arguments",
                str(err.exception),
        )

    def test_name_arg(self):

        class A(Singleton):

            def __init__(self, x, name):
                self.x = x
                self.name = name

        a1 = A(1, 'alice')

        self.assertEqual(a1.x, 1)
        self.assertEqual(a1.name, 'alice')

        self.assertSame(a1, A(1, 'alice'))
        self.assertSame(a1, A(1, 'bob'))
        self.assertNotSame(a1, A(2, 'alice'))

        # Make sure we only use the first name.
        a2 = A(1, 'bob')
        
        self.assertSame(a2, a1)
        self.assertEqual(a2.x, 1)
        self.assertEqual(a2.name, 'alice')

    def test_canonicalize_init_kwargs(self):

        class A(Singleton):

            def __init__(self, x, y=None):
                self.x = x
                self.y = y

            @staticmethod
            def _canonicalize_init_kwargs(kwargs):
                kwargs['y'] = kwargs['y'] or kwargs['x']
                return kwargs

        a = A(1)

        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 1)

        self.assertSame(a, A(1))
        self.assertSame(a, A(1, 1))
        self.assertSame(a, A(1, y=1))
        self.assertSame(a, A(x=1, y=1))

        self.assertNotSame(a, A(2))
        self.assertNotSame(a, A(1, 2))
        self.assertNotSame(a, A(2, y=1))
        self.assertNotSame(a, A(1, y=2))
        self.assertNotSame(a, A(x=2, y=1))
        self.assertNotSame(a, A(x=1, y=2))

    def test_inheritance_same_args(self):

        class A(Singleton):

            def __init__(self, x):
                self.x = x

        class B(A):
            pass

        a = A(1)
        b = B(2)

        self.assertEqual(a.x, 1)
        self.assertEqual(b.x, 2)

        self.assertNotSame(a, b)
        self.assertNotSame(a, B(1))
        self.assertNotSame(b, A(2))

        self.assertSame(a, A(1))
        self.assertSame(a, A(x=1))

        self.assertSame(b, B(2))
        self.assertSame(b, B(x=2))

    def test_inheritance_diff_args(self):

        class A(Singleton):

            def __init__(self, x):
                self.x = x

        class B(A):

            def __init__(self, x, y):
                self.x = x
                self.y = y

        a = A(1)
        b = B(2, 3)

        self.assertEqual(a.x, 1)

        with self.assertRaises(AttributeError):
            a.y

        self.assertEqual(b.x, 2)
        self.assertEqual(b.y, 3)

        self.assertNotSame(a, b)
        self.assertNotSame(a, B(1, 1))

        self.assertSame(a, A(1))
        self.assertSame(a, A(x=1))

        self.assertSame(b, B(2, 3))
        self.assertSame(b, B(2, y=3))
        self.assertSame(b, B(x=2, y=3))

    def test_dict_key(self):

        class A(Singleton):

            def __init__(self, x):
                self.x = x

        d = {A(1): 3, A(2): 4}

        self.assertEqual(d[A(1)], 3)
        self.assertEqual(d[A(2)], 4)

    def test_pickle(self):
        a1 = Picklable(1)

        self.assertEqual(a1.x, 1)
        self.assertSame(a1, Picklable(1))

        a2 = pickle.loads(pickle.dumps(a1))

        self.assertEqual(a2.x, 1)
        self.assertSame(a1, a2)

        b = pickle.loads(pickle.dumps(Picklable(2)))

        self.assertEqual(b.x, 2)

        self.assertNotSame(a1, b)
        self.assertNotSame(a2, b)

    def test_pickle_name(self):
        a1 = PicklableWithName(1, 'alice')

        self.assertEqual(a1.x, 1)
        self.assertEqual(a1.name, 'alice')

        a2 = pickle.loads(pickle.dumps(a1))
        
        self.assertSame(a2, a1)
        self.assertEqual(a2.x, 1)
        self.assertEqual(a2.name, 'alice')

    def test_copy(self):
        # Doesn't really make sense to try to copy a singleton, but really I'm 
        # just checking that there's no easy way to break the singleton system.

        class A(Singleton):

            def __init__(self, x):
                self.x = x

        a = A(1)

        self.assertSame(a, copy.copy(a))
        self.assertSame(a, copy.deepcopy(a))

    def test_abc(self):

        class A(SingletonABC):

            def __init__(self, x):
                self.x = x

            @abstractmethod
            def do_something(self):
                pass

        class B(A):

            def do_something(self):
                return 2

        class C(A):

            def do_something(self):
                return 3

        class D(A):
            pass  # Don't implement `do_something()`.

        with self.assertRaises(TypeError):
            A(1)
        with self.assertRaises(TypeError):
            D(1)

        b = B(1)
        c = C(2)

        self.assertEqual(b.x, 1)
        self.assertEqual(b.do_something(), 2)

        self.assertEqual(c.x, 2)
        self.assertEqual(c.do_something(), 3)

        self.assertSame(b, B(1))
        self.assertSame(b, B(x=1))

        self.assertSame(c, C(2))
        self.assertSame(c, C(x=2))

        self.assertNotSame(b, c)
        self.assertNotSame(b, C(1))
        self.assertNotSame(c, B(2))


    def assertSame(self, a, b):
        self.assertIs(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def assertNotSame(self, a, b):
        self.assertIsNot(a, b)
        self.assertNotEqual(a, b)


class Picklable(Singleton):

    def __init__(self, x):
        self.x = x

class PicklableWithName(Singleton):

    def __init__(self, x, name):
        self.x = x
        self.name = name


if __name__ == '__main__':
    unittest.main()
