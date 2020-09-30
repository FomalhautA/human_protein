import threading


class ThreadSafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by serializing call to the 'next'
    method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    :param f:
    :return:
    """
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g
