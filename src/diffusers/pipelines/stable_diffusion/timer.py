from decimal import Decimal
from math import ceil, log10
from timeit import default_timer

class Timer(object):
    def __init__(self, name, print_results=True):
        self.elapsed = Decimal()
        self._name = name
        self._print_results = print_results
        self._start_time = None
        self._children = {}
        self._count = 0

    def reset(self):
        self.elapsed = Decimal()
        self._start_time = None
        self._children = {}
        self._count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if self._print_results:
            self.print_results()

    def child(self, name):
        try:
            return self._children[name]
        except KeyError:
            result = Timer(name, print_results=False)
            self._children[name] = result
            return result

    def start(self):
        self._count += 1
        self._start_time = self._get_time()

    def stop(self):
        self.elapsed += self._get_time() - self._start_time

    def print_results(self):
        print('-'*20)
        print("Total:")
        print(self._format_results())
        print('-'*20)
        print("Average:")
        print(self._average_results())

    def _format_results(self, indent='  '):
        children = self._children.values()
        elapsed = self.elapsed or sum(c.elapsed for c in children)
        result = '%24s - %.3fs' % (self._name, elapsed)
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(ceil(log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c.elapsed, reverse=True):
            lines = child._format_results(indent).split('\n')
            child_percent = child.elapsed / elapsed * 100
            lines[0] += ' (%d%%)' % child_percent
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = ('%dx ' % child._count).rjust(count_digits + 2)  + lines[0]
            for line in lines:
                result += '\n' + indent + line
        return result

    def _average_results(self, indent='  '):
        children = self._children.values()
        elapsed = self.elapsed or sum(c.elapsed for c in children)
        result = '%24s - %.1fms' % (self._name, elapsed * 1000 / max(1, self._count))
        for child in sorted(children, key=lambda c: c.elapsed / c._count, reverse=True):
            lines = child._average_results(indent).split('\n')
            for line in lines:
                result += '\n' + indent + line
        return result

    def _get_time(self):
        return Decimal(default_timer())
