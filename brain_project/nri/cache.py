import collections
import numpy as np

"""
Least recently used cache for data files.

Saving the data we batch multiple samples in the same file, to reduce the
number of disk accesses necessary for reading it back. A cache is used
when reading to handle reading files, and loading samples from within
those files.
"""

class LRUCache:
    """LRU cache implementation
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                # Eliminate the least recently used item from the cache
                self.cache.popitem(last=False)
        self.cache[key] = value

    def load(self, file, index_in_file):
        """Load a file from cache or from disk, and a sample from that file.
        Args:
          file : str
            The full file path to be loaded. The file should be a `npy` file
            which will be loaded using `np.load`.
          index_in_file : int
            The index in the loaded numpy array to be returned.
        Returns:
          sample : array
        """
        try:
            arr = self.get(file)
        except KeyError:
            arr = np.load(file)

        # Need to call set each time so we know what the LRU item is.
        self.set(file, arr)

        return arr[index_in_file]
