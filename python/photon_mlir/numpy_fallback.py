"""
Numpy fallback for environments without numpy.
Provides minimal compatibility layer.
"""

class NumpyFallback:
    """Minimal numpy-like interface for basic operations."""
    
    def __init__(self):
        self.random = self._RandomSubmodule()
        self.ndarray = list  # Use list as ndarray fallback
    
    @staticmethod
    def array(data):
        """Create array-like object."""
        return data
    
    @staticmethod
    def zeros(shape):
        """Create zeros array."""
        if isinstance(shape, (list, tuple)):
            if len(shape) == 1:
                return [0.0] * shape[0]
            elif len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0]
    
    @staticmethod
    def ones(shape):
        """Create ones array."""
        if isinstance(shape, (list, tuple)):
            if len(shape) == 1:
                return [1.0] * shape[0]
            elif len(shape) == 2:
                return [[1.0] * shape[1] for _ in range(shape[0])]
        return [1.0]
    
    @staticmethod
    def mean(data):
        """Calculate mean."""
        if isinstance(data, (list, tuple)) and data:
            return sum(data) / len(data)
        return 0.0
    
    @staticmethod  
    def std(data):
        """Calculate standard deviation."""
        if isinstance(data, (list, tuple)) and len(data) > 1:
            mean_val = NumpyFallback.mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        return 0.0
    
    @staticmethod
    def percentile(data, percentile):
        """Calculate percentile."""
        if isinstance(data, (list, tuple)) and data:
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * percentile / 100
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        return 0.0
    
    @staticmethod
    def isfinite(data):
        """Check if values are finite."""
        if isinstance(data, (list, tuple)):
            return [not (x != x or x == float('inf') or x == float('-inf')) for x in data]
        return not (data != data or data == float('inf') or data == float('-inf'))
    
    @staticmethod
    def isinf(data):
        """Check if values are infinite."""
        if isinstance(data, (list, tuple)):
            return [x == float('inf') or x == float('-inf') for x in data]
        return data == float('inf') or data == float('-inf')
        
    @staticmethod
    def exp(data):
        """Exponential function."""
        import math
        if isinstance(data, (list, tuple)):
            return [math.exp(x) for x in data]
        return math.exp(data)
    
    @staticmethod
    def log10(data):
        """Log base 10 function."""
        import math
        if isinstance(data, (list, tuple)):
            return [math.log10(x) if x > 0 else float('-inf') for x in data]
        return math.log10(data) if data > 0 else float('-inf')
    
    @staticmethod
    def sqrt(data):
        """Square root function."""
        import math
        if isinstance(data, (list, tuple)):
            return [math.sqrt(x) if x >= 0 else float('nan') for x in data]
        return math.sqrt(data) if data >= 0 else float('nan')
    
    @staticmethod
    def polyfit(x, y, deg):
        """Polynomial fit (simplified)."""
        # Simple linear fit for deg=1
        if deg == 1 and len(x) == len(y) and len(x) >= 2:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            return [slope, intercept]
        return [0.0, 0.0]
    
    @staticmethod
    def ceil(data):
        """Ceiling function."""
        import math
        if isinstance(data, (list, tuple)):
            return [math.ceil(x) for x in data]
        return math.ceil(data)
    
    @staticmethod
    def random_normal(shape):
        """Create random normal array (simplified)."""
        import random
        if isinstance(shape, (list, tuple)):
            if len(shape) == 1:
                return [random.gauss(0, 1) for _ in range(shape[0])]
            elif len(shape) == 2:
                return [[random.gauss(0, 1) for _ in range(shape[1])] 
                       for _ in range(shape[0])]
        return [random.gauss(0, 1)]
    
    class _RandomSubmodule:
        @staticmethod
        def uniform(low=0.0, high=1.0, size=None):
            import random
            if size is None:
                return random.uniform(low, high)
            if isinstance(size, int):
                return [random.uniform(low, high) for _ in range(size)]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return [[random.uniform(low, high) for _ in range(size[1])] for _ in range(size[0])]
            return random.uniform(low, high)
            
        @staticmethod
        def choice(arr, size=None, replace=True):
            import random
            if size is None:
                return random.choice(arr)
            return [random.choice(arr) for _ in range(size)]
            
        @staticmethod
        def shuffle(arr):
            import random
            random.shuffle(arr)
            return arr  # Return the shuffled array
            
        @staticmethod
        def random(size=None):
            import random
            if size is None:
                return random.random()
            if isinstance(size, int):
                return [random.random() for _ in range(size)]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return [[random.random() for _ in range(size[1])] for _ in range(size[0])]
            return random.random()
            
        @staticmethod
        def normal(loc=0.0, scale=1.0, size=None):
            import random
            if size is None:
                return random.gauss(loc, scale)
            if isinstance(size, int):
                return [random.gauss(loc, scale) for _ in range(size)]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return [[random.gauss(loc, scale) for _ in range(size[1])] for _ in range(size[0])]
            return random.gauss(loc, scale)
            
        @staticmethod
        def exponential(scale=1.0, size=None):
            import random
            import math
            if size is None:
                return -scale * math.log(random.random())
            if isinstance(size, int):
                return [-scale * math.log(random.random()) for _ in range(size)]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return [[-scale * math.log(random.random()) for _ in range(size[1])] for _ in range(size[0])]
            return -scale * math.log(random.random())
    
    dtype = object  # Placeholder for dtype
    float64 = 'float64'  # Add dtype constants
    
    class FInfo:
        """Floating point info class for compatibility."""
        def __init__(self, dtype='float64'):
            self.dtype = dtype
            if dtype in ['float64', 'double']:
                self.eps = 2.220446049250313e-16
            else:
                self.eps = 1.1920928955078125e-07
    
    @staticmethod
    def finfo(dtype='float64'):
        """Get floating point machine limits."""
        return NumpyFallback.FInfo(dtype)
    
    @staticmethod 
    def arange(start, stop=None, step=1):
        """Create array with evenly spaced values."""
        if stop is None:
            stop = start
            start = 0
        
        result = []
        current = start
        while current < stop:
            result.append(current)
            current += step
        
        return result

# Create numpy-like interface
def get_numpy():
    """Get numpy or fallback implementation."""
    try:
        import numpy as np
        return np
    except ImportError:
        return NumpyFallback()