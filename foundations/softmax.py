import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        shift_z = z - np.max(z)
        
        # 2. Compute the exponent of every element using NumPy's vectorized exp
        exponents = np.exp(shift_z)
            
        # 3. Divide each exponent by the sum of all exponents
        return np.round(exponents / np.sum(exponents), 4)