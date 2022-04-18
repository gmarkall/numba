import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestLaplace(CUDATestCase):
    """
    Test simple vector addition
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_laplace(self):

        # set True to regenerate the figures that
        # accompany this example
        plot = False

        # ex_laplace.import.begin
        import numpy as np
        from numba import cuda

        # ex_laplace.import.end
        # ex_laplace.allocate.begin
        # use an odd problem size
        # this is so there can be an element truly
        # in the "middle" for symmetry
        size = 1001
        data = np.zeros(size)

        # middle element is made very hot
        data[500] = 10000
        data_gpu = cuda.to_device(data)

        # this extra array is used in the algorithm for
        # synchronization purposes
        tmp_gpu = cuda.to_device(np.zeros(len(data)))

        niter = 10000
        # ex_laplace.allocate.end

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16*0.66, 9*0.66))
            plt.plot(
                np.arange(len(data_gpu)), 
                data_gpu.copy_to_host(), 
                lw=3, 
                marker="*", 
                color='black'
            )

            plt.title('Initial State', fontsize=24)
            plt.xlabel('Position', fontsize=24)
            plt.ylabel('Temperature', fontsize=24)

            ax.set_xticks(ax.get_xticks(), fontsize=16)
            ax.set_yticks(ax.get_yticks(), fontsize=16)
            plt.xlim(0, len(data))
            plt.ylim(0, 10001)
            plt.savefig('laplace_initial.svg')

        # ex_laplace.kernel.begin
        @cuda.jit
        def solve_heat_equation(data, tmp, size, timesteps, k):
            i = cuda.grid(1)

            # prepare to do a grid-wide synchronization later
            grid = cuda.cg.this_grid()

            for step in range(timesteps):
                # get the current temperature associated with this segment
                curr_temp = data[i]

                # apply formula from finite difference equation
                if i == 0:
                    # Left wall is held at T = 0
                    next_temp = curr_temp + k * (data[i + 1] - (2 * curr_temp))
                elif i == size - 1:
                    # Right wall is held at T = 0
                    next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp))
                else:
                    next_temp = curr_temp + k * (
                        data[i - 1] - (2 * curr_temp) + data[i + 1]
                    )
                tmp[i] = next_temp
                # wait for every thread to write before moving on
                grid.sync()

                # swap data vectors for the next iteration
                data[i] = tmp[i]

        # ex_laplace.kernel.end

        # ex_laplace.launch.begin
        solve_heat_equation.forall(len(data))(
            data_gpu, tmp_gpu, len(data_gpu), niter, 0.25
        )
        # ex_laplace.launch.end

        if plot:
            results = tmp_gpu.copy_to_host()

            fig, ax = plt.subplots(figsize=(16*0.66, 9*0.66))
            plt.plot(np.arange(len(results)), results, lw=3, marker="*", color='black')
            plt.title(f"T = {niter}", fontsize=24)
            plt.xlabel('Position', fontsize=24)
            plt.ylabel('Temperature', fontsize=24)

            ax.set_xticks(ax.get_xticks(), fontsize=16)
            ax.set_yticks(ax.get_yticks(), fontsize=16)

            plt.ylim(0, max(results))
            plt.xlim(0, len(results))
            plt.savefig('laplace_final.svg')

if __name__ == "__main__":
    unittest.main()