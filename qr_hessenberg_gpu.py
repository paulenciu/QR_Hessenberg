import cupy as cp
import matplotlib.pyplot as plt

from time import time


def timing_decorator(func):
    execution_times = []

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        return result

    wrapper.execution_times = execution_times
    return wrapper


@timing_decorator
def create_matrix_and_vector(N):
    A = cp.zeros((N, N))
    b = cp.zeros(N)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i <= j + 1:
                A[i - 1][j - 1] = cp.float128(1) / (cp.float128(i) + cp.float128(j) - cp.float128(1))
                b[i - 1] += A[i - 1][j - 1]
            else:
                pass
    return A, b


@timing_decorator
def backwards_sub(matrix, solution_vector):
    return cp.linalg.solve(matrix, solution_vector)


@timing_decorator
def solve_linear_equation(matrix, solution_vector):
    return cp.linalg.solve(matrix, solution_vector)


@timing_decorator
def calculate_qr(matrix):
    return cp.linalg.qr(matrix)


if __name__ == '__main__':
    cp.cuda.Device(0).use()  # Specify the GPU device to use

    iteration = 0
    N_values = []
    execution_times_create_A_and_b = []
    execution_times_calculate_qr = []
    execution_times_backward_substitution = []
    execution_times_solve_equation = []

    for i in range(1, 100, 1):
        print("i=", i, "\n")
        N_values.append(i)

        A, b = create_matrix_and_vector(i)
        execution_times_create_A_and_b.append(cp.mean(create_matrix_and_vector.execution_times))

        Q, R = calculate_qr(A)
        execution_times_calculate_qr.append(cp.mean(calculate_qr.execution_times))

        c = backwards_sub(Q, b)
        execution_times_backward_substitution.append(cp.mean(backwards_sub.execution_times))

        x = solve_linear_equation(R, c)
        execution_times_solve_equation.append(cp.mean(solve_linear_equation.execution_times))

    plt.plot(N_values, execution_times_create_A_and_b, label='create A and b')
    plt.plot(N_values, execution_times_calculate_qr, label='calculate_qr')
    plt.plot(N_values, execution_times_backward_substitution, label='c=Q^T*b')
    plt.plot(N_values, execution_times_solve_equation, label='Rx=c')

    plt.xlabel('Matrixgröße N')
    plt.ylabel('Laufzeit (Sekunden)')
    plt.title('Laufzeiten der Schritte zur Lösung des Gleichungssystems')
    plt.legend()
    plt.show()
