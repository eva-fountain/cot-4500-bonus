# Eva Fountain
# COT4500, Spring 2023
# Bonus Assignment

import numpy as np

# Number 1
tolerance = 1e-6
max_iterations = 50


def gauss_seidel(matrix: np.array, b_vector: np.array, w: float):
    # initial guess
    x0 = np.ones_like(b_vector, dtype=float)
    iteration = 0
    error = tolerance + 1

    while error > tolerance and iteration < max_iterations:
        x = np.copy(x0)
        for i in range(len(b_vector)):
            first_summation = np.dot(matrix[i, :i], x[:i])
            second_summation = np.dot(matrix[i, i + 1:], x0[i + 1:])
            right_hand_side = w * (b_vector[i] - first_summation - second_summation) / matrix[i, i]
            left_hand_side = (1 - w) * x0[i]
            x[i] = left_hand_side + right_hand_side

        error = np.linalg.norm(x - x0)

        x0 = np.copy(x)
        iteration += 1

    # if we went through ALL iterations, it might not have converged
    if iteration == max_iterations:
        iteration = -1

    return iteration


matrix = np.array([[3, 1, 1],
                   [1, 4, 1],
                   [2, 3, 7]])

b_vector = np.array([1, 3, 0])
value_for_w: float = 1.0
print(gauss_seidel(matrix, b_vector, value_for_w))
print()


# Number 2
tolerance = 1e-6
max_iterations = 50

def jacobi(matrix, b_vector):
    iteration = 0
    error = tolerance + 1

    for i in range(matrix.shape[0]):

        row = ["{}*x{}".format(matrix[i, j], j + 1) for j in range(matrix.shape[1])]

    x = np.zeros_like(b_vector)
    for it_count in range(max_iterations):
        x_new = np.zeros_like(x)

        for i in range(matrix.shape[0]):
            s1 = np.dot(matrix[i, :i], x[:i])
            s2 = np.dot(matrix[i, i + 1:], x[i + 1:])
            x_new[i] = (b_vector[i] - s1 - s2) / matrix[i, i]
            iteration += 1


        if np.allclose(x, x_new, atol=1e-6, rtol=0):
            break


        x = x_new

    return iteration

matrix = np.array([[3, 1, 1],
                   [1, 4, 1],
                   [2, 3, 7]])
b_vector = np.array([1, 3, 0])
print(jacobi(matrix, b_vector))
print()


# Number 3
def function(value):
    return (value ** 3) - (value ** 2) + 2


def custom_derivative(value):
    return (3 * value * value) - (2 * value)


def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    iteration_counter = 0

    # finds f
    x = initial_approximation
    f = eval(sequence)

    # finds f'
    f_prime = custom_derivative(initial_approximation)

    approximation: float = f / f_prime
    while (abs(approximation) >= tolerance):
        # finds f
        x = initial_approximation
        f = eval(sequence)

        # finds f'
        f_prime = custom_derivative(initial_approximation)

        # division operation
        approximation = f / f_prime

        # subtraction property
        initial_approximation -= approximation
        iteration_counter += 1

    return iteration_counter


initial_approximation: float = 0.5
tolerance: float = .000001
sequence: str = "(x**3) - (x**2) + 2"

print(newton_raphson(initial_approximation, tolerance, sequence))
print()


# Number 4
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            left = matrix[i][j - 1]
            diagonal_left = matrix[i - 1][j - 1]
            numerator = left - diagonal_left
            denominator = matrix[i][0] - matrix[i - j + 1][0]
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


def hermite_interpolation():
    x_points = [0, 1, 2]
    y_points = [1, 2, 4]
    slope = [1.06, 1.23, 1.55]
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2))

    # x values (every 2 rows)
    for x in range(num_of_points):
        matrix[x * 2][0] = x_points[x]
        matrix[x * 2 + 1][0] = x_points[x]

    # y values (every 2 rows)
    for x in range(num_of_points):
        matrix[x * 2][1] = y_points[x]
        matrix[x * 2 + 1][1] = y_points[x]

    # derivatives (every 2 rows)
    for x in range(num_of_points):
        matrix[x * 2 + 1][2] = slope[x]

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)


hermite_interpolation()
print()


# Number 5
def function(t: float, w: float):
    return w - (t ** 3)


def do_work(t, w, h):
    basic_function_call = function(t, w)

    incremented_t = t + h
    incremented_w = w + (h * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_w)

    return basic_function_call + incremented_function_call


def modified_eulers():
    original_w = 0.5
    start_of_t, end_of_t = (0, 3)
    num_of_iterations = 100

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # create a function for the inner work
        inner_math = do_work(t, w, h)

        # this gets the next approximation
        next_w = w + ((h / 2) * inner_math)

        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w

    print("%.5f" % original_w)
    return None


modified_eulers()
