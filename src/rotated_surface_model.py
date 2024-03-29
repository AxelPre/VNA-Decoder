import numpy as np
from random import random
from numba import njit
import random as rand
import matplotlib.pyplot as plt

class RotSurCode():
    nbr_eq_classes = 4

    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((size, size), dtype=np.uint8)
        self.plaquette_defects = np.zeros((size + 1, size + 1))

    def generate_random_error(self, p_x, p_y, p_z):
        size = self.system_size
        for i in range(size):
            for j in range(size):
                q = 0
                r = rand.random()
                if r < p_z:
                    q = 3
                if p_z < r < (p_z + p_x):
                    q = 1
                if (p_z + p_x) < r < (p_z + p_x + p_y):
                    q = 2
                self.qubit_matrix[i, j] = q
        self.syndrome()


    def chain_lengths(self):
        nx = np.count_nonzero(self.qubit_matrix[:, :] == 1)
        ny = np.count_nonzero(self.qubit_matrix[:, :] == 2)
        nz = np.count_nonzero(self.qubit_matrix[:, :] == 3)
        return nx, ny, nz

    def count_errors(self):
        return _count_errors(self.qubit_matrix)

    def apply_logical(self, operator: int, X_pos=0, Z_pos=0):
        return _apply_logical(self.qubit_matrix, operator, X_pos, Z_pos)

    def apply_stabilizer(self, row: int, col: int, operator: int):
        return _apply_stabilizer(self.qubit_matrix, row, col, operator)

    def apply_random_logical(self):
        return _apply_random_logical(self.qubit_matrix)

    def apply_random_stabilizer(self):
        return _apply_random_stabilizer(self.qubit_matrix)

    def apply_stabilizers_uniform(self, p=0.5):
        return _apply_stabilizers_uniform(self.qubit_matrix, p)

    def define_equivalence_class(self):
        return _define_equivalence_class(self.qubit_matrix)
    
    def to_class(self, eq):
        eq_class = self.define_equivalence_class()
        op = eq_class ^ eq
        return self.apply_logical(op)[0]

    def syndrome(self):
        size = self.qubit_matrix.shape[1]
        qubit_matrix = self.qubit_matrix
        for i in range(size-1):
            for j in range(size-1):
                self.plaquette_defects[i+1, j+1] = _find_syndrome(qubit_matrix, i, j, 1)
        for i in range(int((size - 1)/2)):
            for j in range(4):
                row = 0
                col = 0
                if j == 0:
                    row = 0
                    col = 2 * i + 2
                elif j == 1:
                    row = 2 * i + 2
                    col = size
                elif j == 2:
                    row = size
                    col = 2 * i + 1
                elif j == 3:
                    row = 2 * i + 1
                    col = 0
                self.plaquette_defects[row, col] = _find_syndrome(qubit_matrix, i, j, 3)
    def update_matrix(self, newmatrix):
        size = self.qubit_matrix.shape[1]
        for i in range(size):
            for j in range(size):
                self.qubit_matrix[i,j] = newmatrix[i,j]
        self.syndrome()

    def plot(self, title):
        system_size = self.system_size
        xLine = np.linspace(0, system_size - 1, system_size)
        a = range(system_size)
        X, Y = np.meshgrid(a, a)
        XLine, YLine = np.meshgrid(a, xLine)
        plaquette_defect_coordinates = np.where(self.plaquette_defects)

        x_error = np.where(self.qubit_matrix[:, :] == 1)
        y_error = np.where(self.qubit_matrix[:, :] == 2)
        z_error = np.where(self.qubit_matrix[:, :] == 3)

        def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
            x = np.arange(center_x, center_x + radius + stepsize, stepsize)
            y = np.sqrt(radius ** 2 - x ** 2)
            x = np.concatenate([x, x[::-1]])
            y = np.concatenate([y, -y[::-1]])
            return x, y + center_y

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        # Plot grid lines
        ax = plt.subplot(111)

        x, y = generate_semicircle(0, 1, 0.5, 0.01)

        for i in range(int((system_size - 1) / 2)):
            ax.plot(y + 0.5 + i * 2, x + system_size - 1, color='black', linewidth=linewidth)
            ax.plot(-y + 1.5 + 2 * i, -x, color='black', linewidth=linewidth)
            ax.plot(x + system_size - 1, y - 0.5 + i * 2, color='black', linewidth=linewidth)
            ax.plot(-x, -y + 0.5 + system_size - 1 - 2 * i, color='black', linewidth=linewidth)

        ax.plot(XLine, YLine, 'black', linewidth=linewidth)
        ax.plot(YLine, XLine, 'black', linewidth=linewidth)

        ax.plot(X, Y, 'o', color='black', markerfacecolor='white', markersize=markersize_qubit + 1)

        ax.plot(x_error[1], system_size - 1 - x_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$X$')
        ax.plot(y_error[1], system_size - 1 - y_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$Y$')
        ax.plot(z_error[1], system_size - 1 - z_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$Z$')

        for i in range(len(plaquette_defect_coordinates[1])):
            if plaquette_defect_coordinates[1][i] == 0:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5 + 0.25, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[0][i] == 0:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5 - 0.25, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[1][i] == system_size:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5 - 0.25, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[0][i] == system_size:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5 + 0.25, 'o', color='red', label="flux", markersize=markersize_excitation)
            else:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)

        # ax.plot(plaquette_defect_coordinates[1] - 0.5, system_size - plaquette_defect_coordinates[0] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
        ax.axis('off')

        plt.axis('equal')
        #plt.show()
        plt.savefig('plots/graph_'+str(title)+'.png')
        # plt.close()


@njit('(uint8[:,:],)')
def _count_errors(qubit_matrix):
    return np.count_nonzero(qubit_matrix)


@njit('(uint8[:,:], int64, int64, int64)')
def _find_syndrome(qubit_matrix, row: int, col: int, operator: int):

    def flip(a):
        if a == 0:
            return 1
        elif a == 1:
            return 0

    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    defect = 0
    op = 0

    if operator == 1:  # full
        qarray = [[0 + row, 0 + col], [0 + row, 1 + col], [1 + row, 0 + col], [1 + row, 1 + col]]
        if row % 2 == 0:
            if col % 2 == 0:
                op = 1
            else:
                op = 3
        else:
            if col % 2 == 0:
                op = 3
            else:
                op = 1
    elif operator == 3:  # half
        if col == 0:
            op = 1
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
        elif col == 1:
            op = 3
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
        elif col == 2:
            op = 1
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
        elif col == 3:
            op = 3
            qarray = [[row*2, 0], [row*2 + 1, 0]]

    for i in qarray:
        old_qubit = result_qubit_matrix[i[0], i[1]]
        if old_qubit != 0 and old_qubit != op:
            defect = flip(defect)

    return defect


@njit('(uint8[:,:], int64, int64, int64)')  # Z-biased noise
def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
    
    result_qubit_matrix = np.copy(qubit_matrix)

    # List to store how errors redestribute when logical is applied
    n_eq = [0, 0, 0, 0]

    if operator == 0:
        return result_qubit_matrix, (0, 0, 0)
    
    size = qubit_matrix.shape[0]

    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    if do_X:
        for i in range(size):
            old_qubit = result_qubit_matrix[i, X_pos]
            new_qubit = 1 ^ old_qubit
            result_qubit_matrix[i, X_pos] = new_qubit
            
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1
    if do_Z:
        for i in range(size):
            old_qubit = result_qubit_matrix[Z_pos, i]
            new_qubit = 3 ^ old_qubit
            result_qubit_matrix[Z_pos, i] = new_qubit
            
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1

    return result_qubit_matrix, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:],)')
def _apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[0]

    op = int(random() * 4)

    if op == 1 or op == 2:
        X_pos = int(random() * size)
    else:
        X_pos = 0
    if op == 3 or op == 2:
        Z_pos = int(random() * size)
    else:
        Z_pos = 0

    return _apply_logical(qubit_matrix, op, X_pos, Z_pos)


@njit('(uint8[:,:], int64, int64, int64)')
def _apply_stabilizer(qubit_matrix, row: int, col: int, operator: int):

    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)

    # List to store how errors redestribute when stabilizer is applied
    n_eq = [0, 0, 0, 0]
    
    op = 0

    if operator == 1:  # full
        qarray = [[0 + row, 0 + col], [0 + row, 1 + col], [1 + row, 0 + col], [1 + row, 1 + col]]
        if row % 2 == 0:
            if col % 2 == 0:
                op = 1
            else:
                op = 3
        else:
            if col % 2 == 0:
                op = 3
            else:
                op = 1
    elif operator == 3:  # half
        if col == 0:
            op = 1
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
        elif col == 1:
            op = 3
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
        elif col == 2:
            op = 1
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
        elif col == 3:
            op = 3
            qarray = [[row*2, 0], [row*2 + 1, 0]]

    for i in qarray:
        old_qubit = result_qubit_matrix[i[0], i[1]]
        new_qubit = op ^ old_qubit
        result_qubit_matrix[i[0], i[1]] = new_qubit
        
        n_eq[old_qubit] -= 1
        n_eq[new_qubit] += 1

    return result_qubit_matrix, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:],)')
def _apply_random_stabilizer(qubit_matrix):
    size = qubit_matrix.shape[0]
    rows = int((size-1)*random())
    cols = int((size-1)*random())
    rows2 = int(((size - 1)/2) * random())
    cols2 = int(4 * random())
    phalf = (size**2 - (size-1)**2 - 1)/(size**2-1)
    if rand.random() > phalf:
        # operator = 1 = full stabilizer
        return _apply_stabilizer(qubit_matrix, rows, cols, 1)
    else:
        # operator = 3 = half stabilizer
        return _apply_stabilizer(qubit_matrix, rows2, cols2, 3)


@njit('(uint8[:,:],)')
def _define_equivalence_class(qubit_matrix):

    x_errors = np.count_nonzero(qubit_matrix[0, :] == 1)
    x_errors += np.count_nonzero(qubit_matrix[0, :] == 2)

    z_errors = np.count_nonzero(qubit_matrix[:, 0] == 3)
    z_errors += np.count_nonzero(qubit_matrix[:, 0] == 2)

    if x_errors % 2 == 0:
        if z_errors % 2 == 0:
            return 0
        else:
            return 3
    else:
        if z_errors % 2 == 0:
            return 1
        else:
            return 2


def _apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)

    # Apply full stabilizers
    random_stabilizers = np.random.rand(size-1, size-1)
    random_stabilizers = np.less(random_stabilizers, p)

    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, 1)
        it.iternext()

    # Apply half stabilizers
    random_stabilizers = np.random.rand(int((size - 1)/2), 4)
    random_stabilizers = np.less(random_stabilizers, p)
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, 3)
        it.iternext()

    return result_qubit_matrix