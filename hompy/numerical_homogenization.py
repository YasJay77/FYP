import scipy.sparse.linalg
import scipy
import math
import numpy as np
from . import utilities as ut

try:
    import pypardiso

    use_pardiso = True
except ModuleNotFoundError:
    use_pardiso = False

DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT = 24


def homogenize(unit_cell_length_on_x, unit_cell_length_on_y, unit_cell_length_on_z, lambda_list, mu_list, voxels,
               direct_solution=False):
    """numerical homogenisation: effective stiffness tensor from a voxel unit cell"""
    # input type check
    if type(lambda_list) == (float or int):
        lambda_list = [lambda_list]
    elif type(lambda_list) == tuple:
        list(lambda_list)

    if type(mu_list) == (float or int):
        mu_list = [mu_list]
    elif type(mu_list) == tuple:
        list(mu_list)

    """INITIALIZE"""
    (number_of_elements_along_x_axis, number_of_elements_along_y_axis, number_of_elements_along_z_axis) = voxels.shape
    number_of_nodes_along_x_axis = number_of_elements_along_x_axis + 1
    number_of_nodes_along_y_axis = number_of_elements_along_y_axis + 1
    number_of_nodes_along_z_axis = number_of_elements_along_z_axis + 1

    """stiffness matrix"""
    hx = unit_cell_length_on_x / number_of_elements_along_x_axis
    hy = unit_cell_length_on_y / number_of_elements_along_y_axis
    hz = unit_cell_length_on_z / number_of_elements_along_z_axis
    number_of_elements = number_of_elements_along_x_axis * number_of_elements_along_y_axis * number_of_elements_along_z_axis

    ke_lambda, ke_mu, fe_lambda, fe_mu = calculate_split_parts_of_element_stiffness_matrix_and_load_vector(hx, hy,
                                                                                                           hz)

    """node numbers and element degrees of freedom for full (not periodic) mesh"""
    node_numbers = np.reshape(
        np.arange(1, number_of_nodes_along_x_axis * number_of_nodes_along_y_axis * number_of_nodes_along_z_axis + 1),
        (number_of_nodes_along_x_axis, number_of_nodes_along_y_axis, number_of_nodes_along_z_axis))
    elemental_degrees_of_freedom = np.reshape(
        3 * node_numbers[0:number_of_elements_along_x_axis, 0:number_of_elements_along_y_axis,
            0:number_of_elements_along_z_axis] + 1, (number_of_elements, 1))
    addx_var = 3 * number_of_elements_along_x_axis
    addx = np.array(
        [0, 1, 2, addx_var + 3, addx_var + 4, addx_var + 5, addx_var, addx_var + 1, addx_var + 2, -3, -2, -1])
    addxy = 3 * (number_of_elements_along_y_axis + 1) * (number_of_elements_along_x_axis + 1) + addx
    arr1 = np.array([addx, addxy])
    edof1 = np.tile(elemental_degrees_of_freedom, (1, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    edof2 = np.tile(arr1, (number_of_elements, 1))
    edof2 = np.reshape(edof2, (number_of_elements, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    edof = edof1 + edof2

    """impose periodic boundary conditions use original edofmat to index into list with the periodic dofs"""
    ndof, edof = impose_periodic_boundary_conditions(number_of_nodes_along_x_axis, number_of_nodes_along_y_axis,
                                                     number_of_nodes_along_z_axis, number_of_elements_along_x_axis,
                                                     number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                                     edof)

    """assemble global stiffness matrix and load vectors"""
    dim_1, dim_2, dim_3 = np.shape(edof)
    j_k_arr = np.kron(edof.reshape(1, dim_1 * dim_2) - np.ones((1, dim_1 * dim_2)),
                      np.ones((1, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))).astype('int32').reshape(
        dim_1 * dim_2 * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)
    i_k_arr = np.kron(edof.reshape((dim_1, dim_2)) - np.ones((dim_1, dim_2)),

                      np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 1))).astype('int32').reshape(
        dim_1 * dim_2 * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)

    lambda_ = 0
    mu_ = 0
    for i in range(len(lambda_list)):
        lambda_ = lambda_list[i] * (voxels == (i + 1)) + lambda_
        mu_ = mu_list[i] * (voxels == (i + 1)) + mu_

    ke_lambda = np.reshape(ke_lambda, (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 1))
    lambda_ = np.reshape(lambda_, (1, number_of_elements))
    ke_mu = np.reshape(ke_mu, (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 1))
    mu_ = np.reshape(mu_, (1, number_of_elements))
    stiffness_matrices_of_each_element = ke_lambda[:] * lambda_[:] + ke_mu[:] * mu_[:]

    dim1, dim2 = np.shape(stiffness_matrices_of_each_element)

    # keep contiguous numpy, not .tolist()
    s_k_arr = np.ascontiguousarray(stiffness_matrices_of_each_element.T.ravel())

    k = scipy.sparse.csr_matrix((s_k_arr, (i_k_arr, j_k_arr)), shape=(ndof, ndof))
    k = 0.5 * (k + k.transpose())

    """assemble three load cases"""
    i_f = np.tile(edof.T, (6, 1))
    i_f = i_f[0]
    j_f = np.concatenate((np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          2 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          3 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          4 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          5 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          6 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements))), axis=0)
    fe_lambda = np.reshape(fe_lambda, (144, 1), order='F')
    fe_mu = np.reshape(fe_mu, (144, 1), order='F')
    s_f = fe_lambda[:] * lambda_[:] + fe_mu[:] * mu_[:]

    i_f = np.reshape(i_f, (1, 144 * number_of_elements), order='F')
    j_f = np.reshape(j_f, (1, 144 * number_of_elements), order='F')
    s_f = np.reshape(s_f, (1, 144 * number_of_elements), order='F')
    i_f = i_f[0]
    j_f = j_f[0]
    s_f = s_f[0]

    f = scipy.sparse.csr_matrix((s_f, (i_f, j_f)), shape=(ndof + 1, 7))
    f = scipy.sparse.csr_matrix.toarray(f)[1:, 1:]

    """SOLUTION"""
    voxels = np.reshape(voxels, (
        number_of_elements_along_x_axis * number_of_elements_along_y_axis * number_of_elements_along_z_axis, 1))
    active_degrees_of_freedom = []

    voxel_types = [v_type + 1 for v_type in range(len(lambda_list))]

    for i in range(number_of_elements):

        if voxels[i] in voxel_types:
            active_degrees_of_freedom.append(edof[i, :])
    active_degrees_of_freedom = np.array(active_degrees_of_freedom)
    active_degrees_of_freedom = np.unique(active_degrees_of_freedom)
    active_degrees_of_freedom = [int(x) for x in active_degrees_of_freedom]

    x = np.zeros((ndof, 6))

    c_style_active_indices = [x - 1 for x in active_degrees_of_freedom[3:]]
    f_reduced = np.take(f, c_style_active_indices, axis=0)

    k_reduced = ut.take_sparse(k, c_style_active_indices)

    if direct_solution:
        print("using direct solver...")
        for i in range(6):
            if not use_pardiso:
                _x = scipy.sparse.linalg.spsolve(k_reduced, f_reduced[:, i])
            else:
                _x = pypardiso.spsolve(k_reduced, f_reduced[:, i])
            np.put(x[:, i], c_style_active_indices, _x)
    else:
        print("using iterative solver...")
        for i in range(6):
            _x = scipy.sparse.linalg.cg(
                k_reduced,
                f_reduced[:, i],
                x0=None,
                rtol=1e-10,
                maxiter=300
            )[0]
            np.put(x[:, i], c_style_active_indices, _x)

    # Homogenization
    x0 = np.zeros((number_of_elements, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6))
    x0_e = np.zeros((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6))
    ke = ke_mu.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)) + ke_lambda.reshape(
        (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    fe_mu_local = fe_mu.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6), order='F')
    fe_lambda_local = fe_lambda.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6), order='F')
    fe = fe_mu_local + fe_lambda_local
    fx = np.delete(np.arange(DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT), [0, 1, 2, 4, 5, 11], axis=0)
    ke_reduced = np.take(np.take(ke, fx, 0), fx, 1)
    fe_reduced = np.take(fe, fx, 0)
    x0_e_reduced = np.zeros((18, 6))

    for i in range(6):
        x0_e_reduced[:, i] = np.linalg.solve(ke_reduced, fe_reduced[:, i])
        ia = np.linalg.solve(ke_reduced, fe_reduced[:, i])
        np.put(x0_e[:, i], fx, ia)
        x0[:, :, i] = np.kron(x0_e[:, i].T, np.ones((number_of_elements, 1)))

    homogenized_constitutive_matrix = np.zeros((6, 6))
    unit_cell_volume = unit_cell_length_on_x * unit_cell_length_on_y * unit_cell_length_on_z
    for i in range(6):
        for j in range(6):
            def beta(edof, q, ndof, noe_x, noe_y, noe_z, DOF_FEE, X):
                """element strain-displacement / thermal-load helper"""
                w = ((edof + q * ndof - 1) % ndof).reshape((noe_x * noe_y * noe_z, DOF_FEE)).astype(int)
                be = np.zeros((noe_x * noe_y * noe_z, DOF_FEE))
                for k in range(DOF_FEE):
                    be[:, k] = np.take(X[:, q], w[:, k], 0)
                return be

            delta_l = np.matmul((x0[:, :, i] - beta(edof, i, ndof, number_of_elements_along_x_axis,
                                                    number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                                    DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, x)), ke_lambda.reshape(
                (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)))

            delta_m = np.matmul((x0[:, :, i] - beta(edof, i, ndof, number_of_elements_along_x_axis,
                                                    number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                                    DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, x)), ke_mu.reshape(
                (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)))

            epsilon = x0[:, :, j] - beta(edof, j, ndof, number_of_elements_along_x_axis,
                                         number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                         DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, x)

            sum_l = delta_l * epsilon
            sum_m = delta_m * epsilon

            sum_l = np.sum(sum_l, axis=1).reshape(number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                                                  number_of_elements_along_z_axis)
            sum_m = np.sum(sum_m, axis=1).reshape(number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                                                  number_of_elements_along_z_axis)
            homogenized_constitutive_matrix[i][j] = np.sum(
                np.sum(np.sum((lambda_.reshape(number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                                               number_of_elements_along_z_axis) * sum_l + mu_.reshape(
                    number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                    number_of_elements_along_z_axis) * sum_m), 0), 0), 0) / unit_cell_volume

    return homogenized_constitutive_matrix


def apply_thermal_homogenization(unit_cell_length_on_x, unit_cell_length_on_y, unit_cell_length_on_z,
                                 thermal_conductivity_list, voxels, thermal_conductivity_of_space=0,
                                 direct_solution=False):
    """numerical homogenisation of effective thermal conductivity from a voxel unit cell"""
    # input type check
    thermal_conductivity = 0
    thermal_conductivity_list = [thermal_conductivity_of_space] + thermal_conductivity_list

    """INITIALIZE"""
    (number_of_elements_along_x_axis, number_of_elements_along_y_axis, number_of_elements_along_z_axis) = voxels.shape
    number_of_nodes_along_x_axis = number_of_elements_along_x_axis + 1
    number_of_nodes_along_y_axis = number_of_elements_along_y_axis + 1
    number_of_nodes_along_z_axis = number_of_elements_along_z_axis + 1

    """stiffness matrix"""
    hx = unit_cell_length_on_x / number_of_elements_along_x_axis
    hy = unit_cell_length_on_y / number_of_elements_along_y_axis
    hz = unit_cell_length_on_z / number_of_elements_along_z_axis
    number_of_elements = number_of_elements_along_x_axis * number_of_elements_along_y_axis * number_of_elements_along_z_axis
    ke_lambda, ke_mu, fe_lambda, fe_mu = calculate_split_parts_of_element_stiffness_matrix_and_load_vector(hx, hy, hz,
                                                                                                           True)
    ke_mu[0::3, 0::3] = ke_mu[0::3, 0::3] + ke_mu[1::3, 1::3] + ke_mu[2::3, 2::3]

    """node numbers and element degrees of freedom for full (not periodic) mesh"""
    node_numbers = np.reshape(
        np.arange(1, number_of_nodes_along_x_axis * number_of_nodes_along_y_axis * number_of_nodes_along_z_axis + 1),
        (number_of_nodes_along_x_axis, number_of_nodes_along_y_axis, number_of_nodes_along_z_axis))
    elemental_degrees_of_freedom = np.reshape(
        3 * node_numbers[0:number_of_elements_along_x_axis, 0:number_of_elements_along_y_axis,
            0:number_of_elements_along_z_axis] + 1, (number_of_elements, 1))
    addx_var = 3 * number_of_elements_along_x_axis
    addx = np.array(
        [0, 1, 2, addx_var + 3, addx_var + 4, addx_var + 5, addx_var, addx_var + 1, addx_var + 2, -3, -2, -1])
    addxy = 3 * (number_of_elements_along_y_axis + 1) * (number_of_elements_along_x_axis + 1) + addx
    arr1 = np.array([addx, addxy])
    edof1 = np.tile(elemental_degrees_of_freedom, (1, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    edof2 = np.tile(arr1, (number_of_elements, 1))
    edof2 = np.reshape(edof2, (number_of_elements, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    edof = edof1 + edof2

    """impose periodic boundary conditions use original edofmat to index into list with the periodic dofs"""
    ndof, edof = impose_periodic_boundary_conditions(number_of_nodes_along_x_axis, number_of_nodes_along_y_axis,
                                                     number_of_nodes_along_z_axis, number_of_elements_along_x_axis,
                                                     number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                                     edof)

    """assemble global stiffness matrix and load vectors"""
    dim_1, dim_2, dim_3 = np.shape(edof)
    j_k_arr = np.kron(edof.reshape(1, dim_1 * dim_2) - np.ones((1, dim_1 * dim_2)),
                      np.ones((1, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))).astype('int32').reshape(
        dim_1 * dim_2 * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)

    i_k_arr = np.kron(edof.reshape((dim_1, dim_2)) - np.ones((dim_1, dim_2)),
                      np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 1))).astype('int32').reshape(
        dim_1 * dim_2 * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)

    for i in range(len(thermal_conductivity_list)):
        thermal_conductivity = thermal_conductivity_list[i] * (voxels == i) + thermal_conductivity

    ke_mu = np.reshape(ke_mu, (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT * DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 1))
    thermal_conductivity = np.reshape(thermal_conductivity, (1, number_of_elements))
    stiffness_matrices_of_each_element = ke_mu[:] * thermal_conductivity[:]

    dim1, dim2 = np.shape(stiffness_matrices_of_each_element)

    # keep contiguous numpy, not .tolist()
    s_k_arr = np.ascontiguousarray(stiffness_matrices_of_each_element.T.ravel())

    k = scipy.sparse.csr_matrix((s_k_arr, (i_k_arr, j_k_arr)), shape=(ndof, ndof))
    k = 0.5 * (k + k.transpose())

    """assemble three load cases"""
    i_f = np.tile(edof.T, (6, 1))
    i_f = i_f[0]
    j_f = np.concatenate((np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          2 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          3 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          4 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          5 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements)),
                          6 * np.ones((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, number_of_elements))), axis=0)
    fe_lambda = np.reshape(fe_lambda, (144, 1), order='F')
    fe_mu = np.reshape(fe_mu, (144, 1), order='F')
    s_f = fe_mu[:] * thermal_conductivity[:]

    i_f = np.reshape(i_f, (1, 144 * number_of_elements), order='F')
    j_f = np.reshape(j_f, (1, 144 * number_of_elements), order='F')
    s_f = np.reshape(s_f, (1, 144 * number_of_elements), order='F')
    i_f = i_f[0]
    j_f = j_f[0]
    s_f = s_f[0]

    f = scipy.sparse.csr_matrix((s_f, (i_f, j_f)), shape=(ndof + 1, 7))
    f = scipy.sparse.csr_matrix.toarray(f)[1:, 1:]

    """SOLUTION"""
    x = np.zeros((ndof, 3))

    if direct_solution:
        for i in range(3):
            if not use_pardiso:
                _x = scipy.sparse.linalg.spsolve(ut.take_sparse(k, np.arange(3, ndof, 3).tolist()),
                                                 np.take(f, np.arange(3 + i, ndof, 3), axis=0)[:, i])
            else:
                _x = pypardiso.spsolve(ut.take_sparse(k, np.arange(3, ndof, 3).tolist()),
                                       np.take(f, np.arange(3 + i, ndof, 3), axis=0)[:, i])
            np.put(x[:, i], np.arange(3, ndof, 3), _x)
    else:
        for i in range(3):
            _x, e = scipy.sparse.linalg.cg(ut.take_sparse(k, np.arange(3, ndof, 3).tolist()),
                                           np.take(f, np.arange(3 + i, ndof, 3), axis=0)[:, i], None, 1e-10, 300)
            np.put(x[:, i], np.arange(3, ndof, 3), _x)

    # Homogenization
    x0 = np.zeros((number_of_elements, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6))
    x0_e = np.zeros((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6))
    ke = ke_mu.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)) + ke_lambda.reshape(
        (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT))
    fe_mu_local = fe_mu.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6), order='F')
    fe_lambda_local = fe_lambda.reshape((DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 6), order='F')
    fe = fe_mu_local + fe_lambda_local

    for i in range(3):
        x0_e_ = np.linalg.solve(
            np.take(np.take(ke, np.arange(3, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 3), 0),
                    np.arange(3, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 3), 1),
            np.take(fe, np.arange(3 + i, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 3), axis=0)[:, i])
        np.put(x0_e[:, i], np.arange(3, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, 3), x0_e_)
        x0[:, :, i] = np.kron(x0_e[:, i].T, np.ones((number_of_elements, 1)))

    homogenized_constitutive_matrix = np.zeros((6, 6))
    unit_cell_volume = unit_cell_length_on_x * unit_cell_length_on_y * unit_cell_length_on_z

    for i in range(3):
        for j in range(3):
            def beta(edof, q, ndof, noe_x, noe_y, noe_z, DOF_FEE, X):
                """element strain-displacement / thermal-load helper"""
                w = ((edof + q * ndof - 1) % ndof).reshape((noe_x * noe_y * noe_z, DOF_FEE)).astype(int)
                be = np.zeros((noe_x * noe_y * noe_z, DOF_FEE))
                for k in range(DOF_FEE):
                    be[:, k] = np.take(X[:, q], w[:, k], 0)
                return be

            delta_m = np.matmul((x0[:, :, i] - beta(edof, i, ndof, number_of_elements_along_x_axis,
                                                    number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                                    DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, x)), ke_mu.reshape(
                (DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT)))

            epsilon = x0[:, :, j] - beta(edof, j, ndof, number_of_elements_along_x_axis,
                                         number_of_elements_along_y_axis, number_of_elements_along_z_axis,
                                         DEGREES_OF_FREEDOM_FOR_EACH_ELEMENT, x)

            sum_m = delta_m * epsilon
            sum_m = np.sum(sum_m, axis=1).reshape(number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                                                  number_of_elements_along_z_axis)
            homogenized_constitutive_matrix[i][j] = np.sum(np.sum(np.sum((thermal_conductivity.reshape(
                number_of_elements_along_x_axis, number_of_elements_along_y_axis,
                number_of_elements_along_z_axis) * sum_m), 0), 0), 0) * (1 / unit_cell_volume)

    return homogenized_constitutive_matrix


def calculate_split_parts_of_element_stiffness_matrix_and_load_vector(hx, hy, hz, thermal=False):
    """constitutive matrix contributions"""
    c_lambda = np.zeros((6, 6))
    if thermal:
        c_mu = np.diag([1, 1, 1, 0, 0, 0])
    else:
        c_mu = np.diag([2, 2, 2, 1, 1, 1])
        c_lambda[0: 3, 0: 3] = 1
    """Three Gauss points in both directions"""
    gaussian_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
    weights = [5 / 9, 8 / 9, 5 / 9]
    """Initialize"""
    ke_lambda = np.zeros((24, 24))
    ke_mu = np.zeros((24, 24))
    fe_lambda = np.zeros((24, 6))
    fe_mu = np.zeros((24, 6))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                """integration point"""
                x = gaussian_points[i]
                y = gaussian_points[j]
                z = gaussian_points[k]

                """stress strain displacement matrix"""
                qx = [-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                      ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                      ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8]

                qy = [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                      ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                      ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8]

                qz = [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                      ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                      ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]

                """Jacobian"""
                J1 = np.matrix([qx, qy, qz])
                arr1 = np.array([-0.5 * hx, 0.5 * hx, 0.5 * hx, -0.5 * hx, -0.5 * hx, 0.5 * hx, 0.5 * hx, -0.5 * hx])
                arr2 = np.array([-0.5 * hy, -0.5 * hy, 0.5 * hy, 0.5 * hy, -0.5 * hy, -0.5 * hy, 0.5 * hy, 0.5 * hy])
                arr3 = np.array([-0.5 * hz, -0.5 * hz, -0.5 * hz, -0.5 * hz, 0.5 * hz, 0.5 * hz, 0.5 * hz, 0.5 * hz])
                J2 = np.matrix([arr1, arr2, arr3]).T
                J = J1 * J2
                qxyz = scipy.linalg.solve(J, J1)

                "qxyz fix loop"
                for a in range(0, 8, 2):
                    temporary_variable = qxyz[1][a]
                    qxyz[1][a] = qxyz[1][a + 1]
                    qxyz[1][a + 1] = temporary_variable

                    temporary_variable = qxyz[2][a]
                    qxyz[2][a] = qxyz[2][a + 1]
                    qxyz[2][a + 1] = temporary_variable

                B_e = np.zeros((6, 3, 8))

                for i_B in range(0, 8):
                    B_e[:, :, i_B] = np.matrix([[qxyz[0, i_B], 0, 0], [0, qxyz[1, i_B], 0], [0, 0, qxyz[2, i_B]],
                                                [qxyz[1, i_B], qxyz[0, i_B], 0], [0, qxyz[2, i_B], qxyz[1, i_B]],
                                                [qxyz[2, i_B], 0, qxyz[0, i_B]]])

                B = np.concatenate(
                    (B_e[:, :, 0], B_e[:, :, 1], B_e[:, :, 2], B_e[:, :, 3], B_e[:, :, 4], B_e[:, :, 5], B_e[:, :, 6],
                     B_e[:, :, 7]), axis=1)

                """Weight factor at this point"""
                weight = np.linalg.det(J) * weights[i] * weights[j] * weights[k]

                """Element matrices"""
                ke_lambda = ke_lambda + weight * np.matmul(B.T, np.matmul(c_lambda, B))
                ke_mu = ke_mu + weight * np.matmul(np.matmul(B.T, c_mu), B)

                """Element loads"""
                fe_lambda = fe_lambda + weight * np.matmul(B.T, c_lambda)
                fe_mu = fe_mu + weight * np.matmul(B.T, c_mu)
    return ke_lambda, ke_mu, fe_lambda, fe_mu


def impose_periodic_boundary_conditions(number_of_nodes_along_x_axis, number_of_nodes_along_y_axis,
                                        number_of_nodes_along_z_axis, number_of_elements_along_x_axis,
                                        number_of_elements_along_y_axis, number_of_elements_along_z_axis, edof):
    """remap dofs so opposite faces of the unit cell share nodes (periodic mesh)"""
    total_number_of_nodes = number_of_nodes_along_x_axis * number_of_nodes_along_y_axis * number_of_nodes_along_z_axis
    total_number_of_unique_nodes = number_of_elements_along_x_axis * number_of_elements_along_y_axis * \
                                   number_of_elements_along_z_axis
    initnnparray = np.reshape(np.arange(1, total_number_of_unique_nodes + 1), (
        number_of_elements_along_x_axis, number_of_elements_along_y_axis, number_of_elements_along_z_axis))
    nnparray = np.zeros(
        (number_of_elements_along_x_axis + 1, number_of_elements_along_y_axis + 1, number_of_elements_along_z_axis + 1))
    nnparray[0:number_of_elements_along_x_axis, 0:number_of_elements_along_y_axis,
    0:number_of_elements_along_z_axis] = initnnparray
    nnparray[np.shape(nnparray)[0] - 1, :, :] = nnparray[0, :, :]
    nnparray[:, np.shape(nnparray)[1] - 1, :] = nnparray[:, 0, :]
    nnparray[:, :, np.shape(nnparray)[2] - 1] = nnparray[:, :, 0]
    dofvector = np.zeros((3 * total_number_of_nodes, 1))
    nnparray = np.reshape(nnparray, (nnparray.size, 1))

    for i in range(0, len(dofvector), 3):
        dofvector[0 + i] = 3 * nnparray[math.floor(i / 3)] - 2
        dofvector[1 + i] = 3 * nnparray[math.floor(i / 3)] - 1
        dofvector[2 + i] = 3 * nnparray[math.floor(i / 3)]
    edof = dofvector[edof - 1]
    ndof = 3 * total_number_of_unique_nodes
    return ndof, edof
