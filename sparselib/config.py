import numpy as np

def __get_config_info():
    max_level_6_dim_1 = {
        "title": "max_level_6_dim_1",
        "initialize_system_domain": Domain(dim=1, max_level=6, bounds=np.array([[0, 2*np.pi]])),
        "inner_product_domain": Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]])),
        "triple_product_domain": Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]])),
    }
    max_level_6_dim_2 = {
        "title": "max_level_5_dim_3",
        "initialize_system_domain": Domain(dim=2, max_level=6, bounds=np.array([[0, 2*np.pi], 
                                                                                [0, 2*np.pi]])),
        "inner_product_domain": Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]])),
        "triple_product_domain": Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]])),
    }
    max_level_5_dim_3 = {
        "title": "max_level_5_dim_3",
        "initialize_system_domain": Domain(dim=3, max_level=5, bounds=np.array([[0, 2*np.pi], 
                                                                                [0, 2*np.pi], 
                                                                                [0, 2*np.pi]])),
        "inner_product_domain": Domain(dim=1, max_level=5, bounds=np.array([[0,2*np.pi]])),
        "triple_product_domain": Domain(dim=1, max_level=5, bounds=np.array([[0,2*np.pi]])),
    }
    return max_level_6_dim_1

def get_A_matrix_domain():
    return Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]]))

def get_L_matrix_domain():
    return Domain(dim=1, max_level=6, bounds=np.array([[0,2*np.pi]]))

def __use_cpp_A_matrix_as_primary():
    """
    Whether or not to use the C++ generated A matrix as the primary
    A matrix for further computations. If true, the C++ A matrix tmust
    be generated. If false, some other tables must be generated. 
    """
    return False

def __use_cpp_L_matrix_as_primary():
    """
    Whether or not to use the C++ generated L matrix as the primary
    L matrix for further computations. If true, the C++ L matrix tmust
    be generated. If false, some other tables must be generated. 
    """
    return False

def __use_cpp_triple_product_tables_as_primary():
    """
    Whether or not to use the C++ generated triple product tables as the primary
    triple product tables for further computations. If true, the C++ tables must
    be generated. If false, some other tables must be generated. 
    """
    return False

def __use_cpp_B_matrix_as_primary():
    return False



def compare_cpp_B_matrix_to_python():
    return __compare_cpp_B_matrix_to_python()

def __use_python_A_matrix_as_primary():
    """
    Whether or not to use the Python generated A matrix as the primary
    A matrix for further computations. If true, the Python A matrix tmust
    be generated. If false, some other tables must be generated. 
    """
    return not __use_cpp_A_matrix_as_primary()

def __use_python_L_matrix_as_primary():
    """
    Whether or not to use the Python generated L matrix as the primary
    L matrix for further computations. If true, the Python L matrix tmust
    be generated. If false, some other tables must be generated. 
    """
    return not __use_cpp_L_matrix_as_primary()


def __use_python_triple_product_tables_as_primary():
    """
    Whether or not to use the Python generated triple product tables as the primary
    triple product tables for further computations. If true, the Python tables must
    be generated. If false, some other tables must be generated. 
    """
    return not __use_cpp_triple_product_tables_as_primary()

def __compare_cpp_A_matrix_to_python():
    """
    If true, compare the C++ generated A matrix to the Python generated A matrix.
    If false, do not compare the C++ generated A matrix to the Python generated A matrix.
    Note that a False from this function should not imply that the C++ A matrix
    will not be generated at _all_, only that it will not necessarily be compared to the Python
    generated A matrix.
    """
    return False
    

def __compare_cpp_L_matrix_to_python():
    """
    If true, compare the C++ generated L matrix to the Python generated L matrix.
    If false, do not compare the C++ generated L matrix to the Python generated L matrix.
    Note that a False from this function should not imply that the C++ L matrix
    will not be generated at _all_, only that it will not necessarily be compared to the Python
    generated L matrix.
    """
    return False
    
def __compare_cpp_triple_product_tables_to_python():
    """
    If true, the C++ generated triple product tables will be compared to the Python generated
    triple product tables. This requires that both the C++ and Python triple product tables
    be generated. If false, the C++ generated triple product tables will not be compared to the
    Python generated triple product tables.
    """
    return False


def get_cpp_L_matrix_comparison_file_path():
    if __compare_cpp_L_matrix_to_python():
        return get_python_L_matrix_file_path()
    else:
        return None

def get_cpp_A_matrix_file_path():
    if __use_cpp_A_matrix_as_primary():
        return get_A_matrix_file_path()
    else:
        return "precomp/A_cpp.npy"

def get_python_A_matrix_file_path():
    if __use_python_A_matrix_as_primary():
        return get_A_matrix_file_path()
    else:
        return "precomp/A_python.npy"

def should_build_L_matrix():
    return False or should_build_python_B_matrix()

def get_cpp_B_matrix_file_path():
    if __use_cpp_B_matrix_as_primary():
        return get_B_matrix_file_path()
    else:
        return "precomp/B_cpp.npy"

def get_python_B_matrix_file_path():
    if __use_python_B_matrix_as_primary():
        return get_B_matrix_file_path()
    else:
        return "precomp/B_python.npy"

def get_cpp_L_matrix_file_path():
    if __use_cpp_L_matrix_as_primary():
        return get_L_matrix_file_path()
    else:
        return "precomp/L_cpp.npy"

def get_python_L_matrix_file_path():
    if __use_python_L_matrix_as_primary():
        return get_L_matrix_file_path()
    else:
        return "precomp/L_python.npy"

def get_python_table1_file_path():
    if __use_python_triple_product_tables_as_primary():
        return get_table1_file_path()
    else:
        return "precomp/table1_python.npy"

def get_python_table2_file_path():
    if __use_python_triple_product_tables_as_primary():
        return get_table2_file_path()
    else:
        return "precomp/table2_python.npy"


def get_initialize_system_domain():
    return __get_config_info()["initialize_system_domain"]

def get_inner_product_domain():
    return __get_config_info()["inner_product_domain"]

def get_triple_product_domain():
    return __get_config_info()["triple_product_domain"]

def get_precomp_directory_path():
    return 'precomp'

def get_global_dimwise_indices_file_path():
    return 'precomp/global_dimwise_indices.npy'

def get_scaling_file_path():
    return 'precomp/scaling.npy'

def get_table1_file_path():
    return 'precomp/table1.npy'

def get_table2_file_path():
    return 'precomp/table2.npy'

def get_table3_file_path():
    return 'precomp/table3.npy'

def get_A_matrix_file_path():
    return 'precomp/A.npy'

def get_B_matrix_file_path():
    return "precomp/B.npy"

def get_D1_matrix_file_path():
    return "precomp/D1.npy"

def get_L_matrix_file_path():
    return 'precomp/L.npy'

def get_P_matrix_file_path():
    return 'precomp/P.npy'


#
# Should not be modified by user
#

# Whether or not to call the C++ code to generate the L matrix at all.
def should_build_cpp_A_matrix():
    return __compare_cpp_A_matrix_to_python() or __use_cpp_A_matrix_as_primary()

# Whether or not to call the Python code to generate the L matrix at all.
def should_build_python_A_matrix():
    return __compare_cpp_A_matrix_to_python() or __use_python_A_matrix_as_primary()

def should_build_python_B_matrix():
    return __compare_cpp_B_matrix_to_python() or __use_python_B_matrix_as_primary()

def should_build_cpp_B_matrix():
    return __compare_cpp_B_matrix_to_python() or __use_cpp_B_matrix_as_primary()

def __compare_cpp_B_matrix_to_python():
    return False

def __use_python_B_matrix_as_primary():
    return not __use_cpp_B_matrix_as_primary()


# Whether or not to call the C++ code to generate the L matrix at all.
def should_build_cpp_L_matrix():
    return should_build_L_matrix() and (__compare_cpp_L_matrix_to_python() or __use_cpp_L_matrix_as_primary())

# Whether or not to call the Python code to generate the L matrix at all.
def should_build_python_L_matrix():
    return should_build_L_matrix() and (__compare_cpp_L_matrix_to_python() or __use_python_L_matrix_as_primary())

def should_build_cpp_triple_product_tables():
    return __compare_cpp_triple_product_tables_to_python() or __use_cpp_triple_product_tables_as_primary()

def should_build_python_triple_product_tables():
    return __compare_cpp_triple_product_tables_to_python() or __use_python_triple_product_tables_as_primary()

def get_cpp_table1_file_path():
    if __use_cpp_triple_product_tables_as_primary():
        return get_table1_file_path()
    else:
        return "precomp/table1_cpp.npy"

def get_cpp_table2_file_path():
    if __use_cpp_triple_product_tables_as_primary():
        return get_table2_file_path()
    else:
        return "precomp/table2_cpp.npy"

def get_cpp_table1_comparison_file_path():
    if __compare_cpp_triple_product_tables_to_python():
        return get_python_table1_file_path()
    else:
        return None

def get_cpp_table2_comparison_file_path():
    if __compare_cpp_triple_product_tables_to_python():
        return get_python_table2_file_path()
    else:
        return None

class Domain:
    def __init__(self, dim, max_level, bounds):
        """
        Class to store info on the state domain.
        
        :param      dim:        The system dimension
        :type       dim:        int
        :param      max_level:  The maximum level in each dimension
        :type       max_level:  np.array(int)
        :param      bounds:     The state bounds
        :type       bounds:     np.array
        """
        self.dim = dim
        self.bounds = bounds
        self.max_level = max_level