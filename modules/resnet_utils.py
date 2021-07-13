import math
def get_dimensions(input_dim , layer_num):
    dims = [math.ceil(dim/(2**layer_num)) for dim in input_dim]
    return (dims[0], dims[1])