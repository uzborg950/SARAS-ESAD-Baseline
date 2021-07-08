def get_dimensions(input_dim , layer_num):
    dims = [dim/(2**layer_num) for dim in input_dim]
    return (dims[0], dims[1])