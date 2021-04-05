def get_dicts_from_numpy(nparray):

    array_dict = {(i+1):val for i,val in enumerate(nparray)}

    return array_dict
