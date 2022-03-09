def get_dicts_from_numpy(nparray):

    if nparray.ndim is 1:
        array_dict = {(i+1):val for i,val in enumerate(nparray)}
    elif nparray.ndim is 2:
        array_dict = {(i+1,j+1):val \
                      for j,val in enumerate(row)\
                      for i,row in enumerate(nparray)}

    return array_dict
