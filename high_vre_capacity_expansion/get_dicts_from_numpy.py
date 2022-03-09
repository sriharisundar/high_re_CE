def get_dicts_from_numpy(nparray):

    if nparray.ndim == 1:
        array_dict = {(i+1):val for i,val in enumerate(nparray)}

    elif nparray.ndim == 2:
        array_dict = {(i+1,j+1):nparray[i,j] \
                      for j in range(nparray.shape[1])\
                      for i in range(nparray.shape[0])}

    else:
        return

    return array_dict
