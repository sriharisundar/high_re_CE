def get_dicts_from_dataframe(dframe):
    names = dframe.columns.to_list()

    dframe_np = dframe.to_numpy().T
    param_set = {(i + 1, t + 1): dframe_np[i, t] \
                 for t in range(dframe_np.shape[1]) \
                 for i in range(dframe_np.shape[0]) \
                 }

    return names, param_set
