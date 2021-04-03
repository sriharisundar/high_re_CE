def get_dicts_from_dataframe(dframe):
    names = dframe.columns.to_list()

    param_set = {}
    for i in range(1, dframe.columns.size+1):
        for t in range(1, dframe.index.size+1):
            param_set[(i, t)] = dframe.iloc[t-1][dframe.columns[i-1]]

    return names, param_set
