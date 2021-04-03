"""Main module."""
import pyomo.environ as pyo


# TODO: 1. separate modules
#   2. Make code work when storage is False

def model_initialize(time_steps, demand, solar_nsites=0, wind_nsites=0, othergens_n=0, storage_included=True,
                     solar_params=None, wind_params=None, other_params=None, storage_params=None,
                     ):
    """

    :param time_steps:
    :param demand:
    :param solar_nsites:
    :param wind_nsites:
    :param othergens_n:
    :param storage_included:
    :param solar_params:
    :param wind_params:
    :param other_params:
    :param storage_params:
    :return:
    """
    model = pyo.ConcreteModel()

    model.time = pyo.RangeSet(time_steps[0], time_steps[1])
    model.time_storage = pyo.RangeSet(time_steps[0] - 1, time_steps[1])

    model.solar_nsites = solar_nsites
    model.wind_nsites = wind_nsites
    model.othergens_n = othergens_n
    model.storage_included = storage_included

    model.solar_sitelist = pyo.RangeSet(model.solar_nsites)
    model.solar_capacities = pyo.Var(model.solar_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.solar_generation = pyo.Var(model.solar_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.solar_nsites != 0:
        model.InstallCost_solar = pyo.Param(initialize=solar_params['InstallCost_solar'])
        model.VarCost_solar = pyo.Param(initialize=solar_params['VarCost_solar'])
        model.solar_potential = pyo.Param(model.solar_sitelist, model.time, initialize=solar_params['solar_potential'])

    model.wind_sitelist = pyo.RangeSet(model.wind_nsites)
    model.wind_capacities = pyo.Var(model.wind_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.wind_generation = pyo.Var(model.wind_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.wind_nsites != 0:
        model.InstallCost_wind = pyo.Param(initialize=wind_params['InstallCost_wind'])
        model.VarCost_wind = pyo.Param(initialize=wind_params['VarCost_wind'])
        model.wind_potential = pyo.Param(model.wind_sitelist, model.time, initialize=wind_params['wind_potential'])

    model.othergens_sitelist = pyo.RangeSet(model.othergens_n)
    model.other_capacities = pyo.Var(model.othergens_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.other_generation = pyo.Var(model.othergens_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.othergens_n != 0:
        model.InstallCost_other = pyo.Param(model.othergens_sitelist, initialize=other_params['InstallCost_other'])
        model.VarCost_other = pyo.Param(model.othergens_sitelist, initialize=other_params['VarCost_other'])
        model.other_CF = pyo.Param(model.othergens_sitelist, initialize=other_params['other_CF'])
        model.other_maxusage = pyo.Param(initialize=other_params['other_maxusage'])

    model.storage_capacities = pyo.Var(initialize=0, domain=pyo.NonNegativeReals)
    model.storage_state = pyo.Var(model.time_storage, initialize=0, domain=pyo.NonNegativeReals)
    model.storage_charge = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    model.storage_discharge = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    if model.storage_included is True:
        model.InstallCost_storage = pyo.Param(initialize=storage_params['InstallCost_storage'])
        model.VarCost_storage = pyo.Param(initialize=storage_params['VarCost_storage'])
        model.storage_EP_ratio = pyo.Param(initialize=storage_params['EP_ratio'])
        model.storage_round_trip_efficiency = pyo.Param(initialize=storage_params['round_trip_efficiency'])
        model.storage_decay_rate = pyo.Param(initialize=storage_params['decay_rate'])

    model.demand = pyo.Param(pyo.RangeSet(1),model.time, initialize=demand)

    return model


def set_model_objective(model):
    """

    :param model:
    :return:
    """

    expr_solar_capacitycost = sum(model.InstallCost_solar * model.solar_capacities[i] for i in model.solar_sitelist)
    expr_solar_varcost = sum(model.VarCost_solar * model.solar_generation[i, t]
                             for i in model.solar_sitelist for t in model.time)

    expr_wind_capacitycost = sum(model.InstallCost_wind * model.wind_capacities[i] for i in model.wind_sitelist)
    expr_wind_varcost = sum(model.VarCost_wind * model.wind_generation[i, t]
                            for i in model.wind_sitelist for t in model.time)

    expr_other_capacitycost = sum(model.InstallCost_other[i] * model.other_capacities[i]
                                  for i in model.othergens_sitelist)
    expr_other_varcost = sum(model.VarCost_other[i] * model.other_generation[i, t]
                             for i in model.othergens_sitelist for t in model.time)

    expr_storage_capacitycost = model.InstallCost_storage * model.storage_capacities
    expr_storage_varcost = sum(model.VarCost_storage * (model.storage_charge[t] + model.storage_discharge[t])
                               for t in model.time)

    expr = expr_solar_capacitycost + expr_solar_varcost\
           + expr_wind_capacitycost + expr_wind_varcost\
           + expr_other_capacitycost + expr_other_varcost\
           + expr_storage_capacitycost + expr_storage_varcost

    model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

    return


def set_model_solar_constraints(model):
    model.solar_gen_constraint = pyo.ConstraintList()
    for t in model.time:
        for i in model.solar_sitelist:
            model.solar_gen_constraint.add(model.solar_generation[i, t]
                                           <= model.solar_capacities[i] * model.solar_potential[i, t])
    return


def set_model_wind_constraints(model):
    model.wind_gen_constraint = pyo.ConstraintList()
    for t in model.time:
        for i in model.wind_sitelist:
            model.wind_gen_constraint.add(model.wind_generation[i, t]
                                          <= model.wind_capacities[i] * model.wind_potential[i, t])
    return


def set_model_othergen_constraints(model):
    model.other_gen_constraint = pyo.ConstraintList()
    expr = sum(model.other_generation[i, t] for i in model.othergens_sitelist for t in model.time) <= \
           model.other_maxusage * sum(model.demand[1,t] for t in model.time)
    model.other_gen_constraint.add(expr)
    for t in model.time:
        for i in model.othergens_sitelist:
            model.other_gen_constraint.add(model.other_generation[i, t]
                                           <= model.other_CF[i] * model.other_capacities[i])

    return


def set_model_storage_constraints(model):
    model.storage_constraint = pyo.ConstraintList()
    for t in model.time:
        if t == 1:
            model.storage_constraint.add(model.storage_state[t] == 0)
        else:
            model.storage_constraint.add(model.storage_state[t] == (1-model.storage_decay_rate)*model.storage_state[t-1]
                                         + model.storage_round_trip_efficiency*model.storage_charge[t]
                                         - model.storage_discharge[t])

        model.storage_constraint.add(model.storage_charge[t] <= model.storage_capacities)
        model.storage_constraint.add(model.storage_discharge[t] <= model.storage_capacities)
        model.storage_constraint.add(model.storage_state[t] <= model.storage_EP_ratio*model.storage_capacities)

    return


def set_model_demand_constraints(model):
    model.demand_constraint = pyo.ConstraintList()

    for t in model.time:
        solar_gen_t = sum(model.solar_generation[i, t] for i in model.solar_sitelist)
        wind_gen_t = sum(model.wind_generation[i, t] for i in model.wind_sitelist)
        other_gen_t = sum(model.other_generation[i, t] for i in model.othergens_sitelist)
        storage_t = - model.storage_charge[t] + model.storage_discharge[t]

        model.demand_constraint.add(solar_gen_t + wind_gen_t + storage_t + other_gen_t == model.demand[1,t])

    return
