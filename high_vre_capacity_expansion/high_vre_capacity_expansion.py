"""Main module."""
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


# TODO: 1. separate modules
#   2. Make code work when storage is False
#   3. Include hydro - Large portions of WECC are highly dependent on this

def model_initialize(time_steps, demand, solar_nsites=0, wind_nsites=0, othergens_n=0, storage_n=0,
                     lossofload_penalty=1e20,
                     solar_params=None, wind_params=None, other_params=None, storage_params=None
                     ):
    """

    :param lossofload_penalty:
    :param time_steps:
    :param demand:
    :param solar_nsites:
    :param wind_nsites:
    :param othergens_n:
    :param storage_n:
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
    model.storage_n = storage_n
    model.lossofload_penalty = pyo.Param(initialize=lossofload_penalty)

    model.lossofload = pyo.Var(model.time, initialize=0, domain=pyo.NonNegativeReals)

    model.solar_sitelist = pyo.RangeSet(model.solar_nsites)
    model.solar_capacities = pyo.Var(model.solar_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.solar_generation = pyo.Var(model.solar_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.solar_nsites != 0:
        model.InstallCost_solar = pyo.Param(initialize=solar_params['InstallCost_solar'])
        model.VarCost_solar = pyo.Param(initialize=solar_params['VarCost_solar'])
        model.solar_sitearea = pyo.Param(model.solar_sitelist, initialize=solar_params['solar_sitearea'])
        model.solar_capacitycap = pyo.Param(model.solar_sitelist, initialize=solar_params['solar_capacitycap'])
        # capacity density
        model.solar_site_CD = pyo.Param(model.solar_sitelist, initialize=solar_params['solar_site_CD'])
        model.solar_potential = pyo.Param(model.solar_sitelist, model.time, initialize=solar_params['solar_potential'])

    model.wind_sitelist = pyo.RangeSet(model.wind_nsites)
    model.wind_capacities = pyo.Var(model.wind_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.wind_generation = pyo.Var(model.wind_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.wind_nsites != 0:
        model.InstallCost_wind = pyo.Param(initialize=wind_params['InstallCost_wind'])
        model.VarCost_wind = pyo.Param(initialize=wind_params['VarCost_wind'])
        model.wind_sitearea = pyo.Param(model.wind_sitelist, initialize=wind_params['wind_sitearea'])
        model.wind_capacitycap = pyo.Param(model.wind_sitelist, initialize=wind_params['wind_capacitycap'])
        # capacity density
        model.wind_site_CD = pyo.Param(model.wind_sitelist, initialize=wind_params['wind_site_CD'])
        model.wind_potential = pyo.Param(model.wind_sitelist, model.time, initialize=wind_params['wind_potential'])

    model.othergens_sitelist = pyo.RangeSet(model.othergens_n)
    model.other_capacities = pyo.Var(model.othergens_sitelist, initialize=0, domain=pyo.NonNegativeReals)
    model.other_generation = pyo.Var(model.othergens_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.othergens_n != 0:
        model.InstallCost_other = pyo.Param(model.othergens_sitelist, initialize=other_params['InstallCost_other'])
        model.VarCost_other = pyo.Param(model.othergens_sitelist, initialize=other_params['VarCost_other'])
        model.other_CF = pyo.Param(model.othergens_sitelist, initialize=other_params['other_CF'])
        model.other_capacitycap = pyo.Param(model.othergens_sitelist, initialize=other_params['other_capacitycap'])
        model.other_maxusage = pyo.Param(initialize=other_params['other_maxusage'])

    model.storage_sitelist = pyo.RangeSet(model.storage_n)
    model.storage_capacities = pyo.Param(model.storage_sitelist, initialize=storage_params['storage_capacity'])
    model.storage_state = pyo.Var(model.storage_sitelist, model.time_storage, initialize=0, domain=pyo.NonNegativeReals)
    model.storage_charge = pyo.Var(model.storage_sitelist, model.time, domain=pyo.NonNegativeReals)
    model.storage_discharge = pyo.Var(model.storage_sitelist, model.time, domain=pyo.NonNegativeReals)
    if model.storage_n != 0:
        model.InstallCost_storage = pyo.Param(model.storage_sitelist, initialize=storage_params['InstallCost_storage'])
        model.VarCost_storage = pyo.Param(model.storage_sitelist, initialize=storage_params['VarCost_storage'])
        model.storage_EP_ratio = pyo.Param(model.storage_sitelist, initialize=storage_params['EP_ratio'])
        model.storage_round_trip_efficiency = pyo.Param(model.storage_sitelist,
                                                        initialize=storage_params['round_trip_efficiency'])
        model.storage_decay_rate = pyo.Param(model.storage_sitelist, initialize=storage_params['decay_rate'])
        # Decay rate is for hourly decay, so if the period is different, this has to be changed.

    model.demand = pyo.Param(model.time, initialize=demand)

    return model


def set_objective_capacity_expansion(model):
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

    expr_storage_capacitycost = sum(model.InstallCost_storage[i] * model.storage_capacities[i]
                                    for i in model.storage_sitelist)
    expr_storage_varcost = sum(model.VarCost_storage[i] * (model.storage_charge[i, t] + model.storage_discharge[i, t])
                               for i in model.storage_sitelist for t in model.time)

    expr_totallossofload = sum(model.lossofload_penalty * model.lossofload[t]
                               for t in model.time)

    expr = expr_solar_capacitycost + expr_solar_varcost \
           + expr_wind_capacitycost + expr_wind_varcost \
           + expr_other_capacitycost + expr_other_varcost \
           + expr_storage_capacitycost + expr_storage_varcost \
           + expr_totallossofload

    model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

    return


def set_objective_economic_dispatch(model):
    expr_solar_varcost = sum(model.VarCost_solar * model.solar_generation[i, t]
                             for i in model.solar_sitelist for t in model.time)

    expr_wind_varcost = sum(model.VarCost_wind * model.wind_generation[i, t]
                            for i in model.wind_sitelist for t in model.time)

    expr_other_varcost = sum(model.VarCost_other[i] * model.other_generation[i, t]
                             for i in model.othergens_sitelist for t in model.time)

    expr_storage_varcost = sum(model.VarCost_storage[i] * (model.storage_charge[i, t] + model.storage_discharge[i, t])
                               for i in model.storage_sitelist for t in model.time)

    expr_totallossofload = sum(model.lossofload_penalty * model.lossofload[t]
                               for t in model.time)

    expr = expr_solar_varcost + expr_wind_varcost \
           + expr_other_varcost + expr_storage_varcost \
           + expr_totallossofload

    model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

    return


def set_model_solar_constraints(model):
    model.solar_gen_constraint = pyo.ConstraintList()
    for i in model.solar_sitelist:
        model.solar_gen_constraint.add(model.solar_capacities[i] <= model.solar_sitearea[i] * model.solar_site_CD[i])
        model.solar_gen_constraint.add(model.solar_capacities[i] <= model.solar_capacitycap[i])
    for t in model.time:
        for i in model.solar_sitelist:
            model.solar_gen_constraint.add(model.solar_generation[i, t]
                                           <= model.solar_capacities[i] * model.solar_potential[i, t])
    return


def set_model_wind_constraints(model):
    model.wind_gen_constraint = pyo.ConstraintList()
    for i in model.wind_sitelist:
        model.wind_gen_constraint.add(model.wind_capacities[i] <= model.wind_sitearea[i] * model.wind_site_CD[i])
        model.wind_gen_constraint.add(model.wind_capacities[i] <= model.wind_capacitycap[i])
    for t in model.time:
        for i in model.wind_sitelist:
            model.wind_gen_constraint.add(model.wind_generation[i, t]
                                          <= model.wind_capacities[i] * model.wind_potential[i, t])
    return


def set_model_othergen_constraints(model):
    model.other_gen_constraint = pyo.ConstraintList()
    expr = sum(model.other_generation[i, t] for i in model.othergens_sitelist for t in model.time) <= \
           model.other_maxusage * sum(model.demand[t] for t in model.time)
    model.other_gen_constraint.add(expr)

    for i in model.othergens_sitelist:
        model.other_gen_constraint.add(model.other_capacities[i] <= model.other_capacitycap[i])

    for t in model.time:
        for i in model.othergens_sitelist:
            model.other_gen_constraint.add(model.other_generation[i, t]
                                           <= model.other_CF[i] * model.other_capacities[i])

    return


def set_model_storage_constraints(model):
    model.storage_constraint = pyo.ConstraintList()
    for t in model.time:
        for i in model.storage_sitelist:
            if t == 1:
                model.storage_constraint.add(model.storage_state[i, t] == 0)
            else:
                model.storage_constraint.add(
                    model.storage_state[i, t] == (1 - model.storage_decay_rate[i]) * model.storage_state[i, t - 1]
                    + model.storage_round_trip_efficiency[i] ** 0.5 * model.storage_charge[i, t]
                    - model.storage_round_trip_efficiency[i] ** 0.5 * model.storage_discharge[i, t])

            model.storage_constraint.add(model.storage_charge[i, t] <= model.storage_capacities[i])
            model.storage_constraint.add(model.storage_discharge[i, t] <= model.storage_capacities[i])
            model.storage_constraint.add(
                model.storage_state[i, t] <= model.storage_EP_ratio[i] * model.storage_capacities[i])

    return


def set_model_demand_constraints(model):
    model.demand_constraint = pyo.ConstraintList()

    for t in model.time:
        solar_gen_t = sum(model.solar_generation[i, t] for i in model.solar_sitelist)
        wind_gen_t = sum(model.wind_generation[i, t] for i in model.wind_sitelist)
        other_gen_t = sum(model.other_generation[i, t] for i in model.othergens_sitelist)
        storage_t = sum(- model.storage_charge[i, t] + model.storage_discharge[i, t] for i in model.storage_sitelist)

        model.demand_constraint.add(
            model.demand[t] - (solar_gen_t + wind_gen_t + storage_t + other_gen_t) == model.lossofload[t])

    return


def set_planning_reserve_margin_constraint(model, time_step_max_demand):
    model.prm_constraint = pyo.ConstraintList()

    t = time_step_max_demand

    solar_gen_t = sum(model.solar_capacities[i] * model.solar_potential[i, t] for i in model.solar_sitelist)
    wind_gen_t = sum(model.wind_capacities[i] * model.wind_potential[i, t] for i in model.wind_sitelist)
    other_gen_t = sum(model.other_CF[i] * model.other_capacities[i] for i in model.othergens_sitelist)
    storage_t = sum(- model.storage_charge[i, t] + model.storage_discharge[i, t] for i in model.storage_sitelist)

    model.prm_constraint.add((solar_gen_t + wind_gen_t + storage_t + other_gen_t) <= 1.15 * model.demand[t])

    return


def collect_resutls(model, results):
    if (results.solver.status == SolverStatus.ok) and \
        (results.solver.termination_condition == TerminationCondition.optimal):

        installed_capacities = dict()
        installed_capacities['Solar'] = np.array([model.solar_capacities[i]() for i in model.solar_sitelist])
        installed_capacities['Wind'] = np.array([model.wind_capacities[i]() for i in model.wind_sitelist])
        installed_capacities['Natural gas'] = np.array([model.other_capacities[i]() for i in model.othergens_sitelist])
        installed_capacities['Storage power'] = np.array([model.storage_capacities[i]()
                                                          for i in model.storage_sitelist])

        cost = model.obj()

        hourly_outputs = dict()
        hourly_outputs['solar_generation'] = pd.DataFrame(np.array([[model.solar_generation[i, t]() for t in model.time]
                                                                    for i in model.solar_sitelist]).T,
                                                          index=[t for t in model.time])

        hourly_outputs['wind_generation'] = pd.DataFrame(np.array([[model.wind_generation[i, t]() for t in model.time]
                                                                   for i in model.wind_sitelist]).T,
                                                         index=[t for t in model.time])

        hourly_outputs['other_generation'] = pd.DataFrame(np.array([[model.other_generation[i, t]() for t in model.time]
                                                                    for i in model.othergens_sitelist]).T,
                                                          index=[t for t in model.time])

        hourly_outputs['storage'] = pd.DataFrame(np.array([[model.storage_charge[1, t]() for t in model.time],
                                                           [model.storage_discharge[1, t]() for t in model.time],
                                                           [model.storage_state[1, t]() for t in model.time]]).T,
                                                 columns=['charge', 'discharge', 'state'],
                                                 index=[t for t in model.time])

        hourly_outputs['loss_of_load'] = pd.DataFrame([model.lossofload[t]() for t in model.time],
                                                      columns=['Unserved'],
                                                      index=[t for t in model.time])

        return installed_capacities, hourly_outputs, cost

    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # Do something when model in infeasible
        raise RuntimeWarning("Model infeasible")

    else:
        # Something else is wrong
        raise RuntimeWarning("Some other error occurred")
