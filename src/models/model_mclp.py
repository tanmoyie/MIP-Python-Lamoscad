""" MCLP model """

# %% Data
# data processing libraries
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time


def build_model_mclp(OilSpills, Stations, v_o, NumberStMax, Distance, DistanceMax):
    print('--------------MCLP--------')
    model = Model("MCLP classical")
    # ---------------------------------------- Decision variable ---------------------------------------------------
    y_os = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="y_os")  # Spill coverage
    x_s = model.addVars(Stations, vtype=GRB.BINARY, name="x_s")  # Station open decision

    # ------------------------------------------------ Constraints -------------------------------------------------
    # C1: facility must be open to cover oil spill
    model.addConstrs((y_os[o, s] <= x_s[s] for o in OilSpills for s in Stations), name='C1_open_facility_to_cover')
    # C2
    model.addConstrs((quicksum(y_os[o, s] for s in Stations) <= 1 for o in OilSpills), name='C2_open_facility_to_cover')
    # C3
    for o in OilSpills:
        for s in Stations:
            model.addConstr(Distance[o, s] * y_os[o, s] <= 0.8*DistanceMax, name=f"C3_DistanceConstr_{o}_{s}")
    # C4: max number of facilities to be open
    model.addConstr((quicksum(x_s[s] for s in Stations) == NumberStMax), name='C4_max_facility')

    # ----------------------------------------------- Objective function -------------------------------------------
    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(quicksum(v_o[o] * y_os[o, s] for o in OilSpills for s in Stations))
    # model.write(f'../results/artifacts/model_mclp.lp')

    # Solve the model
    return model, x_s, y_os


def solve_model_mclp(model_mclp, x_s, y_os, OilSpills):
    start_time = time.time()
    model_mclp.params.OutputFlag = 0
    model_mclp.params.TimeLimit = 10*60
    model_mclp.optimize()
    runtime_mclp = time.time() - start_time
    x_s1 = pd.Series(model_mclp.getAttr('X', x_s))[lambda x: x > 0.5]
    y_os1_mclp = pd.Series(model_mclp.getAttr('X', y_os))[lambda x: x > 0.5]

    # print('y_os1 mclp: ', y_os1_mclp)
    coverage_percentage = int(100 * len(y_os1_mclp) / len(OilSpills))
    print(f'MCLP Coverage Percentage: {coverage_percentage}%')

    number_facility_selected = len(x_s1)
    num_sensitive_spills = 21

    return (model_mclp, y_os1_mclp, int(coverage_percentage), int(num_sensitive_spills),
            runtime_mclp, int(number_facility_selected))
