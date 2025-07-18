from gurobipy import Model, GRB, quicksum
import pandas as pd


def build_model(Stations, OilSpills, Resources, Vehicles, W,
                v_o, eta_o, t_os, gamma, M, demand_or, demand_ov, L_p_or,
                NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, model_name):

    w1, w2, w3, w4, w5, w6, w7, w8 = W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7]

    # Create the model
    model = Model("Oil Spill Response Optimization")
    # Decision Variables
    x_s = model.addVars(Stations, vtype=GRB.BINARY, name="x_s")  # Station open decision
    y_os = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="y_os")  # Spill coverage
    z_sor = model.addVars(Stations, OilSpills, Resources, vtype=GRB.INTEGER, name="z_sor")  # Resource deployment
    h_sov = model.addVars(Stations, OilSpills, Vehicles, vtype=GRB.INTEGER, name="h_sov")  # Vehicle

    b_os = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="b_os")  # Binary penalty variable
    b_o_prime = model.addVars(OilSpills, vtype=GRB.BINARY, name="b_o_prime")  # Binary penalty variable
    # Constraint for b_o_prime based on eta_o
    model.addConstrs((b_o_prime[o] >= (1 if eta_o[o] >= 0.8 else 0) for o in OilSpills), name="C_b_o_prime")

    # Objective Function
    objective_1 = quicksum((w1 * v_o[o] + w2 * eta_o[o] - w3 * t_os[o, s]) * y_os[o, s]
                           for o in OilSpills for s in Stations)
    objective_2 = w4 * quicksum(F_s[s] * x_s[s] for s in Stations) \
                  + quicksum(
        (w5 * C_r[r] - w6 * Eff_sor[s, o, r]) * z_sor[s, o, r] for s in Stations for o in OilSpills for r in
        Resources) \
                  + quicksum(
        w7 * c_v[v] * Distance[(o, s)] + w8 * quicksum(pn_sor[s, o, r] for r in Resources) * h_sov[s, o, v]
        for s in Stations for o in OilSpills for v in Vehicles)

    model.setObjectiveN(objective_1, index=0, priority=3, weight=-1, name='objective_1')  # weight=-1 so maximize
    model.setObjectiveN(objective_2, index=1, priority=2, weight=1, name='objective_2')

    # Constraints
    model.addConstrs((y_os[o, s] <= x_s[s] for o in OilSpills for s in Stations), name="c3_facility_cover")  # (3)
    # Constraints 4 - 8
    model.addConstr(quicksum(x_s[s] for s in Stations) == NumberStMax, "c4_Max_Facilities")  # Constraint (4)
    # ++ <=  ==
    for o in OilSpills:
        for s in Stations:
            model.addConstr(
                Distance[o, s] * y_os[o, s] <= DistanceMax + M * b_o_prime[o] * (1 if eta_o[o] >= 0.8 else 0),
                name=f"c5_DistanceConstr_{o}_{s}")  # Constraint 5

    if model_name not in ['model_c', 'model_3']:
        model.addConstr(quicksum(x_s[s] for s in ['s8', 's10', 's11', 's14', 's17', 's19'])
                        >= nH, "c6_Hudson_Facilities")  # Constraint (6)
        model.addConstr(quicksum(x_s[s] for s in ['s9', 's12', 's13', 's15', 's16', 's18', 's20'])
                        <= nUN, "c7_Up_North")  # Constraint (7)

    for r in Resources:
        for s in Stations:
            model.addConstr(quicksum(z_sor[s, o, r] for o in OilSpills)
                            <= A_sr[s, r] * x_s[s], name=f"c8_Resource_Deploy_{s}_{r}")  # Constraint (8)

    # Constraint 9 - 13
    for o in OilSpills:
        model.addConstr((quicksum(y_os[o, s] for s in Stations) <= gamma), name=f"c9_max_facilities_{o}")  # (9)

    for o in OilSpills:
        model.addConstr(
            (quicksum(y_os[o, s] for s in Stations) >= 1 - M * b_o_prime[o] * (1 if eta_o[o] < 0.8 else 0)),
            name=f"c10_coverage_constraint_{o}")  # Constraint 10

    for o in OilSpills:
        for r in Resources:
            model.addConstr(quicksum(z_sor[s, o, r] for s in Stations)
                            <= demand_or[o, r] * quicksum(y_os[o, s] for s in Stations),  # +++
                            name=f"c11_resource_demand_{o}_{r}")  # (11)
    # (12) Minimum resources deployed per station
    for s in Stations:
        model.addConstr((quicksum(z_sor[s, o, r] for r in Resources for o in OilSpills)
                         >= nQ), name=f"c12_min_resource_deployment_{s}")

    # (13) Resources are only allocated if spill is covered
    for o in OilSpills:
        model.addConstr(quicksum(z_sor[s, o, r] for s in Stations for r in Resources)
                        <= 2*M * (1 - quicksum(b_os[o, s] for s in Stations)), name=f"c13_spill_resource_allocation_o{o}")

    # Constraint 14 - 17
    # (14) Limit on resources deployed
    for o in OilSpills:
        for r in ['c', 'i']:
            model.addConstr(quicksum(z_sor[s, o, r] for s in Stations) <= L_p_or[o, r],
                            name=f"c14_resource_limit_{o}_{r}")

    # (15) Vessel capacity constraint
    for o in OilSpills:
        for v in Vehicles:
            model.addConstr(quicksum(h_sov[s, o, v] for s in Stations) >= demand_ov[o, v],
                            name=f"c15_vessel_capacity_{o}_{v}")
    # (16) Vessel deployment from station to oil spill
    for s in Stations:
        for o in OilSpills:
            model.addConstr(quicksum(h_sov[s, o, v] for v in Vehicles) >= y_os[o, s],  # M *
                            name=f"c16_vessel_deployment_{s}_{o}")

    # (17) Resource deployment of vessels per facility
    for s in Stations:
        model.addConstr(quicksum(h_sov[s, o, v] for o in OilSpills for v in Vehicles)
                        <= quicksum(n_vs[v, s] for v in Vehicles), name=f"c17_facility_vessel_capacity_{s}")  # ++

    # (18) Vehicle Capacity
    for s in Stations:
        for o in OilSpills:
            for v in Vehicles:
                model.addConstr(
                    quicksum(z_sor[s, o, r] for r in Resources)
                    <= quicksum(Q_vr[v, r] for r in Resources) * h_sov[s, o, v], name=f"c18_capacity_link_{s}_{o}_{v}")
    # for s in Stations:
    #     for o in OilSpills:
    #         for v in Vehicles:
    #             for r in Resources:
    #                 model.addConstr(
    #                     z_sor[s, o, r] <= Q_vr[v, r] * h_sov[s, o, v],
    #                     name=f"c18_capacity_link_{s}_{o}_{v}")
    # Solve the model
    # model.write('../results/artifacts/model_lamoscad_july2025.lp')
    return model, x_s, y_os, z_sor, h_sov


def solve_model(model, x_s, y_os, z_sor, h_sov, OilSpills, needMultiSolutions=False, uncertaintyEvaluation=False,
                ):
    if needMultiSolutions:
        model.setParam('PoolSolutions', 20)  # useful for exploring pareto frontier
        model.setParam('PoolSearchMode', 2)

    if uncertaintyEvaluation:
        model.setParam('MIPGap', 0.05)  # 0.05% = 0.0005
        model.params.TimeLimit = 10 * 60


    model.params.OutPutFlag = 0
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. ")
        model.computeIIS()
        print("IIS Constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"{c.constrName}")
        # print("\nIIS Variables:")
        # for v in model.getVars():
        #     if v.IISLB or v.IISUB:
        #         print(f"{v.varName}: Lower Bound {v.IISLB}, Upper Bound {v.IISUB}")
        # model.write("../results/artifacts/infeasible_model.ilp")  # Save IIS to a file

    x_s1 = pd.Series(model.getAttr('X', x_s))[lambda x: x > 0.5]
    y_os1 = pd.Series(model.getAttr('X', y_os))[lambda x: x > 0.5]
    z_sor1 = pd.Series(model.getAttr('X', z_sor))[lambda x: x > 0]
    h_sov1 = pd.Series(model.getAttr('X', h_sov))[lambda x: x > 0]

    if needMultiSolutions:
        objVals = []
        for i in range(model.SolCount):
            model.setParam(GRB.Param.SolutionNumber, i)  # Select solution
            obj_values = [round(model.getObjective(j).getValue(), 2) for j in
                          range(model.NumObj)]  # Get all objective values
            objVals.append(obj_values)
        solution_values1 = []
        for i in range(model.SolCount):
            model.setParam(GRB.Param.SolutionNumber, i)  # Select solution
            obj_val = round(model.ObjNVal, 2)  # Get the objective value of the selected solution
            solution_values1.append(obj_val)
        solution_values1 = list(set(solution_values1))
        solution_values2 = [objVals[i][1] for i in range(len(solution_values1))]
        # Print all extracted solutions
        solution_values = [[solution_values1[i], solution_values2[i]] for i in range(len(solution_values1))]
        # print("Extracted Objective Values:", solution_values)
    else:
        objVals = [round(model.getObjective(0).getValue(), 2), round(model.getObjective(1).getValue(), 2)]
        solution_values = []

    print('objVals: ', objVals)
    coverage_percentage = int(100 * len(y_os1) / len(OilSpills))
    resource_stockpile_r = []

    return objVals, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values
