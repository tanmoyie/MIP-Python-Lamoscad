from gurobipy import *

def solve_master_problem(OilSpills, Stations, Resources, F_s, v_o, eta_o, t_os, demand_or, W, A_sr, M, gamma, NumberStMax, dual_cuts):
    w1, w2, w3, w4, *_ = W

    model = Model("RMP")
    model.Params.OutputFlag = 0

    x_s = model.addVars(Stations, vtype=GRB.BINARY, name="x")
    y_os = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="y")
    y_prime = model.addVar(lb=-GRB.INFINITY, name="y_prime")
    b_prime = model.addVars(OilSpills, vtype=GRB.BINARY, name="b_prime")

    obj_mp = quicksum((w1 * v_o[o] + w2 * eta_o[o] - w3 * t_os[o, s]) * y_os[o, s]
                           for o in OilSpills for s in Stations) \
          - w4 * quicksum(F_s[s] * x_s[s] for s in Stations) # + y_prime

    model.setObjective(obj_mp, GRB.MAXIMIZE)

    model.addConstrs(y_os[o, s] <= x_s[s] for o in OilSpills for s in Stations)
    model.addConstr(quicksum(x_s[s] for s in Stations) == NumberStMax, name="facility_count")
    model.addConstrs(quicksum(y_os[o, s] for s in Stations) <= gamma for o in OilSpills)
    model.addConstrs(quicksum(y_os[o, s] for s in Stations) >= 1 - M * (1 - b_prime[o]) for o in OilSpills)

    for i, cut in enumerate(dual_cuts):
        model.addConstr(
            y_prime >= quicksum(cut['alpha'][s, r] * A_sr[s, r] * x_s[s] for s in Stations for r in Resources) +
                      quicksum(cut['beta'][o, r] * demand_or[o, r] * sum(y_os[o, s] for s in Stations) for o in OilSpills for r in Resources),
            name=f"benders_cut_{i}")

    model.optimize()
    print('MP obj: ', model.ObjVal)
    milp_obj1_from_mp = sum((w1 * v_o[o] + w2 * eta_o[o] - w3 * t_os[o, s]) * y_os[o, s].X
                            for o in OilSpills for s in Stations)
    print("MILP obj equivalent: :", milp_obj1_from_mp)
    x_bar = {s: x_s[s].X for s in Stations}
    y_bar = {(o, s): y_os[o, s].X for o in OilSpills for s in Stations}
    return x_bar, y_bar, y_prime.X, model.ObjVal, milp_obj1_from_mp


def solve_primal_subproblem(OilSpills, Stations, Resources, Vehicles,
                             A_sr, Eff_sor, Distance, t_os, pn_sor, C_r,
                             demand_or, demand_ov, Q_vr, n_vs, L_p_or,
                             x_bar, y_bar, W, M, nQ):
    w5, w6, w7, w8 = W[4:]

    model = Model("SP")
    model.Params.OutputFlag = 0

    z = model.addVars(Stations, OilSpills, Resources, lb=0, vtype=GRB.CONTINUOUS, name="z")
    h = model.addVars(Stations, OilSpills, Vehicles, lb=0, vtype=GRB.CONTINUOUS, name="h")
    b_os = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="b_os")

    obj = quicksum((w5 *  C_r[r]  - w6 * Eff_sor[s, o, r]) * z[s, o, r] for s in Stations for o in OilSpills for r in Resources)
    obj += quicksum((w7 * Distance[o, s] + w8 * sum(pn_sor[s, o, r] for r in Resources)) * h[s, o, v]
                    for s in Stations for o in OilSpills for v in Vehicles)
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstrs(quicksum(z[s, o, r] for o in OilSpills) <= A_sr[s, r] * x_bar[s] for s in Stations for r in Resources)
    model.addConstrs(quicksum(z[s, o, r] for s in Stations) <= demand_or[o, r] * sum(y_bar[o, s] for s in Stations) for o in OilSpills for r in Resources)
    model.addConstrs(quicksum(z[s, o, r] for o in OilSpills for r in Resources) >= nQ for s in Stations)
    model.addConstrs(quicksum(z[s, o, r] for s in Stations for r in Resources) <= M * (1 - quicksum(b_os[o, s] for s in Stations)) for o in OilSpills)
    model.addConstrs(quicksum(z[s, o, r] for s in Stations) <= L_p_or[o, r] for o in OilSpills for r in Resources if r in ["c", "i"])
    model.addConstrs(quicksum(h[s, o, v] for s in Stations) >= demand_ov[o, v] for o in OilSpills for v in Vehicles)
    model.addConstrs(quicksum(h[s, o, v] for v in Vehicles) <= M * y_bar[o, s] for s in Stations for o in OilSpills)
    model.addConstrs(quicksum(h[s, o, v] for o in OilSpills for v in Vehicles) <= sum(n_vs[v, s] for v in Vehicles) for s in Stations)
    model.addConstrs(quicksum(z[s, o, r] for r in Resources) <= quicksum(Q_vr[v, r] * h[s, o, v] for v in Vehicles for r in Resources) for s in Stations for o in OilSpills)

    model.optimize()
    print('SP obj: ', model.ObjVal)

    if model.Status == GRB.OPTIMAL:
        z_sol = {(s, o, r): z[s, o, r].X for s in Stations for o in OilSpills for r in Resources}
        h_sol = {(s, o, v): h[s, o, v].X for s in Stations for o in OilSpills for v in Vehicles}
        duals = 'n/a'  # \
        #     {
        #     "alpha": {(s, r): model.getConstrs()[i].Pi for i, (s, r) in enumerate([(s, r) for s in Stations for r in Resources])},
        #     "beta": {(o, r): model.getConstrs()[i].Pi for i, (o, r) in enumerate([(o, r) for o in OilSpills for r in Resources], start=len(Stations)*len(Resources))},
        #     "gamma": {s: model.getConstrs()[i].Pi for i, s in enumerate(Stations, start=2*len(Stations)*len(Resources))}
        # }
        return z_sol, h_sol, model.ObjVal, duals
    else:
        return None, None, float('inf'), None


def generate_cuts(OilSpills, Stations, Resources, A_sr, demand_or, x_bar, y_bar, duals, y_prime_val, epsilon=1e-4):
    alpha = duals['alpha']
    beta = duals['beta']

    lhs = sum(alpha[s, r] * A_sr[s, r] * x_bar[s] for s in Stations for r in Resources)
    lhs += sum(beta[o, r] * demand_or[o, r] * sum(y_bar[o, s] for s in Stations) for o in OilSpills for r in Resources)

    violated = lhs > y_prime_val + epsilon
    cut = {"alpha": alpha, "beta": beta, "lhs": lhs}

    return violated, cut


def branch_and_cut_loop(OilSpills, Stations, Resources, Vehicles,
                        A_sr, C_r, Eff_sor, Distance, F_s,  v_o, eta_o, t_os, pn_sor,
                        demand_or, demand_ov, nQ, Q_vr, n_vs, L_p_or, M, gamma, W, NumberStMax,
                        max_iters = 50, tolerance = 0.01, stable_iterations = 3):

    dual_cuts = []
    UB, LB = float('inf'), -float('inf')
    best_solution = None
    stable_count = 0

    for it in range(max_iters):
        print(f"\n📘 Iteration {it+1}")

        x_bar, y_bar, y_prime_val, UB_candidate, milp_obj1_from_mp = solve_master_problem(
            OilSpills, Stations, Resources, F_s, v_o, eta_o, t_os, demand_or, W, A_sr, M, gamma, NumberStMax, dual_cuts)

        z_sol, h_sol, SP_obj, duals = solve_primal_subproblem(OilSpills, Stations, Resources, Vehicles,
                             A_sr, Eff_sor, Distance, t_os, pn_sor, C_r,
                             demand_or, demand_ov, Q_vr, n_vs, L_p_or,
                             x_bar, y_bar, W, M, nQ)

        LB = max(LB, SP_obj)
        UB = min(UB, UB_candidate)

        # gap = (UB - LB) / max(abs(UB), 1e-6)
        # print(f"GAP = {gap*100:.2f}% | LB = {LB:.3f}, UB = {UB:.3f}")
        #
        # if gap <= tolerance:
        #     stable_count += 1
        #     if stable_count >= stable_iterations:
        #         print("🎯 Converged!")
        #         best_solution = {"x": x_bar, "y": y_bar, "z": z_sol, "h": h_sol}
        #         break
        # else:
        #     stable_count = 0
        #
        # violated, cut = generate_cuts(OilSpills, Stations, Resources, A_sr, demand_or, x_bar, y_bar, duals, y_prime_val)
        # if violated:
        #     print("➕ Adding Benders cut")
        #     dual_cuts.append(cut)
        # else:
        #     print("✔ No violated Benders cut found")

    if best_solution is None:
        best_solution = {"x": x_bar, "y": y_bar, "z": z_sol, "h": h_sol}

    return best_solution, LB, UB, milp_obj1_from_mp
