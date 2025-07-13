from gurobipy import *

def solve_relaxed_master_problem(OilSpills, Stations, F_s, t_os, W, A_sr, b_prime, M, gamma, N, dual_cuts):
    w1, w2, w3_weight, w4, *_ = W

    model = Model("RMP")
    model.Params.OutputFlag = 0

    x = model.addVars(Stations, vtype=GRB.BINARY, name="x")
    y = model.addVars(OilSpills, Stations, vtype=GRB.BINARY, name="y")
    y_prime = model.addVar(lb=-GRB.INFINITY, name="y_prime")

    w3 = {(o, s): w3_weight * t_os[(o, s)] for o in OilSpills for s in Stations}

    obj = quicksum((w1 + w2 - w3[o, s]) * y[o, s] for o in OilSpills for s in Stations) \
          - w4 * quicksum(F_s[s] * x[s] for s in Stations) + y_prime
    model.setObjective(obj, GRB.MAXIMIZE)

    model.addConstrs(y[o, s] <= x[s] for o in OilSpills for s in Stations)
    model.addConstr(quicksum(x[s] for s in Stations) == N, name="facility_count")
    model.addConstrs(quicksum(y[o, s] for s in Stations) <= gamma for o in OilSpills)
    model.addConstrs(quicksum(y[o, s] for s in Stations) >= 1 - M * (1 - b_prime[o]) for o in OilSpills)

    for i, cut in enumerate(dual_cuts):
        model.addConstr(
            y_prime >= quicksum(cut['alpha'][s, r] * A_sr[s, r] * x[s] for s in Stations for r in Resources) +
                      quicksum(cut['beta'][o, r] * demand_or[o, r] * sum(y[o, s] for s in Stations) for o in OilSpills for r in Resources),
            name=f"benders_cut_{i}"
        )

    model.optimize()

    x_sol = {s: x[s].X for s in Stations}
    y_sol = {(o, s): y[o, s].X for o in OilSpills for s in Stations}
    return x_sol, y_sol, y_prime.X, model.ObjVal


def solve_primal_subproblem(OilSpills, Stations, Resources, Vehicles,
                             A_sr, Eff_sor, Distance, F_s, t_os, pn_sor,
                             demand_or, demand_ov, Q_vr, n_vs, L_p_or, b_os,
                             x_bar, y_bar, W, M, q):
    w5, w6, w7, w8 = W[4:]  # w5 to w8

    model = Model("SP")
    model.Params.OutputFlag = 0

    z = model.addVars(Stations, OilSpills, Resources, lb=0, vtype=GRB.CONTINUOUS, name="z")
    h = model.addVars(Stations, OilSpills, Vehicles, lb=0, vtype=GRB.CONTINUOUS, name="h")

    obj = quicksum((w5 * 1 - w6 * Eff_sor[s, o, r]) * z[s, o, r] for s in Stations for o in OilSpills for r in Resources)
    obj += quicksum((w7 * Distance[s, o] + w8 * sum(pn_sor[s, o, r] for r in Resources)) * h[s, o, v]
                    for s in Stations for o in OilSpills for v in Vehicles)
    model.setObjective(obj, GRB.MINIMIZE)

    con_alpha = model.addConstrs((quicksum(z[s, o, r] for o in OilSpills) <= A_sr[s, r] * x_bar[s]
                                  for s in Stations for r in Resources), name="con_alpha")

    con_beta = model.addConstrs((quicksum(z[s, o, r] for s in Stations) <= demand_or[o, r] * sum(y_bar[o, s] for s in Stations)
                                 for o in OilSpills for r in Resources), name="con_beta")

    con_gamma = model.addConstrs((quicksum(z[s, o, r] for o in OilSpills for r in Resources) >= q
                                 for s in Stations), name="con_gamma")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        z_sol = {(s, o, r): z[s, o, r].X for s in Stations for o in OilSpills for r in Resources}
        h_sol = {(s, o, v): h[s, o, v].X for s in Stations for o in OilSpills for v in Vehicles}
        duals = {
            "alpha": {(s, r): con_alpha[s, r].Pi for s in Stations for r in Resources},
            "beta": {(o, r): con_beta[o, r].Pi for o in OilSpills for r in Resources},
            "gamma": {s: con_gamma[s].Pi for s in Stations}
        }
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
                        A_sr, Eff_sor, Distance, F_s, t_os, pn_sor,
                        demand_or, demand_ov, Q_vr, n_vs, L_p_or, b_os, b_prime,
                        M, gamma, W, N, max_iters=50, tolerance=0.01, stable_iterations=3):

    dual_cuts = []
    UB, LB = float('inf'), -float('inf')
    best_solution = None
    stable_count = 0

    for it in range(max_iters):
        print(f"\n📘 Iteration {it+1}")

        x_sol, y_sol, y_prime_val, UB_candidate = solve_relaxed_master_problem(
            OilSpills, Stations, F_s, t_os, W, A_sr, b_prime, M, gamma, N, dual_cuts)

        z_sol, h_sol, SP_obj, duals = solve_primal_subproblem(
            OilSpills, Stations, Resources, Vehicles,
            A_sr, Eff_sor, Distance, F_s, t_os, pn_sor,
            demand_or, demand_ov, Q_vr, n_vs, L_p_or, b_os,
            x_sol, y_sol, W, M, W[3])

        LB = max(LB, SP_obj)
        UB = min(UB, UB_candidate)

        gap = (UB - LB) / max(abs(UB), 1e-6)
        print(f"GAP = {gap*100:.2f}% | LB = {LB:.3f}, UB = {UB:.3f}")

        if gap <= tolerance:
            stable_count += 1
            if stable_count >= stable_iterations:
                print("🎯 Converged!")
                best_solution = {"x": x_sol, "y": y_sol, "z": z_sol, "h": h_sol}
                break
        else:
            stable_count = 0

        violated, cut = generate_cuts(OilSpills, Stations, Resources, A_sr, demand_or, x_sol, y_sol, duals, y_prime_val)
        if violated:
            print("➕ Adding Benders cut")
            dual_cuts.append(cut)
        else:
            print("✔ No violated Benders cut found")

    if best_solution is None:
        best_solution = {"x": x_sol, "y": y_sol, "z": z_sol, "h": h_sol}

    return best_solution, LB, UB
