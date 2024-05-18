""" A Bender decomposition based Branch-and-Cut algorithm """

from gurobipy import *
import custom_func

class BranchAndCut(object):
    def __init__(self, Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill, SizeSpill_n,
                 Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMax, Distance,
                 Distance_n,
                 W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
                 BigM, MaxFO):
        self.Stations = Stations
        self.OilSpills = OilSpills
        self.ResourcesD = ResourcesD
        self.coordinates_st = coordinates_st
        self.coordinates_spill = coordinates_spill
        self.SizeSpill = SizeSpill
        self.SizeSpill_n = SizeSpill_n
        self.Demand = Demand
        self.Sensitivity_R = Sensitivity_R
        self.Sensitivity_n = Sensitivity_n
        self.Eff = Eff
        self.Effectiveness_n = Effectiveness_n
        self.Availability = Availability
        self.NumberStMax = NumberStMax
        self.Distance = Distance
        self.Distance_n = Distance_n
        self.W = W
        self.QuantityMin = QuantityMin
        self.DistanceMax = DistanceMax
        self.Cf_s = Cf_s
        self.CostU = CostU
        self.Budget = Budget
        self.BigM = BigM
        self.MaxFO = MaxFO

    def solveMasterProblem(self, relaxed=True):
        """ X_s is the binary decision variable to open facility or not """
        model_mp = Model('Bender Decomposition MP')
        model_mp.Params.OutputFlag = 0
        if relaxed:
            select = model_mp.addVars(self.Stations.keys(), vtype=GRB.CONTINUOUS, name="x")
        else:
            select = model_mp.addVars(self.Stations.keys(), vtype=GRB.INTEGER, name="x")

        slack_nH = model_mp.addVars(self.Stations, ub=GRB.INFINITY, name='slack_nH')
        slack_nUN = model_mp.addVars(self.Stations, ub=GRB.INFINITY, name='slack_nUN')

        model_mp.setObjective(quicksum(10 ** -4 * self.W[3] * self.Cf_s[s] * select[s] for s in self.Stations),
                              GRB.MINIMIZE)

        for s in self.Stations.keys():
            # max number of facilities to be open
            C4_max_facility = model_mp.addConstr((quicksum(select[s]) <= self.NumberStMax), name='C_max_facility')
            C5_budget = model_mp.addConstr(select.prod(self.Cf_s) <= self.Budget, name="C_budget")
            C6_HudsonFacility = model_mp.addConstr((quicksum(select[s] + slack_nH[s]
                                                             for s in ['s8', 's10', 's11', 's14', 's17', 's19'])
                                                    >= self.nH), name='C_HudsonFacility')
            C7_UpNorthFacility = model_mp.addConstr((quicksum(select[s] - slack_nUN[s]
                                                              for s in ['s9', 's12', 's13', 's15', 's16', 's18', 's20'])
                                                     <= self.nUN), name='C_UpNorthFacility')

        model_mp.optimize()
        duals = {}

        if model_mp.status == GRB.OPTIMAL:
            if relaxed:
                for o in self.OilSpills.keys():
                    duals[o] = model_mp.getConstrByName(f"C1_{o}").Pi
        else:
            print("No solution found")
        return duals, model_mp

    def solveSubploblem(self, duals):
        """ Solve the subploblem """
        model_sp = Model('Bender Decomposition Sub-problem')
        model_sp.Params.OutputFlag = 0
        cover = {}


        for o in duals.keys():
            cover[o] = model_sp.addVar(vtype=GRB.INTEGER, name=f"y_{o}")

        os_pair = {(o, s): custom_func.compute_distance(self.coordinates_spill[1][o], self.coordinates_st[1][s])
                   for o in self.OilSpills for s in self.Stations
                   if custom_func.compute_distance(tuple(self.coordinates_spill[1][o]),
                                                   tuple(self.coordinates_st[1][s])) < self.DistanceMax}
        os_pair = tuple(os_pair.keys())

        st_o = list(set([item[1] for item in os_pair]))
        sr_pair = []
        for s in st_o:
            for r in self.ResourcesD:
                sr_pair.append((s, r))
        sr_pair = tuple(sr_pair)

        osr_pair = {(o, s, r): custom_func.compute_distance(self.coordinates_spill[1][o], self.coordinates_st[1][s])
                    for o in self.OilSpills
                    for s in self.Stations
                    for r in self.ResourcesD
                    if
                    custom_func.compute_distance(tuple(self.coordinates_spill[1][o]),
                                                 tuple(self.coordinates_st[1][s])) < self.DistanceMax}
        osr_pair = tuple(osr_pair.keys())
        osr_C_pair = tuple(t for t in osr_pair if t[2] == 'c')
        osr_I_pair = tuple(t for t in osr_pair if t[2] == 'i')

        select = model_sp.addVars(st_o, vtype=GRB.BINARY, name='select')
        deploy = model_sp.addVars(osr_pair, vtype=GRB.CONTINUOUS, lb=0, name='deploy')
        vehicle = model_sp.addVars(osr_pair, vtype=GRB.BINARY, name='vehicle')

        slack_cLim = model_sp.addVars(osr_C_pair, ub=GRB.INFINITY, name='slack_cLim')
        slack_iLim = model_sp.addVars(osr_I_pair, ub=GRB.INFINITY, name='slack_iLim')

        # model_sp.setObjective(sum(cover[length] * dual_values for length, dual_values in duals.items()), GRB.MAXIMIZE)
        objective_1 = quicksum((self.W[0] * self.SizeSpill_n[o] + 100 * self.W[1] * self.Sensitivity_n[o] - self.W[2] * self.response_timeN[o, s])
                                  * cover[o, s] for o, s in os_pair)
        objective_3 = quicksum((10 ** -2 * self.W[4] * self.CostU[s, r] - 10 * self.W[5] * self.Effectiveness_n[s, r])
                                  * deploy[o, s, r] for o, s, r in osr_pair) \
                      + quicksum((self.W[6] * self.ct * self.Distance_n[o, s] + self.W[7] * self.pn_sor[s, o, r]
                                     * vehicle[o, s, r]) for o, s, r in osr_pair)
        model_sp.setObjectiveN(objective_1, GRB.MINIMIZE)
        model_sp.setObjectiveN(objective_3, GRB.MINIMIZE)

        C3_open_facility_to_cover = model_sp.addConstrs((cover[o, s] <= select[s]
                                                         for o, s in os_pair), name='C_open_facility_to_cover')

        C8_resource_capacity = model_sp.addConstrs(
            (deploy.sum('*', s, r) <= self.BigM * self.Availability[s, r] * select[s]  #
             for s, r in sr_pair), name='C_open_facility')
        C9_few_facility_per_spill = model_sp.addConstrs((cover.sum(o, '*') <= self.MaxFO
                                                         for o, s in os_pair),
                                                        name='C_few_facility_per_spill')  # ++partly solved
        C10_sensitivity = model_sp.addConstrs((cover.sum(o, '*') >= 0 for o, s in os_pair
                                               if self.Sensitivity_n[o] >= self.snT),
                                              name='C_sensitivity')
        C11_deploy_demand = model_sp.addConstrs((deploy[o, s, r] <= self.Demand[o, r] * cover[o, s]
                                                 for o, s, r in osr_pair), name='C_deploy_demand')
        C12_minimum_quantity = model_sp.addConstrs(deploy.sum('*', s, r) >= self.nQ for o, s, r in osr_pair)
        C14_chem_limit = model_sp.addConstrs(
            (deploy.sum('*', s, r) - slack_cLim[o, s, r] <= self.lc_p[o] for o, s, r in osr_C_pair),
            name='C_chem_limit')
        C15_burn_limit = model_sp.addConstrs(
            (deploy.sum('*', s, r) - slack_iLim[o, s, r] <= self.lc_i[o] for o, s, r in osr_I_pair),
            name='C_burn_limit')
        C17_vessel_route = model_sp.addConstrs((vehicle.sum('*', s, r) <= cover[o, s]
                                                for o, s, r in osr_pair), name='C_vessel_route')
        C17_vessel_route2 = model_sp.addConstrs((vehicle.sum(o, '*', r) <= cover[o, s]
                                                 for o, s, r in osr_pair), name='C_vessel_route2')
        model_sp.optimize()

        new_config = {}
        if model_sp.status == GRB.OPTIMAL:
            reduced_cost = 1 - model_sp.ObjVal
            if reduced_cost < -0.000000001:
                for o in duals.keys():
                    if cover[o].X > 0.00001:
                        new_config[o] = int(cover[o].X)
            else:
                new_config = None
        else:
            print("No solution found")
        return new_config

    def solve(self):
        iteration = 0
        while True:
            duals, model = self.solveMasterProblem(relaxed=True)
            new_config = self.solveSubploblem(duals)
            if new_config is not None:
                self.OilSpills[max(self.OilSpills.keys()) + 1] = new_config
            else:
                break
            iteration = iteration + 1

        duals, model = self.solveMasterProblem(relaxed=False)
        solution = []

        for o in self.OilSpills.keys():
            var = model.getVarByName(f"x[{o}]")
            if var.X > 0.0001:
                solution.append((self.OilSpills[o], round(var.X)))

        return solution

    def branch_and_cut(self):
        # for our problem Gurobi default solver (when we call model.optimize()) use BnC algorithm anyway
        pass
