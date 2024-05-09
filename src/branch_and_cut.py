""" A Bender decomposition based Branch-and-Cut algorithm """

from gurobipy import *


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
        model = Model('Bender Decomposition MP')
        model.Params.OutputFlag = 0
        if relaxed:
            x = model.addVars(self.Stations.keys(), vtype=GRB.CONTINUOUS, name="x")
        else:
            x = model.addVars(self.Stations.keys(), vtype=GRB.INTEGER, name="x")

        slack_nH = model.addVars(self.Stations, ub=GRB.INFINITY, name='slack_nH')
        slack_nUN = model.addVars(self.Stations, ub=GRB.INFINITY, name='slack_nUN')

        model.setObjective(quicksum(10 ** -4 * self.W[3] * self.Cf_s[s] * x[s] for s in self.Stations), GRB.MINIMIZE)

        for s in self.Stations.keys():
            # max number of facilities to be open
            C4_max_facility = model.addConstr((quicksum(x[s]) <= self.NumberStMax), name='C_max_facility')

            # Cost of building facility does not exceed budget
            C5_budget = model.addConstr(x.prod(self.Cf_s) <= self.Budget, name="C_budget")

            # Ref Fig3a Canadian Arctic s8, s10 s11, s14 consider it as soft constraint to avoid infeasibitliy. so,
            # adding slack variable is a way to build soft const
            C6_HudsonFacility = model.addConstr((quicksum(x[s] + slack_nH[s]
                                                             for s in ['s8', 's10', 's11', 's14', 's17', 's19'])
                                                 >= self.nH), name='C_HudsonFacility')

            # Up North  s9, s12, s13, s15, s16, s18, s20
            C7_UpNorthFacility = model.addConstr((quicksum(x[s] - slack_nUN[s]
                                                              for s in ['s9', 's12', 's13', 's15', 's16', 's18', 's20'])
                                                  <= self.nUN),
                                                 name='C_UpNorthFacility')

        model.optimize()
        duals = {}

        if model.status == GRB.OPTIMAL:
            if relaxed:
                for o in self.OilSpills.keys():
                    duals[o] = model.getConstrByName(f"C1_{o}").Pi
        else:
            print("No solution found")
        return duals, model


    def solveSubploblem(self, duals):
        """ Solve the subploblem """
        model = Model('Bender Decomposition Sub-problem')
        model.Params.OutputFlag = 0
        var_dict = {}
        for o in duals.keys():
            var_dict[o] = model.addVar(vtype=GRB.INTEGER, name=f"y_{o}")

        model.setObjective(sum(var_dict[length] * dual_values for length, dual_values in duals.items()), GRB.MAXIMIZE)

        # model.addConstr(sum(var_dict[length] * length for length, dual_values in duals.items()) <= self.NumberStMax)

        model.optimize()

        # print(model.display())
        new_config = {}
        if model.status == GRB.OPTIMAL:
            reduced_cost = 1 - model.ObjVal
            if reduced_cost < -0.000000001:
                for o in duals.keys():
                    if var_dict[o].X > 0.00001:
                        new_config[o] = int(var_dict[o].X)
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
        pass