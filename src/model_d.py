""" Optimization model: maximize demand and minimize cost: MIP
Outline:
1. Define Decision variables
2. Add Constraints
3. Add objective functions
4. Set some gurobi parameters, & write the model
5. Solve the model
6. Write log file
7. Get some variables out of the model for further analysis

Developer: Tanmoy Das
Date: July 05, 2023 (revise May 2024) """

# %% Data
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import custom_func
import math
from datetime import datetime, date


# %% Model 2: MIP-2
class ModelD:
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


    def solve(Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill, SizeSpill_n,
              Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, pn_sor, Availability, NumberStMax, Distance,
              Distance_n,
              W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
              BigM, MaxFO):

        w1, w2, w3, w4, w5, w6, w7, w8 = W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7]
        # nN = nS, nH, nUH, nF, nQ
        nS, nH, nUN, nF, nQ = 1, 1, 1, MaxFO, 0
        lc_p = dict(zip(OilSpills, len(OilSpills) * [10]))
        lc_i = dict(zip(OilSpills, len(OilSpills) * [20]))
        snT = 0.8
        response_timeN = Distance_n
        ct = 20
        # slack_nH ++

        # ---------------------------------------- Set & Index ---------------------------------------------------------
        os_pair = {(o, s): custom_func.compute_distance(coordinates_spill[1][o], coordinates_st[1][s])
                   for o in OilSpills
                   for s in Stations
                   if
                   custom_func.compute_distance(tuple(coordinates_spill[1][o]),
                                                tuple(coordinates_st[1][s])) < DistanceMax}
        os_pair = tuple(os_pair.keys())

        # sr_pair (based on unique stations in pair_os )
        st_o = list(set([item[1] for item in os_pair]))
        o_st = list(set([item[0] for item in os_pair]))  # unique oil spills
        sr_pair = []
        for s in st_o:
            for r in ResourcesD:
                sr_pair.append((s, r))
        sr_pair = tuple(sr_pair)

        osr_pair = {(o, s, r): custom_func.compute_distance(coordinates_spill[1][o], coordinates_st[1][s])
                    for o in OilSpills
                    for s in Stations
                    for r in ResourcesD
                    if
                    custom_func.compute_distance(tuple(coordinates_spill[1][o]),
                                                 tuple(coordinates_st[1][s])) < DistanceMax}
        osr_pair = tuple(osr_pair.keys())
        osr_C_pair = tuple(t for t in osr_pair if t[2] == 'c')
        osr_I_pair = tuple(t for t in osr_pair if t[2] == 'i')

        print('--------------MIP-moo--------')
        model = gp.Model("MIP-moo-LAMOSCAD")
        # ---------------------------------------- Decision variable ---------------------------------------------------
        cover = model.addVars(os_pair, vtype=GRB.BINARY, name='cover')  # OilSpills
        select = model.addVars(st_o, vtype=GRB.BINARY, name='select')
        deploy = model.addVars(osr_pair, vtype=GRB.CONTINUOUS, lb=0, name='deploy')
        vehicle = model.addVars(osr_pair, vtype=GRB.BINARY, name='vehicle')

        slack_nH = model.addVars(st_o, ub=GRB.INFINITY, name='slack_nH')
        slack_nUN = model.addVars(st_o, ub=GRB.INFINITY, name='slack_nUN')
        slack_cLim = model.addVars(osr_C_pair, ub=GRB.INFINITY, name='slack_cLim')
        slack_iLim = model.addVars(osr_I_pair, ub=GRB.INFINITY, name='slack_iLim')
        # slack = dev_model.addVars(I, J, K, name="slack")

        # %% -----------------------------------------------------------------------------------------------------------
        # ------------------------------------------------ Constraints -------------------------------------------------
        # Facility related constraint : Xs (Master problem)
        # ---------------------------------------- Facility constraints (select ) --------------------------------------
        # facility must be open to cover oil spill
        C3_open_facility_to_cover = model.addConstrs((cover[o, s] <= select[s]
                                                      for o, s in os_pair), name='C_open_facility_to_cover')

        # max number of facilities to be open
        C4_max_facility = model.addConstr((gp.quicksum(select[s]
                                                       for s in st_o) <= NumberStMax), name='C_max_facility')

        # Cost of building facility does not exceed budget
        C5_budget = model.addConstr(select.prod(Cf_s) <= Budget, name="C_budget")

        # Ref Fig3a Canadian Arctic s8, s10 s11, s14
        # consider it as soft constraint to avoid infeasibitliy. so, adding slack variable is a way to build soft const
        C6_HudsonFacility = model.addConstr((gp.quicksum(select[s] + slack_nH[s]
                                                         for s in ['s8', 's10', 's11', 's14', 's17', 's19'])
                                             >= nH), name='C_HudsonFacility')

        # Up North  s9, s12, s13, s15, s16, s18, s20
        C7_UpNorthFacility = model.addConstr((gp.quicksum(select[s] - slack_nUN[s]
                                                          for s in ['s9', 's12', 's13', 's15', 's16', 's18', 's20'])
                                              <= nUN),
                                             name='C_UpNorthFacility')

        # resource capacity constaint & deploy only when facility is open & resource are available
        C8_resource_capacity = model.addConstrs((deploy.sum('*', s, r) <= BigM * Availability[s, r] * select[s]  #
                                                 for s, r in sr_pair), name='C_open_facility')

        # ---------------------------------------- Coverage constraints (cover) ----------------------------------------

        # Each oil spill should be covered by MaxFO number of stations
        C9_few_facility_per_spill = model.addConstrs((cover.sum(o, '*') <= MaxFO
                                                      for o, s in os_pair),
                                                     name='C_few_facility_per_spill')  # ++partly solved

        C10_sensitivity = model.addConstrs((cover.sum(o, '*') >= 0 for o, s in os_pair
                                            if Sensitivity_n[o] >= snT),
                                           name='C_sensitivity')
        # gp.quicksum(inv[n, k, h, t] for h in H for t in T if t - h == SHL[k] + 1)

        # ---------------------------------------- Deploy constraints (deploy) -----------------------------------------
        # deploy less than demand
        C11_deploy_demand = model.addConstrs((deploy[o, s, r] <= Demand[o, r] * cover[o, s]
                                              for o, s, r in osr_pair), name='C_deploy_demand')

        C12_minimum_quantity = model.addConstrs(deploy.sum('*', s, r) >= nQ for o, s, r in osr_pair)

        # C13: it's already implemented in the Pairing (if condition in osr_pair e.g.)
        C14_chem_limit = model.addConstrs(
            (deploy.sum('*', s, r) - slack_cLim[o, s, r] <= lc_p[o] for o, s, r in osr_C_pair),
            name='C_chem_limit')

        C15_burn_limit = model.addConstrs(
            (deploy.sum('*', s, r) - slack_iLim[o, s, r] <= lc_i[o] for o, s, r in osr_I_pair),
            name='C_burn_limit')

        # C16 is already implemented for simple case of ours in C8_resource_capacity (UFLP)
        C17_vessel_route = model.addConstrs((vehicle.sum('*', s, r) <= cover[o, s]
                                             for o, s, r in osr_pair), name='C_vessel_route')
        C17_vessel_route2 = model.addConstrs((vehicle.sum(o, '*', r) <= cover[o, s]
                                              for o, s, r in osr_pair), name='C_vessel_route2')

        # C_18_vessel_per_facility = model.addConstrs(())

        # %% -----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------- Objective function -------------------------------------------
        model.ModelSense = GRB.MINIMIZE
        objective_1 = gp.quicksum((w1 * SizeSpill_n[o] + 100 * w2 * Sensitivity_n[o] - w3 * response_timeN[o, s])
                                  * cover[o, s] for o, s in os_pair)

        objective_2 = gp.quicksum(10 ** -4 * w4 * Cf_s[s] * select[s] for s in st_o)
        objective_3 = gp.quicksum((10 ** -2 * w5 * CostU[s, r] - 10 * w6 * Effectiveness_n[s, r])
                                  * deploy[o, s, r] for o, s, r in osr_pair) \
                      + gp.quicksum((w7 * ct * Distance_n[o, s] + w8 * pn_sor[s, o, r]
                                     * vehicle[o, s, r]) for o, s, r in osr_pair)
        # - gp.quicksum(10 ** 3 * (slack_nH[s] + slack_nUN[s]) for s in st_o)

        model.setObjectiveN(objective_1, index=0, priority=3, weight=-1, name='objective_re_1')
        model.setObjectiveN(objective_2, index=1, priority=2, weight=1, name='objective_cost_2')
        model.setObjectiveN(objective_3, index=1, priority=1, weight=1, name='objective_3')

        # %% Model parameters
        # Organizing model
        # Limit how many solutions to collect
        model.setParam(GRB.Param.PoolSolutions, 1024)
        # Limit the search space by setting a gap for the worst possible solution that will be accepted
        model.setParam(GRB.Param.PoolGap, 0.80)
        # do a systematic search for the k-best solutions
        # model.setParam(GRB.Param.PoolSearchMode, 2)
        today = date.today()
        now = datetime.now()
        date_time = str(date.today().strftime("%b %d"))
        filename = 'model (' + date_time + ')'

        # Write the model
        model.write(f'../models/model_moo.lp')
        model.Params.LogFile = f"../models/model_moo({date_time}).log"  # write the log file

        # %% Solve the model
        model.optimize()
        # Debugging model
        # model.computeIIS()
        model.write('../models/model_moo.sol')

        # %% Query number of multiple objectives, and number of solutions
        x = model.getVars()
        select_series = pd.Series(model.getAttr('X', select))
        deploy_series = pd.Series(model.getAttr('X', deploy))
        # select_series[select_series > 0.5]  # +++
        # deploy_series[deploy_series > 0.5]
        nSolutions = model.SolCount
        nObjectives = model.NumObj
        print('Problem has', nObjectives, 'objectives')
        print('Gurobi found', nSolutions, 'solutions')
        solutions = []
        for s in range(nSolutions):
            # Set which solution we will query from now on
            model.params.SolutionNumber = s

            # Print objective value of this solution in each objective
            print('Solution', s, ':', end='')
            for o in range(nObjectives):
                # Set which objective we will query
                model.params.ObjNumber = o
                # Query the o-th objective value
                print(' ', round(model.ObjNVal, 2), end=' ')
                # query the full vector of the o-th solution
                solutions.append(model.getAttr('Xn', x))

        # %% Output the result
        mvars = model.getVars()
        names = model.getAttr('VarName', mvars)
        values = model.getAttr('X', mvars)

        objValues = []
        nSolutions = model.SolCount
        nObjectives = model.NumObj
        for s in range(nSolutions):
            model.params.SolutionNumber = s
            print('Solution', s, ':', end='')
            for o in range(nObjectives):
                model.params.ObjNumber = o
                objValues.append(model.ObjNVal)
        cover_series = pd.Series(model.getAttr('X', cover))

        select_series = pd.Series(model.getAttr('X', select))
        select_1s = select_series[select_series > 0.5]
        deploy_series = pd.Series(model.getAttr('X', deploy))
        deploy_1s = deploy_series[deploy_series > 0.5]
        cover_series = pd.Series(model.getAttr('X', cover))
        cover_1s = cover_series[cover_series > 0.5]

        # Saving the file
        modelStructure_output_code = python_code = logfile = model_structure = outputs = inputs = ""
        # Reading data from files
        with open('../models/model_moo.lp') as fp:
            model_structure = fp.read()
        with open('../models/model_moo.sol') as fp:
            outputs = fp.read()
        with open(f'../models/model_moo({date_time}).log') as fp:
            logfile = fp.read()
        with open('model.py') as fp:
            python_code = fp.read()
        # Merging 2 files
        # To add the data of file2
        # from next line
        modelStructure_output_code += "------------------------------- Model Structure ------------------------------\n"
        modelStructure_output_code += model_structure
        modelStructure_output_code += "\n------------------------------- Model Outputs ------------------------------\n"
        modelStructure_output_code += outputs
        modelStructure_output_code += "\n------------------------------- Model logfile ------------------------------\n"
        modelStructure_output_code += logfile
        modelStructure_output_code += "\n------------------------------- Python Code --------------------------------\n"
        modelStructure_output_code += python_code

        with open(f'../models/Structure, outputs & python code of {filename}.txt', 'w') as fp:
            fp.write(modelStructure_output_code)

        # Extract assignment variables
        sol_y = pd.Series(model.getAttr('X', deploy))
        sol_y.name = 'Assignments'
        sol_y.index.names = ['Spill #', 'Station no.', 'Resource type']
        assignment4 = sol_y[sol_y > 0.5].to_frame()
        assignment_name = assignment4.reset_index()
        # print('assignment_name', assignment_name)

        # %%
        # organize data # need to clean this section ++
        spill_df = pd.DataFrame(coordinates_spill[1]).T.reset_index()
        spill_df.columns = ['Spill #', 'Spill_Latitude', 'Spill_Longitude']
        spill_df['Resource needed'] = pd.DataFrame(SizeSpill)  # ++ update with spill size later
        spill_df['Sensitivity'] = Sensitivity_R  # ++

        station_df = pd.DataFrame(coordinates_st[1]).T.reset_index()
        station_df.columns = ['Station no.', 'St_Latitude', 'St_Longitude']

        assignment2 = pd.merge(assignment_name[['Spill #', 'Station no.']],
                               station_df[['Station no.', 'St_Latitude', 'St_Longitude']])

        assignment3 = pd.merge(assignment2, spill_df[['Spill #', 'Spill_Latitude', 'Spill_Longitude']])
        deploy_reset = deploy_1s.reset_index()
        deploy_reset.columns = ['Spill #', 'Station no.', 'Resource Type', 'Quantity deployed']
        assignment = pd.merge(assignment3, deploy_reset)

        assignment['Distance'] = [
            math.sqrt((assignment.loc[i]['St_Latitude'] - assignment.loc[i]['Spill_Latitude']) ** 2 \
                      + (assignment.loc[i]['St_Longitude'] - assignment.loc[i][
                'Spill_Longitude']) ** 2)
            for i in assignment.index]

        # Outputs from the model +++
        # Calculate Coverage # chance later ++
        coverage_percentage = int(100 * len(cover_1s) / len(OilSpills))  # / len(cover_series)
        # Calculate total distance travelled
        DistanceTravelled = []
        for i in range(len(assignment)):
            st_coord = (assignment[['St_Latitude', 'St_Longitude']]).iloc[i, :].values
            sp_coord = (assignment[['Spill_Latitude', 'Spill_Longitude']]).iloc[i, :].values
            aaa = DistanceTravelled.append(custom_func.compute_distance(st_coord, sp_coord))

        DistanceTravelled = sum(DistanceTravelled) * 80  # 80 for convering GIS data into kilometer
        ResponseTimeM = round((DistanceTravelled / 60) / len(assignment), 2)  # len() +++ OilSpills
        print(f'Coverage Percentage: {coverage_percentage}%')
        print(f'Mean Response Time: {ResponseTimeM}')

        """ 1. C_hudson
                we are adding this as soft constraint (a DV slack_nH is used to that the inequalty level violates.
                Solver will choose a positive value of slack_nH so that LHS is always >= RHS
                we dont wanna penalize obj function (which is not the case of sensitivity constraint c10)
                https://towardsdatascience.com/taking-your-optimization-skills-to-the-next-level-de47a9c51167 
                https://support.gurobi.com/hc/en-us/community/posts/5628368009233-Soft-Constraints-being-treated-as-Hard-Constraints
                
            2. C_DebtsSettledOnce = model.addConstrs(gp.quicksum(BV_DebtSettled[i_cal_month, i_tradelineinopt, i_lit]
                                                          for i_cal_month in CAL_Month for i_lit in Lits
                                                          if statement) == 1 for i_tradelineinopt in Tradelines) 
        """
        return model, select, deploy, mvars, names, values, objValues, \
            spill_df, station_df, cover_1s, select_1s, deploy_1s, ResponseTimeM, coverage_percentage, assignment
