""" Optimization model: maximize effectiveness and minimize cost: MIP

Outline:

1. Define Decision variables
2. Add Constraints
3. Add objective functions
4. Set some gurobi parameters, & write the model
5. Solve the model
6. Write log file
7. Get some variables out of the model for further analysis

Note:

Developer: Tanmoy Das
Date: July 05, 2023
"""

# %% Data
# data processing libraries
import pandas as pd
from datetime import datetime, date
# optimization
import gurobipy as gp
from gurobipy import GRB
# import custom functions or classes
import custom_func
import math


# %% Model 2: MIP-2
def solve(Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill, SizeSpill_n,
          Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMax, Distance, Distance_n,
          W, DistanceMax, Cf_s, Cu_sor, Budget,
          BigM, MaxF):
    """
    :param Stations:
    :param OilSpills:
    :param ResourcesD:
    :param coordinates_st:
    :param coordinates_spill:
    :param SizeSpill:
    :param SizeSpill_n:
    :param Demand:
    :param Sensitivity:
    :param Sensitivity_n:
    :param Eff:
    :param Availability:
    :param NumberStMax:
    :param Distance:
    :param Distance_n:
    :param DistanceMax:
    :param Cf_s:
    :param Cu_sor:
    :return:
    """

    import gurobipy as gp
    from gurobipy import GRB
    from datetime import datetime, date
    w1, w2, w3, w4 = W[1], W[2], W[3], W[4]
    # ---------------------------------------- Set & Index -------------------------------------------------------------
    os_pair = {(o, s): custom_func.compute_distance(coordinates_spill[1][o], coordinates_st[1][s])
               for o in OilSpills
               for s in Stations
               if
               custom_func.compute_distance(tuple(coordinates_spill[1][o]), tuple(coordinates_st[1][s])) < DistanceMax}
    os_pair = tuple(os_pair.keys())

    # sr_pair (based on unique statoins in pair_os )
    st_o = list(set([item[1] for item in os_pair]))
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
                custom_func.compute_distance(tuple(coordinates_spill[1][o]), tuple(coordinates_st[1][s])) < DistanceMax}
    osr_pair = tuple(osr_pair.keys())

    print('--------------MIP-moo--------')
    model = gp.Model("MIP-moo-LAMOSCAD")
    # ---------------------------------------- Decision variable -------------------------------------------------------
    cover = model.addVars(os_pair, vtype=GRB.BINARY, name='cover')  # OilSpills
    select = model.addVars(st_o, ResourcesD, vtype=GRB.BINARY, name='select')
    deploy = model.addVars(osr_pair, vtype=GRB.CONTINUOUS, lb=0, name='deploy')

    #model.update()
    #model.write(f'Outputs/model_interim.lp')

    #%% ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Constraints -----------------------------------------------------

    # ---------------------------------------- Coverage constraints (cover) --------------------------------------------

    # C10: facility must be open to cover oil spill
    C_open_facility_to_cover = model.addConstrs((cover[o, s] <= select[s, r]
                                                 for o, s, r in osr_pair), name='C_open_facility_to_cover')


    # C15: Each oil spill should be covered by only one station (rethink formulation later)
    C_facility_to_each_spill = model.addConstrs((cover.sum(o, '*') <= MaxF
                                                  for o, s in os_pair), name='C_facility_to_each_spill') # ++
    """
    C_DebtsSettledOnce = model.addConstrs(gp.quicksum(cover[o, s] 
                           for s in Stations 
                           == 1 for o in OilSpills) # think from SFS model prototype++ 
                           
    C_DebtsSettledOnce = model.addConstrs(gp.quicksum(BV_DebtSettled[i_cal_month, i_tradelineinopt] 
                       for i_cal_month in CAL_Month 
                       == 1 for i_tradelineinopt in Tradelines) 
    """

    # ---------------------------------------- Facility constraints (select ) ------------------------------------------
    # C14: max number of facilities to be open
    C_max_facility = model.addConstr((gp.quicksum(select[s, r]
                                                  for s, r in sr_pair) <= NumberStMax), name='C_max_facility')  # SFS style ++

    # C25: Cost of building facility does not exceed budget
    C_budget = model.addConstr(select.prod(Cf_s) <= Budget, name="C_budget") # m.addConstr(build.prod(cost) <= budget, name="budget")

    # ---------------------------------------- Deploy constraints (deploy) ---------------------------------------------
    # C10: resource capacity constaint & deploy only when facility is open
    C_resource_capacity = model.addConstrs((deploy[o, s, r] <= BigM * Availability[s, r] * select[s, r]
                                        for o, s, r in osr_pair), name='C_open_facility')
    # C16: deploy less than demand
    C_deploy_demand = model.addConstrs((deploy[o, s, r] <= Demand[o, r]
                                        for o, s, r in osr_pair), name='C_deploy_demand')

    #%% ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Objective function -----------------------------------------------
    model.ModelSense = GRB.MINIMIZE
    objective_1_re = gp.quicksum((w1 * SizeSpill_n[o] + w2 * Sensitivity_n[o]) * cover[o, s] for o, s in os_pair) \
                    - gp.quicksum(w3 * Distance_n[o, s]  * cover[o, s] for o, s in os_pair) \
                    + gp.quicksum(w4 * Effectiveness_n[s, r] * deploy[o, s, r] for o, s, r in osr_pair)
    objective_2_cost = gp.quicksum(select[s, r] * Cf_s[s] for s, r in sr_pair)
    # w1 w2 ++
    # \                    + slack_sensi

    model.setObjectiveN(objective_1_re, index=0, priority=2, weight=-1, name='objective_re_1')
    model.setObjectiveN(objective_2_cost, index=1, priority=1, weight=1, name='objective_cost_2')

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
    date_time = str(date.today().strftime("%b %d,") + datetime.now().strftime("%H%M"))
    filename = 'model (' + date_time + ')'

    # Write the model
    model.write(f'Outputs/model_moo.lp')
    model.Params.LogFile = f"Outputs/Logfiles/model_moo({date_time}).log"  # write the log file

    # %% Solve the model
    model.optimize()
    # Debugging model
    # model.computeIIS()
    model.write('Outputs/model_moo.sol')

    # %% Query number of multiple objectives, and number of solutions
    x = model.getVars()
    select_series = pd.Series(model.getAttr('X', select))
    deploy_series = pd.Series(model.getAttr('X', deploy))
    select_series[select_series > 0.5]
    deploy_series[deploy_series > 0.5]
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
        print()

    for j in range(len(x)):
        if x[j].Xn > 0:
            print(x[j].VarName, x[j].Xn, end=' ')
            print(' ')

    # %% Output the result
    # Obtain model results & carry them outside the model scope
    model.printAttr('X')
    mvars = model.getVars()  # these values are NOT accessible outside the model scope
    names = model.getAttr('VarName', mvars)
    values = model.getAttr('X', mvars)

    # def extract_ones_DV(model, cover, select, amount, spill_data):

    cover_series = pd.Series(model.getAttr('X', cover))
    cover_1s = cover_series[cover_series > 0.5]

    select_series = pd.Series(model.getAttr('X', select))
    select_1s = select_series[select_series > 0.5]
    print('\nselect_1s\n', select_1s)
    deploy_series = pd.Series(model.getAttr('X', deploy))
    deploy_1s = deploy_series[deploy_series > 0.5]
    print('\ndeploy_1s\n', deploy_1s)
    cover_series = pd.Series(model.getAttr('X', cover))
    cover_1s = cover_series[cover_series > 0.5]
    print('\ncover_1s\n', cover_1s)

    # Saving the file
    modelStructure_output_code = python_code = logfile = model_structure = outputs = inputs = ""
    # Reading data from files
    with open('Outputs/model_moo.lp') as fp:
        model_structure = fp.read()
    with open('Outputs/model_moo.sol') as fp:
        outputs = fp.read()
    with open(f'Outputs/Logfiles/model_moo({date_time}).log') as fp:
        logfile = fp.read()
    with open('model.py') as fp:
        python_code = fp.read()
    # Merging 2 files
    # To add the data of file2
    # from next line
    modelStructure_output_code += "------------------------------- Model Structure ----------------------------------\n"
    modelStructure_output_code += model_structure
    modelStructure_output_code += "\n------------------------------- Model Outputs ----------------------------------\n"
    modelStructure_output_code += outputs
    modelStructure_output_code += "\n------------------------------- Model logfile ----------------------------------\n"
    modelStructure_output_code += logfile
    modelStructure_output_code += "\n------------------------------- Python Code ------------------------------------\n"
    modelStructure_output_code += python_code

    with open(f'Outputs/Structure, outputs & python code of {filename}.txt', 'w') as fp:
        fp.write(modelStructure_output_code)

    # Extract assignment variables
    sol_y = pd.Series(model.getAttr('X', deploy))
    sol_y.name = 'Assignments'
    sol_y.index.names = ['Spill #', 'Station no.', 'Resource type']
    assignment4 = sol_y[sol_y > 0.5].to_frame()
    assignment_name = assignment4.reset_index()
    print('assignment_name', assignment_name)

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

    assignment['Distance'] = [math.sqrt((assignment.loc[i]['St_Latitude']-assignment.loc[i]['Spill_Latitude'])**2 \
                                        + (assignment.loc[i]['St_Longitude']-assignment.loc[i]['Spill_Longitude'])**2)
                              for i in assignment.index]

    # Outputs from the model +++
    # Calculate Coverage # chance later ++
    coverage_percentage = int(100 * len(cover_1s) ) #/ len(cover_series)
    # Calculate total distance travelled
    DistanceTravelled = []
    for i in range(len(assignment)):
        st_coord = (assignment[['St_Latitude', 'St_Longitude']]).iloc[i, :].values
        sp_coord = (assignment[['Spill_Latitude', 'Spill_Longitude']]).iloc[i, :].values
        aaa = DistanceTravelled.append(custom_func.compute_distance(st_coord, sp_coord))

    DistanceTravelled = sum(DistanceTravelled)
    ResponseTimeM = int((DistanceTravelled / 60)/len(OilSpills))
    print(f'Coverage Percentage: {coverage_percentage}%')
    print(f'Mean Response Time: {ResponseTimeM}')

    return model, select, deploy, mvars, names, values, \
        spill_df, station_df, cover_1s, select_1s, deploy_1s, ResponseTimeM, coverage_percentage, assignment
