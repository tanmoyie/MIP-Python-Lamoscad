#%%
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
import custom_func
from datetime import datetime, date
import input_data
import grblogtools as glt
#%%
# Input parameters
Stations, OilSpills, ResourcesD, Demand, Availability, Eff, NumberStMax, \
    Distance, DistanceMax, TimeR, TimeRMax, Cf_s, Cu_sor = input_data.generate_input_data()

# Model
model22 = gp.Model("facility")

# Plant open decision variables: open[p] == 1 if s is open.
open_v = model22.addVars(Stations, ResourcesD, vtype=GRB.BINARY, name="open_v")

slack_1 = model22.addVars(OilSpills, ResourcesD, name="slack")  # , ub=
# Transportation decision variables: transport[w,p] captures the
# optimal quantity to transport to warehouse w from plant p
transport = model22.addVars(Stations, OilSpills, ResourcesD, name="transport_v")  # deploy er vai

# The objective is to minimize the total fixed and variable costs
model22.ModelSense = GRB.MINIMIZE

"""
# Production constraints
# Note that the right-hand limit sets the production to zero if the plant is closed
C_open_facility = model22.addConstrs((transport[(s, o, r)] <= Availability[(s, r)] * open_v[s, r]
                                      # need to fix it to correct deploy = 1
                                      for o in OilSpills for r in ResourcesD for s in Stations),
                                     name='open_facility')
                                     """
C_capacity = model22.addConstrs((transport.sum('*', s, r) <= Availability[s, r] * open_v[s, r]
                                 for s in Stations for r in ResourcesD),
                                name="Capacity")

C_max_facility = model22.addConstr((gp.quicksum(open_v[s, r]
                                               for s in Stations for r in ResourcesD) == NumberStMax),
                                  name='max_facility')
"""
# Budget constraint
C_budget = model22.addConstr((gp.quicksum(open_v[s] * Cf_s[s] for s in Stations) +
                              gp.quicksum(transport[s, o, r] * Cu_sor[s, o, r]
                                          for s in Stations for o in OilSpills for r in ResourcesD)
                              <= 10000),
                             name='total_budget')
"""
# Demand constraints
C_demand = model22.addConstrs((transport.sum(s, o, r) == Demand[o, r]  # - slack_1[o, r]
                               for s in Stations for r in ResourcesD for o in OilSpills),
                              name="Demand")  # cover all demands

objective_cost_2 = gp.quicksum(open_v[s, r] * Cf_s[s] for s in Stations for r in ResourcesD) + \
                   gp.quicksum(transport[s, o, r] * Cu_sor[s, o, r]
                               for s in Stations for o in OilSpills for r in ResourcesD)
model22.setObjective(objective_cost_2)
# Save model
model22.write('Outputs/facility1_cost.lp')
model22.optimize()
model22.write('Outputs/model22.sol')
