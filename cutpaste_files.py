
C_deploy_demand = model.addConstrs((gp.quicksum(deploy[o, s, r] for s in Stations)
                                    <= Demand[o, r] for o in OilSpills for r in ResourcesD
                                    if custom_func.compute_distance(tuple(coordinates_spill[1][o]), tuple(coordinates_st[1][s])) < DistanceMax),
                                   name='C_deploy_demand')


C_facility_to_each_spill = model.addConstrs((gp.quicksum(cover[o, s]  for s in Stations) == 1
                                            for o in OilSpills
                                            if custom_func.compute_distance(tuple(coordinates_spill[1][o]), tuple(coordinates_st[1][s])) < DistanceMax)
                                            , name='C_facility_to_each_spill')

m.addConstrs((gp.quicksum(build[t] for t in sites if r in coverage[t]) >= is_covered[r]
                  for r in regions), name="Build2cover")
# https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/cell_tower_coverage/cell_tower_gcl.ipynb#scrollTo=wb16M1Mlza-3

# m.addConstrs( (transport.sum('*', p) <= capacity[p] * open[p] for p in plants), "Capacity")
# m.addConstrs( (transport.sum(w) == demand[w] for w in warehouses), "Demand")

    C_DebtsSettledOnce = model.addConstrs((gp.quicksum(cover[o, s]
                                                       for s in st_o)
                                           <= MaxFO for o in o_st),
                                          name='C_few_facility_per_spill')  # think from SFS model prototype++


