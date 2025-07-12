from gurobipy import Model, GRB, quicksum

# Data
I = [0, 1, 2]  # Facilities
J = [0, 1, 2, 3]  # Customers
f = {0: 100, 1: 120, 2: 140}  # Facility opening costs
c = {(i, j): abs(i - j) * 10 + 5 for i in I for j in J}  # Assignment cost
d = {j: 1 for j in J}  # Demand of customers
s = {0: 2, 1: 3, 2: 2}  # Facility capacities

# Initialize Master Problem
master = Model("FLP_Master")
x = master.addVars(I, vtype=GRB.BINARY, name="x")  # Facility open decision
theta = master.addVar(vtype=GRB.CONTINUOUS, name="theta")  # Subproblem cost

# Master Objective
master.setObjective(quicksum(f[i] * x[i] for i in I) + theta, GRB.MINIMIZE)

# Store Benders Cuts
benders_cuts = []

iteration = 0
max_iterations = 10  # Prevent infinite loops

while iteration < max_iterations:
    iteration += 1
    print(f"\nIteration {iteration}")
    master.params.OutputFlag = 0
    master.optimize()

    # Get the facility opening decision
    x_vals = {i: x[i].x for i in I}

    # If all x_i are 0, we need at least one facility
    if sum(x_vals.values()) == 0:
        print("Forcing at least one facility to open")
        master.addConstr(quicksum(x[i] for i in I) >= 1)
        continue

    # Solve Subproblem (Customer Assignments)
    sub = Model("FLP_Sub")
    y = sub.addVars(I, J, vtype=GRB.BINARY, name="y")

    # Each customer is assigned to exactly one facility
    sub.addConstrs((quicksum(y[i, j] for i in I) == 1 for j in J), name="assign")

    # Customers can only be assigned to open facilities
    sub.addConstrs((y[i, j] <= x_vals[i] for i in I for j in J), name="open_facility")

    # Facility capacity constraints
    sub.addConstrs((quicksum(d[j] * y[i, j] for j in J) <= s[i] * x_vals[i] for i in I), name="capacity")

    # Objective: Minimize assignment costs
    sub.setObjective(quicksum(c[i, j] * y[i, j] for i in I for j in J), GRB.MINIMIZE)
    sub.params.OutputFlag = 0
    sub.optimize()

    # If the subproblem is infeasible, add feasibility cuts
    if sub.status == GRB.INFEASIBLE:
        print("Adding feasibility cut to master problem")
        master.addConstr(quicksum(x[i] for i in I) >= 1)  # Ensure at least one facility is open
        continue  # Re-solve Master Problem

    # Get optimal subproblem cost
    theta_val = sub.objVal

    # Check convergence: if theta value is stable, stop
    if abs(theta.x - theta_val) < 1e-6:
        print("Converged: Theta stable")
        break

    # Add Benders optimality cut
    cut_expr = quicksum(c[i, j] * y[i, j].x for i in I for j in J)
    print(f"Adding Benders cut: θ ≥ {cut_expr}")
    master.addConstr(theta >= cut_expr)

# Final Results
print("\nFinal Solution:")
print("Facilities Opened:", {i: x[i].x for i in I})
print("Final Cost:", master.objVal)
