from coptpy import * 

# Create COPT environment 
env = Envr()

# create model
model = env.createModel("LP example") 

# create variables 
x1 = model.addVar(lb=0, ub=COPT.INFINITY, vtype = COPT.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=COPT.INFINITY, vtype = COPT.CONTINUOUS, name="x2")

# create objective
model.setObjective(8*x1 + 5*x2, sense=COPT.MAXIMIZE)

# create constraints
model.addConstr(x1 + x2 <= 6)
model.addConstr(9*x1 + 5*x2 <= 45)

# solve and output the optimal solution
model.solve()

print("Objective value: {}".format(model.objval))
print("Variable solution:")

for var in model.getVars():
    print(" x[{0}]: {1}".format(var.index, var.x))
