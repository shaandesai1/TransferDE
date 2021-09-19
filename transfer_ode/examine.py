import pandas as pd

df2 = pd.read_pickle("/Users/hayden/Desktop/transfer_ode/results/func_ffnn_bundles/t2/__num_bundles_1__num_forces_1.pickle")
print(df2["scores"])