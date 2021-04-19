import pandas as pd

dat_valid = pd.read_csv("dataset/sales_train_evaluation.csv")
dat_eval = pd.read_csv("dataset/sales_train_validation.csv")
dat_price = pd.read_csv("selling_price_seq.csv")

seq_valid = dat_valid.iloc[:, 6:]
seq_eval = dat_eval.iloc[:, 6:]

# print(dat_valid.iloc[0:3, 0:10])
# print(dat_eval.iloc[:3, :10])

CONST_LEN = 28


def get_predictor(dat, start):
    n, m = dat.shape
    x = []
    for i in range(n):
        print(i)
        # cnt = (m - start[i]) // CONST_LEN
        x.append(dat.iloc[i, (-4 * CONST_LEN):].tolist())

    output_x = pd.DataFrame(x)
    return output_x


# each row is a seq of CONST_LEN * 4 observations
# we will use X = 1-4 CONST_LEN observations to predict Y = 2 - 5 CONST_LEN observations
# note: rows{dat_valid} == rows{dat_eval}
def create_data_set(dat, start):

    n, m = dat.shape
    x = []
    for i in range(n):
        # print(i)
        cnt = (m - start[i]) // CONST_LEN
        for k in range(cnt-4): # 4=5-1
            x.append(dat.iloc[i, (-(k+5)*CONST_LEN):(-k*CONST_LEN)].tolist())

    output_x = pd.DataFrame(x)
    return output_x


start_ls = dat_price.iloc[:, -1].tolist()
x = create_data_set(dat=dat_valid, start=start_ls)
if x is not None:
    x.to_csv("valid_X.csv", index=False)
else:
    print("Error: x is None.")

x_valid = get_predictor(dat=dat_valid, start=start_ls)
x_eval = get_predictor(dat=dat_eval, start=start_ls)
if x_valid is not None:
    x_valid.to_csv("valid_X_pred.csv", index=False)
else:
    print("Error: x_valid is None.")

if x_eval is not None:
    x_eval.to_csv("valid_X_eval.csv", index=False)
else:
    print("Error: x_eval is None.")










