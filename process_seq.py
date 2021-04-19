import pandas as pd
import random

dat_valid = pd.read_csv("dataset/sales_train_evaluation.csv")
dat_eval = pd.read_csv("dataset/sales_train_validation.csv")
dat_price = pd.read_csv("selling_price_seq.csv")

seq_valid = dat_valid.iloc[:, 6:]
seq_eval = dat_eval.iloc[:, 6:]

# print(dat_valid.iloc[0:3, 0:10])
# print(dat_eval.iloc[:3, :10])

# note: rows{dat_valid} == rows{dat_eval}
def create_data_set(dat, start=[], x_period=[], sample_size=1000, resample=False):
    if len(x_period) == 0:
        print("Please choose the periods that you want to use for X")
        print("Example: [[-30, -20],[-5, -1]]")
        print("Interpretation: the range used for X are: [T-30, T-20] and [T-5, T-1],")
        print("where T is the starting time point for Y")
        return None, None
    if len(start) == 0:
        print("Default: use d1 - d1913 as possible X ranges")

    n, m = dat.shape

    if 28 - x_period[0][0] + max(start) > m:
        print("Invalid x_period: distant history")
        return None, None

    if not resample: sample_size = n

    seq_len = 0
    len_ls = []
    for s, e in x_period:
        seq_len += e - s + 1
        len_ls.append(e - s + 1)
    x, y = [[0] * seq_len for _ in range(sample_size)], [[0]*28 for _ in range(sample_size)]

    for i in range(sample_size):
        print(i)
        k = i if not resample else random.randrange(sample_size)
        start_k = start[k] # position, zero-based
        start_y = random.randrange(start_k - x_period[0][0], m - 27)
        temp = []
        for j in range(len(len_ls)):
            temp.extend(dat.iloc[i, start_y+x_period[j][0]:(start_y+x_period[j][1]+1)].tolist())
        x[i] = temp
        y[i] = dat.iloc[i, start_y : (start_y + 28)].tolist()

    output_x = pd.DataFrame(x)
    output_y = pd.DataFrame(y)

    return output_x, output_y


start_ls = dat_price.iloc[:, -1].tolist()
range_ls = [[-56, -1]]
x, y = create_data_set(dat=dat_valid, start=start_ls, x_period=range_ls)
if x is not None:
    x.to_csv("valid_X.csv", index=False)
    y.to_csv("valid_y.csv", index=False)
    pd.concat([x, y], axis=1).to_csv("valid_X_y.csv", index=False)
else:
    print("Error: x is None")











