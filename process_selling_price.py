import pandas as pd
from collections import *

#
dat_date = pd.read_csv("dataset/calendar.csv")

dat_sell = pd.read_csv("dataset/sell_prices.csv")


def process_date(dat):
    # wm_yr_wk: week id
    temp_dat = dat.loc[:, ['d', 'wm_yr_wk', 'event_type_1', 'event_type_2']]
    date_id = defaultdict(list)
    id_list = dat.loc[:, 'wm_yr_wk'].to_numpy()
    for i in range(temp_dat.shape[0]):
        date_id[id_list[i]].append(i)
    # date_id converts a wm_yr_wk id to its corresponding week
    return temp_dat, date_id


def get_selling_price(dat):
    n = dat.shape[0]
    d_col = ['d_' + str(i) for i in range(1, 1969 + 1)]

    store_ = dat.iloc[:, 0].to_numpy()
    item_ = dat.iloc[:, 1].to_numpy()

    item_ls = [item_[i] + '_' + store_[i] for i in range(n)]

    items = []
    unique = set()
    items_pos = dict()
    for i in range(n):
        if item_ls[i] not in unique:
            unique.add(item_ls[i])
            items_pos[item_ls[i]] = len(items)
            items.append(item_ls[i])

    output_matrix = [[0] * (len(d_col)+1) for _ in range(len(items))]

    date_df, date_id_dict = process_date(dat_date)
    i = 0
    while i < n:
        print(i)
        store, item = dat.iloc[i, 0:2]
        item_id = str(item) + "_" + str(store)
        j = i
        while j < n and item == dat.iloc[j, 1]:
            j += 1
        wk = dat.iloc[i:j, 2].to_numpy()
        price = dat.iloc[i:j, 3].to_numpy()

        for k in range(len(wk)):
            r = date_id_dict[wk[k]]
            r1, r2 = r[0], r[-1]
            output_matrix[items_pos[item_id]][r1:(r2+1)] = [price[k]] * (r2 - r1 + 1)

        output_matrix[items_pos[item_id]][-1] = date_id_dict[wk[0]][0]
        i = j

    price_dat = pd.DataFrame(output_matrix, index=items, columns=d_col + ['start'] )
    return price_dat


get_selling_price(dat_sell).to_csv("selling_price_seq.csv", index=True)










