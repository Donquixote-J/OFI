import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1.Read and Pre-treatment
df = pd.read_csv("first_25000_rows.csv", parse_dates=["ts_event"])
df = df.sort_values("ts_event").set_index("ts_event")


# 2.compute Δq
M = 10 
for m in range(M):
    bid_px      = df[f"bid_px_{m:02d}"]
    bid_sz      = df[f"bid_sz_{m:02d}"]
    bid_px_prev = bid_px.shift(1)
    bid_sz_prev = bid_sz.shift(1)

    dq_b = np.where(
        bid_px >  bid_px_prev,     
        + bid_sz,                  
        np.where(
            bid_px == bid_px_prev, 
            bid_sz - bid_sz_prev,  
            - bid_sz_prev          
        )
    )
    df[f"dq_bid{m}"] = pd.Series(dq_b, index=df.index).fillna(0).astype(float)

    ask_px      = df[f"ask_px_{m:02d}"]
    ask_sz      = df[f"ask_sz_{m:02d}"]
    ask_px_prev = ask_px.shift(1)
    ask_sz_prev = ask_sz.shift(1)

    dq_a = np.where(
        ask_px >  ask_px_prev,
        - ask_sz,
        np.where(
            ask_px == ask_px_prev,
            ask_sz - ask_sz_prev,
            + ask_sz
        )
    )
    df[f"dq_ask{m}"] = pd.Series(dq_a, index=df.index).fillna(0).astype(float)


# 3.Sampling by minute (construct 10 layers of OFI)
ofi_list = []
for m in range(M):
    # 1T = 1 minute
    ofi_m = (
        df[f"dq_bid{m}"].resample("1T").sum()
      - df[f"dq_ask{m}"].resample("1T").sum()
    ).rename(f"OFI{m+1}")
    ofi_list.append(ofi_m)
ofi_ml = pd.concat(ofi_list, axis=1) # Column splicing


# 4.Best-Level OFI
ofi1 = ofi_ml["OFI1"]


# 5.Integrated OFI via PCA 
X = ofi_ml.dropna()
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X.values)[:,0]
w = np.abs(pca.components_[0]); w /= w.sum()
ofi_I = pd.Series((X.values*w).sum(axis=1), index=X.index, name="OFI_I")


# 6.Cross-Asset OFI
cross_ofi = ofi_ml.sum(axis=1).rename("CrossOFI")


# 7.Overall
features = pd.concat([ofi1, ofi_ml, ofi_I, cross_ofi], axis=1).dropna()
print(features.head())
features.to_csv("ofi_features_output.csv")