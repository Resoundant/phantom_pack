from copy import deepcopy
from pathlib import Path
import pandas as pd

def per_vial_pdff(data: pd.DataFrame | pd.Series | list[float]) -> pd.DataFrame | list[bool]:
    targets = [40, 30, 20, 10, 0]
    acceptance = [8, 7, 6, 5, 4]
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != len(targets):
            raise ValueError
        target_series = pd.Series(targets, index=data.columns)
        acceptance_series = pd.Series(acceptance, index=data.columns)
        return data.sub(target_series, axis="columns").abs().le(acceptance_series, axis="columns")

    if len(targets) != len(data):
        raise ValueError

    return [abs(data[i] - targets[i]) <= acceptance[i] for i in range(len(targets))]

# todo: modify this to work on a dataframe, where currently it operates on one row of the frame
def linear_fit(data:list[float]):
    mydata = deepcopy(data)
    if (mydata[0] > mydata[-1]):
        mydata.reverse()
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    ys = [float(y) for y in mydata]
    n = len(xs)

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return slope, intercept, r_squared


def main(xls_filepath:str):
    df = pd.read_excel(xls_filepath)
    data_columns = df.columns[2:]
    if len(data_columns) % 5 != 0:
        raise ValueError("Expected data columns after the first two ID columns to be divisible by 5.")
    grouped_columns = [
        df.iloc[:, start:start + 5].copy()
        for start in range(2, df.shape[1], 5)
    ]
    base_columns = grouped_columns[0].columns
    grouped_df = pd.concat(
        [group.set_axis(base_columns, axis=1, copy=False) for group in grouped_columns],
        ignore_index=True,
    ).dropna(how="all")
    output_path = Path(xls_filepath).with_name(f"{Path(xls_filepath).stem}_grouped.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        grouped_df.to_excel(writer, sheet_name="grouped", index=False)


if __name__ == '__main__':
    main(r'C:\testdata\PhantomPack\LIG_Phantom_Analysis_20260406.xlsx')
