# type: ignore
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits


def read_fits():
    files = list(Path("../../data/raw/").glob("*.fit"))
    headers = np.empty([len(files), 9], dtype="object")
    for ii, file in enumerate(files):
        with fits.open(file) as hdul:
            cols = [
                "DATE-OBS",
                "TIME-OBS",
                "NAXIS2",
                "MINFREQ",
                "MAXFREQ",
                "NAXIS1",
            ]
            size = Path(file).stat().st_size / (1024 * 1024)
            data = [file, size]
            for col in cols:
                info = hdul[0].header[col]
                data.append(info)
            data.append(hdul[1].data["TIME"][0][-1])
            headers[ii, :] = np.asarray(data)
    df = pd.DataFrame(headers, columns=["File", "Size", *cols, "DELTA"])
    df["DATE_START"] = pd.to_datetime(
        df["DATE-OBS"] + "T" + df["TIME-OBS"], format="%Y%m%dT%H%M%S.%f"
    )
    df["Size"] = df["Size"].astype(float)
    df["DELTA"] = df["DELTA"].astype(float)
    df["NAXIS1"] = df["NAXIS1"].astype(int)
    df["NAXIS2"] = df["NAXIS2"].astype(int)
    df["MINFREQ"] = df["MINFREQ"].astype(float)
    df["MAXFREQ"] = df["MAXFREQ"].astype(float)
    df[["MINFREQ", "MAXFREQ"]] = df[["MINFREQ", "MAXFREQ"]] / 1e6
    df = (
        df.drop(columns=["DATE-OBS", "TIME-OBS"])
        .sort_values(["MINFREQ", "NAXIS1", "DATE_START"])
        .reset_index(drop=True)
    )
    return df


def read_obs(df):
    if not isinstance(df, pd.DataFrame):
        df = read_fits()
    dfg = df.groupby(["MINFREQ", "MAXFREQ", "NAXIS1"])[
        ["DELTA", "DATE_START"]
    ].apply(
        lambda val: np.abs(
            val.iloc[:, 1].diff().dt.total_seconds()[1:] - val.iloc[1:, 0]
        )
        < 300
    )
    series = dfg.sum()
    groups = series.groupby((series != series.shift()).cumsum())
    contiguous_ones = [group for _, group in groups if group.iloc[0] == 1]
    for ii, group in enumerate(contiguous_ones):
        idx = group.index
        df.loc[idx, "Grupo"] = int(ii)
    df_sum = (
        df.dropna()
        .groupby("Grupo")[
            ["DATE_START", "DELTA", "Size", "MINFREQ", "MAXFREQ", "NAXIS1"]
        ]
        .agg(
            {
                "DATE_START": "first",
                "DELTA": "sum",
                "Size": "sum",
                "MINFREQ": "first",
                "MAXFREQ": "first",
                "NAXIS1": "first",
            }
        )
    )
    df_sum["Size"] = df_sum["Size"] / 1024
    df_sum = df_sum.round({"Size": 2})
    df_sum["DELTA"] = df_sum.DELTA.apply(
        lambda val: pd.Timedelta(val, unit="s").round("s")
    )
    df_sum = df_sum.reset_index().dropna()
    df_sum["Grupo"] = df_sum["Grupo"].astype(int)
    return df_sum


def select_obs(group=None):
    df = read_fits()
    df_obs = read_obs(df)
    print(df_obs)
    if group is None:
        group = int(input("Digite o grupo de observações desejado"))
    idxs = df[df.Grupo == group].index
    return df.iloc[idxs, :]


def chunk_files(obs, threshold=200):
    chunks = []
    file_chunk = []
    size = 0.0
    idxs = obs.index.to_list()
    for idx in idxs:
        if size < threshold:
            file_chunk.append(obs.File.loc[idx])
            size = size + obs.Size.loc[idx]
        else:
            chunks.append(file_chunk)
            file_chunk = []
            size = 0
    return chunks


def load_fits(obs, chunks_idx=None, threshold=200):
    if chunks_idx is None:
        chunks_idx = [0, 10]
    chunks = chunk_files(obs, threshold=threshold)
    if chunks_idx[1] > len(chunks):
        chunks[1] = len(chunks)
    data = []
    timestamps = []
    for chunk in chunks[chunks_idx[0] : chunks_idx[1]]:
        for file in chunk:
            with fits.open(file) as hdul:
                times = hdul[1].data[0][0]
                freqs = da.from_array(
                    hdul[1].data[0][1].byteswap().newbyteorder()
                )
                time_vector = da.from_array(
                    pd.to_datetime(
                        hdul[0].header["DATE-OBS"]
                        + "T"
                        + hdul[0].header["TIME-OBS"]
                    )
                    + pd.to_timedelta(times, unit="s")
                )
                datum = da.from_array(hdul[0].data)
            timestamps.append(time_vector)
            data.append(datum)
        dda = da.concatenate(data)
        time_index = da.concatenate(timestamps)
    Xda = xr.DataArray(
        dda, dims=("Time", "Freq"), coords={"Time": time_index, "Freq": freqs}
    )
    return Xda
