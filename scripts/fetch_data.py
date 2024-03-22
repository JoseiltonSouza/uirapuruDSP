import os
import re
import sys
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
from rich.console import Console

console = Console()


def get_home() -> Path:
    """
    Retrieves the home directory path.

    Returns:
        Path: The home directory path.
    """
    paths: List[str] = sys.path
    home: Path = Path(
        next(
            folder.split("src")[0]
            for folder in [path for path in paths if "src" in path]
        )
    )
    return home


def filesize_MB(val: str) -> Union[float, None]:
    """
    Converts file size to megabytes.

    Args:
        val (str): The file size value to be converted.

    Returns:
        Union[float, None]: The converted file size in megabytes or None if conversion fails.
    """
    kb_units: dict = {"K": 1 / 1024, "M": 1, "G": 1024}
    result: Union[float, None] = None
    match = re.match(r"(\d+(\.\d+)?)([a-zA-Z]+)", val)
    if match:
        result = np.round(
            float(match.group(1)) * kb_units[match.group(3).capitalize()], 2
        )
    return result


def file_description(val: str) -> List[Union[str, None]]:
    """
    Extracts information from a filename.

    Args:
        val (str): The filename to parse.

    Returns:
        List[Union[str, None]]: A list containing file prefix, timestamp, mode, and type.
    """
    result: List[Union[str, None]] = [None, None, None, None]
    pattern: str = r"(.*)(\d{8}[_]\d{6})[_](\d+)\.([a-zA-Z]+)"
    match = re.match(pattern, val)
    if match:
        result = list(match.groups())
    return result


def parse_filenames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses filenames in a DataFrame and adds additional columns.

    Args:
        df (pd.DataFrame): The DataFrame containing filenames.

    Returns:
        pd.DataFrame: The DataFrame with parsed filename components added.
    """
    df["Size"] = df.Size.apply(filesize_MB)
    df[["Prefix", "Timestamp", "Mode", "Type"]] = df.apply(
        lambda val: file_description(val.Filename), axis=1, result_type="expand"
    )
    df["Timestamp"] = pd.to_datetime(df.Timestamp, format="%Y%m%d_%H%M%S")
    return df


def fetch_data_index(
    url: str = "http://150.165.37.33/data/UIRAPURU",
) -> pd.DataFrame:
    """
    Fetches index data from a specified URL.

    Args:
        url (str, optional): The URL to fetch data from. Defaults to "http://150.165.37.33/data/UIRAPURU".

    Returns:
        pd.DataFrame: The fetched index data.
    """
    wwwuser: str = os.getenv("WWWUSER")  # type: ignore  # noqa: PGH003
    wwwpassword: str = os.getenv("WWWPWD")  # type: ignore  # noqa: PGH003
    response = requests.get(url, auth=(wwwuser, wwwpassword), timeout=10)
    index: str = response.text
    colnames: List[str] = ["Filename", "Size"]
    df = (
        pd.read_html(StringIO(index), na_values="&nbsp", header=[0])[0]
        .iloc[:, [1, 3]]
        .set_axis(colnames, axis="columns")
        .dropna()
    )
    df = parse_filenames(df)
    return df


def fetch_file(
    file: str, URL_prefix: Optional[str] = None, dry_run: bool = False
) -> None:
    """
    Fetches a file from a URL and saves it.

    Args:
        file (str): The filename to fetch.
        URL_prefix (str, optional): The URL prefix. Defaults to None.
        dry_run (bool, optional): If True, performs a dry run without saving the file. Defaults to False.
    """
    if not URL_prefix:
        URL_prefix = "http://150.165.37.33/data/UIRAPURU"
    URL: str = f"{URL_prefix}/{file}"
    wwwuser: str = os.getenv("WWWUSER", "user")
    wwwpassword: str = os.getenv("WWWPWD", "password")
    home: Path = get_home()
    filename: Path = Path(home) / "data/raw" / file
    console.print(f"[yellow] Obtendo arquivo {filename}.")
    if not dry_run:
        response = requests.get(
            URL, auth=(wwwuser, wwwpassword), stream=True, timeout=10
        )
        with open(filename, "wb") as fd:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                fd.write(chunk)
                console.print(f"[yellow] Arquivo {filename} salvo.")


def get_file_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary DataFrame of file counts and total sizes per day.

    Args:
        df (pd.DataFrame): The DataFrame containing file information.

    Returns:
        pd.DataFrame: The summary DataFrame with columns: 'Mes-Dia', '#_Files', and 'Size'.
    """
    col_names: List[str] = ["Mes-Dia", "#_Files", "Size"]
    df["Mes-Dia"] = df.Timestamp.dt.strftime("%m/%d")
    df_summary = (
        df.groupby("Mes-Dia")[["Filename", "Size"]]
        .agg({"Filename": "count", "Size": "sum"})
        .reset_index()
    )
    df_summary.columns = pd.Index(col_names)
    return df_summary


def fetch_data(files: Optional[List[str]] = None) -> None:
    """
    Fetches data files based on user input or downloads all files if no input provided.

    Args:
        files (List[str], optional): List of filenames to fetch. Defaults to [].
    """
    if files is None:
        files = []
    if not files:
        console.print("[bold green]Baixando índice dos arquivos.")
        all_files = fetch_data_index()
        all_files["Mes-Dia"] = all_files.Timestamp.dt.strftime("%m/%d")
        df = get_file_summary(all_files)
        console.print(
            "[bold blue]Digite o índice correspondente ao dia para qual deseja efetuar o download."
            "[green][enter] [bold blue] para baixar todos os arquivos"
        )
        console.print(f"[orange]{df}")
        idx = input()
        if not idx:
            files = all_files.Filenames.to_list()
        else:
            indice = int(idx)
            if indice in df.index:
                Date = df[df.index == indice]["Mes-Dia"].values[0]
                files = all_files[all_files["Mes-Dia"] == Date].Filename
        console.print("[bold green]Baixando arquivos.")
    if files is not None:
        for file in files:
            fetch_file(file)
    return


def main() -> None:
    fetch_data()
    return


if __name__ == "__main__":
    main()
