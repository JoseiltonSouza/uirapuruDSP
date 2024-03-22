import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz
import tzlocal
import unidecode
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()
console = Console()

logger = logging.getLogger(__name__)


def get_home() -> Path:
    """
    Get the home directory of the project by finding the root directory containing the 'src' folder.

    Returns:
        Path: The path of the home directory.
    """
    paths: List[str] = sys.path
    home: Path = Path(
        next(
            folder.split("src")[0]
            for folder in [path for path in paths if "src" in path]
        )
    )
    return home


def get_nbname(globals_dict: Dict[str, str]) -> str:
    """
    Get the name of the Jupyter notebook.

    Args:
        globals_dict (Dict[str, str]): A dictionary containing global variables.

    Returns:
        str: The name of the Jupyter notebook.
    """
    try:
        # Attempt to get the notebook name from the '__session__' key
        nb_name = Path(globals_dict["__session__"]).stem
    except KeyError:
        # If '__session__' key is not found, get the notebook name from the '__vsc_ipynb_file__' key
        nb_name = Path(globals_dict["__vsc_ipynb_file__"]).stem
    return nb_name


def get_user() -> str:
    """
    Get the name of the current user.

    Returns:
        str: The name of the current user.
    """
    user = os.getlogin()
    return user


def template_header(
    ctime: Optional[str] = None,
    titulo: Optional[str] = None,
    nb_name: Optional[str] = None,
) -> None:
    """
    Generates and prints a header template with file information.

    Args:
        ctime (Optional[str]): The creation time of the file. Defaults to None.
        titulo (Optional[str]): The title of the file. Defaults to None.
        nb_name (Optional[str]): The name of the Jupyter notebook. Defaults to None.

    Returns:
        None
    """
    # Preparing table with file information for filename management
    user = get_user()
    parent_folder = get_home()
    folder = parent_folder / "notebooks/exploratory/"
    if nb_name and Path(folder / nb_name).exists():
        file = Path(folder / nb_name)
        tz = str(tzlocal.get_localzone())
        fuso = pytz.timezone(tz)
        mtime = (
            pytz.UTC.localize(
                pd.to_datetime(file.stat().st_mtime, origin="unix", unit="s")
            )
            .astimezone(fuso)
            .strftime("%d/%m/%Y")
        )
    else:
        mtime = ctime if ctime else ""
    projeto = get_home().stem
    console = Console()
    md = Markdown(f"# {projeto} - {titulo}")
    console.print(md)
    table = Table(title="File Info")
    table.add_column("Autor", justify="left", style="bold cyan", no_wrap=True)
    table.add_column("Criado em", justify="center", style="bold cyan")
    table.add_column("Modificado em", justify="center", style="bold cyan")
    table.add_row(user, ctime, mtime)
    console.print(table)
    return


def find_formatted_files(nbs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Find files with formatted filenames in the exploratory notebooks folder.

    Args:
        nbs (Optional[List[str]]): List of filenames. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing information about formatted filenames.
    """
    if not nbs:
        parent_folder = get_home()
        nb_folder = Path(parent_folder) / "notebooks/exploratory"
        nbs = [val.stem for val in Path(nb_folder).glob("*.ipynb")]
    pattern = r"\b\d{2}_[a-z0-9]+_[\w-]+_\d{2}_\d{2}_\d{4}\b"
    formatted_files = []
    for filename in nbs:
        if re.match(pattern, filename):
            formatted_files.append(filename)
    df = pd.DataFrame()
    if len(formatted_files) > 0:
        df = pd.DataFrame(formatted_files, columns=["Filenames"])
        df[["F_Number", "User", "Title", "DD", "MM", "YYYY"]] = df[
            "Filenames"
        ].str.split("_", expand=True)
        df["F_Number"] = df["F_Number"].astype(int)
        df["Date"] = df[["YYYY", "MM", "DD"]].apply(
            lambda val: pd.to_datetime("/".join(list(val))), axis=1
        )
        df["Filenames"] = df["Filenames"] + ".ipynb"
    return df


def get_header(file: Path) -> list:
    """
    Extract creation time and title from the header of a Jupyter notebook file.

    Args:
        file (str): The path to the Jupyter notebook file.

    Returns:
        list: A list containing the creation time and title extracted from the header.
    """
    ctime = ""
    titulo = ""
    nb_name = ""
    try:
        with open(file) as data:
            contents = json.load(data)
    except json.JSONDecodeError:
        return [ctime, titulo]
    try:
        info = contents["cells"][0][
            "source"
        ]  # Extract content of the first cell
        for line in info:
            pattern_ct = r"(ctime=\"\d{2}_\d{2}_\d{4}\")"
            pattern_t = r"(titulo=\".*\",)"
            pattern_n = r"(nb_name=\".*\")\)"
            match = re.findall(pattern_ct, line)
            if match and ctime == "":
                ctime = normalize(match[0]).split("=")[-1]
            match = re.findall(pattern_t, line)
            if match and titulo == "":
                titulo = normalize(match[0]).split("=")[-1]
            match = re.findall(pattern_n, line)
            if match and nb_name == "":
                nb_name = match[0].split("=")[-1].replace('"', "")
    except KeyError:
        # Return empty creation time and title if there's an exception loading notebook contents or extracting header info
        pass
    if [ctime, titulo] == ["", ""]:
        titulo = search_header(file)
        ctime = pd.to_datetime(file.stat().st_ctime, unit="s").strftime(
            "%d_%m_$Y"
        )

    return [ctime, titulo]  # Return creation time and title


def normalize(string: str) -> str:
    """
    Normalize a string by converting accented characters to ASCII, converting to uppercase,
    replacing spaces and underscores with hyphens, and removing non-alphanumeric characters
    except for hyphens.

    Args:
        string (str): The string to be normalized.

    Returns:
        str: The normalized string.
    """
    string1 = (
        unidecode.unidecode(string).upper().replace(" ", "-").replace("_", "-")
    )
    clean_string = filter(
        lambda x: x.isalnum() or x == "-", string1
    )  # Filter out non-alphanumeric characters except hyphens
    result = "".join(
        clean_string
    )  # Join the filtered characters back into a string
    return result  # Return the normalized string


def search_header(file: Path) -> str:
    """
    Search for a header in a Jupyter notebook file and return the normalized title.

    Args:
        file (str): The path to the Jupyter notebook file.

    Returns:
        str: The normalized title extracted from the header, or "NAO-DEFINIDO" if not found.
    """
    title = normalize("NAO-DEFINIDO")
    try:
        with open(file) as data:
            contents = json.load(data)
    except json.JSONDecodeError:
        return title
    marks = [
        val["source"]
        for val in contents["cells"]
        if val["cell_type"] == "markdown"
    ]  # Extract content of markdown cells
    marks = [val for sublist in marks for val in sublist]
    title_str = (
        next(
            val if val else None
            for val in next(
                val if val else []
                for val in [
                    re.findall(r"^#[^#].*$", val) if len(val) > 0 else None
                    for val in marks
                ]
            )
        )
        .replace("#", "")
        .strip()
    )
    if title_str:
        title = normalize(title_str)  # Normalize the title

    return title


def subs_header(info: list, filename: Path, ctime: str, titulo: str) -> list:
    """
    Substitute the header line in the provided information with the given arguments.

    Args:
        info (list): List of strings representing the contents of a file.
        args (list): List containing the arguments to be substituted in the header line.

    Returns:
        list: List of strings with the header line substituted.
    """
    nb_name = filename.name
    key = "template_header("  # Key string to identify header
    re_string = f'template_header(ctime="{ctime}", titulo="{titulo}", nb_name="{nb_name}")'  # Construct replacement string
    line = [
        re_string if key in val else val for val in info
    ]  # Substitute header line
    return line  # Return the modified list of strings


def set_nb_name(
    ctime: Optional[str] = None,
    titulo: Optional[str] = None,
    user: Optional[str] = None,
) -> tuple[Path, str, str]:
    """
    Set the name for a new notebook file based on specified parameters or user input.

    Args:
        ctime (str, optional): Creation time of the notebook. Defaults to None.
        titulo (str, optional): Title of the notebook. Defaults to None.
        user (str, optional): User name. Defaults to None.

    Returns:
        tuple: A tuple containing the filename and a list of parameters used to create the filename.
    """
    if not titulo:
        console.print("[bold blue]Digite o Título do novo Notebook")
        titulo = str(input())  # Prompt user for input if title is not provided

    ctime = datetime.now().strftime(
        "%d_%m_%Y"
    )  # Set creation time to current date if not provided

    df = (
        find_formatted_files()
    )  # Find formatted filenames in exploratory notebooks folder

    F_Number_str = f"{df.F_Number.max() + 1:02d}" if not df.empty else "01"
    user = get_user()  # Get the current user if not provided

    if len(titulo) <= 1:
        titulo = "Não Definido"  # Set title to "Não Definido" if length is less than or equal to 1

    parent_folder = get_home()
    folder = parent_folder / "notebooks/exploratory/"

    # Construct the filename
    name = (
        F_Number_str
        + "_"
        + user
        + "_"
        + normalize(titulo)
        + "_"
        + ctime
        + ".ipynb"
    )
    filename = folder / name

    return filename, ctime, titulo


def inject_header(orig: Path, nb_name: Path, ctime: str, titulo: str) -> None:
    """
    Injects header into the provided file.

    Args:
        orig (Union[os.PathLike | str], optional): Path of the original file. Defaults to "".
        args (Optional[List[str]], optional): List of arguments. Defaults to None.

    Returns:
        None
    """
    try:
        with open(orig) as data:
            contents = json.load(data)
    except json.JSONDecodeError:
        return
    info = contents["cells"][0]["source"]
    info_subs = subs_header(info, nb_name, ctime, titulo)
    contents["cells"][0]["source"] = info_subs
    with open(nb_name, "w+") as outfile:
        json.dump(contents, outfile)
        console.print(f"[yellow] saving {nb_name}")
    return


def create_nb(template: str = "standard.ipynb") -> None:
    """
    Create a new notebook file based on a specified template or default template.

    Args:
        template (str, optional): The filename of the template notebook. Defaults to "standard.ipynb".

    Returns:
        None
    """
    console.print(
        "[bold blue]Gerando notebook a partir do template"
    )  # Print message indicating notebook creation process
    (
        filename,
        ctime,
        titulo,
    ) = set_nb_name()  # Generate filename and parameters for the new notebook
    parent_folder = get_home()  # Get the home folder
    template_path = Path(parent_folder) / "notebooks/templates" / template
    inject_header(template_path, filename, ctime, titulo)
    return


def process_files(nbs: List[Path]) -> None:
    """
    Process the Jupyter notebook files.

    Args:
        nbs (list): List of Jupyter notebook filenames.

    Returns:
        None
    """
    parent_folder = get_home()
    nb_folder = parent_folder / "notebooks/exploratory"
    static_folder = parent_folder / "notebooks/static/"
    nbs_formatted = find_formatted_files()
    headers = []

    if not nbs_formatted.empty:
        for file in nbs_formatted.Filenames:
            header = get_header(nb_folder / file)
            headers.append(header)

        # Create DataFrame from headers
        df = pd.DataFrame(headers)
        df.columns = pd.Index(["ctime", "titulo"])
        try:
            df["ctime"] = pd.to_datetime(
                df.ctime.str.replace('"', ""), dayfirst=True, format="%d/%m/%Y"
            )
        except ValueError:
            df["ctime"] = pd.NA
        df["titulo"] = df.titulo.apply(lambda val: normalize(val))
        df = pd.concat([nbs_formatted, df], axis=1)

        # Check for format errors
        format_error = df[df.ctime.isnull()].Filenames
        if format_error.size > 0:
            for file in format_error:
                console.print(
                    f"[red bold] HEADER corrompido. Movendo arquivo {file}  para [yellow] static"
                )
                filename = file[:-6] + "_TO_FIX.ipynb"
                filename = static_folder / filename
                shutil.move(file, filename)

        # Check for header errors
        header_error = (
            df.dropna()
            .apply(
                lambda val: val.Filenames
                if (val.Date != val.ctime) or (val.Title != val.titulo)
                else None,
                axis=1,
            )
            .dropna()
        )
        if header_error.size > 0:
            for file in header_error:
                console.print(
                    f"[red bold] Incompatibilidade de dados no HEADER. Movendo arquivo {file}  para [yellow] static"
                )
                filename = file[:-6] + "_TO_FIX_HEADER.ipynb"
                console.print(
                    f"[red bold] Atualizando Cabeçalho. Arquivo {file}  em [yellow] expĺoratory"
                )
                filename = static_folder / filename
                shutil.move(file, filename)
                ctime, titulo = (
                    header_error[header_error.Filenames == file][
                        ["ctime", "titulo"]
                    ]
                    .to_numpy()
                    .tolist()[0]
                )
                inject_header(filename, ctime, titulo, file)

        nbs_unformatted = [
            val
            for val in nbs
            if val.name not in nbs_formatted.Filenames.to_list()
        ]

    else:
        nbs_unformatted = nbs

    # Process each unformatted file
    for file in nbs_unformatted:
        [title, ctime] = get_header(file)

        user = get_user()
        cell_code = (
            """from scripts.nb_utils import nb_utils\n"""
            + f"""nb_utils.template_header(ctime="{ctime}", titulo="{title},
            nb_name="{file}")"""
        )
        header_dict = {
            "cell_type": "code",
            "execution_count": "null",
            "metadata": {},
            "outputs": [],
            "source": [cell_code],
        }
        # Generate filename and save processed file
        filename, _, _ = set_nb_name(ctime=ctime, titulo=title, user=user)
        # Move original file to static
        console.print(
            f"[red bold] Copiando arquivo {file} fora do template para [yellow] static"
        )
        new_name = f"{Path(file).stem}_[{Path(filename).name}]"
        dst = static_folder / new_name
        shutil.move(file, dst)
        try:
            with open(Path(dst)) as data:
                contents = json.load(data)
        except json.JSONDecodeError:
            return
        contents["cells"] = [header_dict] + contents["cells"]
        console.print(f"[yellow bold] Salvando arquivo {new_name}")
        with open(new_name, "w+") as outfile:
            json.dump(contents, outfile)
    return None


def get_nbs() -> List[Path]:
    """
    Retrieve the list of Jupyter notebook files from the 'exploratory' folder.

    Returns:
        list: List of pathlib.Path objects representing the notebook files.
    """
    parent_folder = get_home()
    nb_folder = parent_folder / "notebooks/exploratory"

    # Retrieve the list of notebook files in the 'exploratory' folder
    nbs = list(Path(nb_folder).glob("*.ipynb"))

    return nbs


def format_nbs() -> None:
    nbs = get_nbs()
    process_files(nbs)
    return


def main() -> None:
    create_nb()
    return None


if __name__ == "__main__":
    main()
