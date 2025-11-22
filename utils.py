import json
from tabulate import tabulate
from typing import Literal, Optional


class NotFittedError(ValueError):
    """
    Exception for methods called before fit().
    """

    def __init__(
        self,
        msg: str = "This instance was not fitted yet. Call the method fit() first.",
    ) -> None:
        super().__init__(msg)


class TableResult:
    """
    A class for results objects that stores data from a table.
    It can print beautiful tables and export the results to various formats.

    Attributes
    ----------
    data : list[list]
        A table.
    _headings : list[str]
        Headings for the table.
    _float_formats : tuple
        Formatting for the float strings of the table.

    Methods
    -------
    export() -> str | None
        Exports the table to a file. Returns None if no filename is provided.
    __str__() -> str
        Returns a string representation of the table.
    __repr__() -> str
        Returns a string representation of the object.
    _to_typst() -> str
        Returns a Typst representation of the table.
    _to_latex() -> str
        Returns a LaTeX representation of the table.
    _to_csv() -> str
        Returns a CSV representation of the table.
    _to_json() -> str
        Returns a JSON representation of the table.
    """

    def __init__(
        self,
        data: list[list],
        headers: list[str],
        float_formats: tuple[str, ...] | str = ".4f",
    ):
        """
        Initialization method for TableResult class.

        Parameters
        ----------
        data : list[list]
            A table.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If data is not a list.
            If any row in data is not a list.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if not all(isinstance(row, list) for row in data):
            raise TypeError("data must be a list of lists")

        self.data = data
        self._headings = headers

        if isinstance(float_formats, str):
            self._float_formats = tuple(float_formats for _ in self._headings)
        else:
            self._float_formats = float_formats

    def __str__(self) -> str:
        """
        Returns a string representation of the table.
        """
        return tabulate(
            self.data,
            headers=self._headings,
            floatfmt=self._float_formats,
            tablefmt="github",
            numalign="center",
            stralign="center",
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return f"<TableResult(n_rows={len(self.data)})>"

    def _to_typst(self) -> str:
        """
        Returns a Typst representation of the table.
        """
        # The initialization of the table
        table = (
            "#table(\n"
            f"   columns: {len(self._headings)},\n"
            "   align: center + horizon,\n"
        )

        # The header of the table
        header_cells = [f"[*{h}*]" for h in self._headings]
        table += "  " + ", ".join(header_cells) + ",\n"

        # The rows of the table
        data_row_strings = []
        for row in self.data:
            cells_in_row = []
            # Pairs each value with its correct formatting
            for value, fmt in zip(row, self._float_formats):
                formatted_val = f"{value:{fmt}}"
                cells_in_row.append(f"[{formatted_val}]")
            data_row_strings.append("   " + ", ".join(cells_in_row))
        # Joins the rows to the table
        table += ",\n".join(data_row_strings)
        table += "\n)"

        return table

    def _to_latex(self) -> str:
        """
        Returns a LaTeX representation of the table.
        """
        table = tabulate(
            self.data,
            headers=self._headings,
            floatfmt=self._float_formats,
            tablefmt="latex",
            numalign="center",
            stralign="center",
        )
        return table

    def _to_csv(self) -> str:
        """
        Returns a CSV representation of the table.
        """
        # Headings of the table
        table = ",".join(self._headings) + "\n"
        data_row_strings = []
        for row in self.data:
            cells_in_row = []
            for value in row:
                cells_in_row.append(str(value))
            # Joins cells with comma and adds to the list of row strings
            data_row_strings.append(",".join(cells_in_row))
        # Joins the rows to the table
        table += "\n".join(data_row_strings)

        return table

    def _to_json(self) -> str:
        """
        Returns a JSON representation of the table.
        """
        objects = []

        for row in self.data:
            row_dict = dict(zip(self._headings, row))
            objects.append(row_dict)

        return json.dumps(objects, indent=4)

    def export(
        self, format: Literal["typst", "latex", "csv", "json"], filename: Optional[str]
    ) -> str | None:
        """
        Exports the convergence table to a string format (and optionally to a file).

        Parameters
        ----------
        format : str
            Exporting format. Must be one of 'typst', 'latex', 'csv', or 'json'.
        filename : str, optional
            If given, saves the formatted table to a file in this path.

        Returns
        -------
        str | None
            The formatted string (if filename=None) or None (if filename is given)

        Raises
        ------
        ValueError
            If format is not one of 'typst', 'latex', 'csv', or 'json'.
        """
        msg_wrong_format = "Format must be one of 'typst', 'latex', 'csv', or 'json'."
        if format not in ["typst", "latex", "csv", "json"]:
            raise ValueError(msg_wrong_format)

        output_str = ""
        if format == "typst":
            output_str = self._to_typst()
        elif format == "latex":
            output_str = self._to_latex()
        elif format == "csv":
            output_str = self._to_csv()
        else:  # format == "json":
            output_str = self._to_json()

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_str)
            return None

        return output_str
