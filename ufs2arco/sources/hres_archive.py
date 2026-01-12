import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source, NOAAGribForecastData

logger = logging.getLogger("ufs2arco")

class HRESArchive(NOAAGribForecastData, Source):
    """
    Access 0.25 degree archives of ECMWF via:
        * After 06 March 2024 (13pl): AWS at https://registry.opendata.aws/ecmwf-forecasts/
    """

    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    file_suffixes = ("", "b")
    static_vars = ("lsm", "orog")

    @property
    def available_levels(self) -> tuple:
        return (
            50, 100, 150, 200, 250, 300, 400,
            500, 600, 700, 850, 925, 1000,
        )

    @property
    def rename(self) -> dict:
        return {
            "time": "t0",
            "step": "lead_time",
            "isobaricInhPa": "level",
        }

    def __init__(
        self,
        t0: dict,
        fhr: dict,
        resolution: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
        accum_hrs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            resolution (str): spatial resolution, 0p25: 0.25 degree; 1p00: 1.0 degree
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
            accum_hrs (dict, optional): period in hours over which to accumulate accumulated variables, e.g.
                {"accum_tp": 1}
                would be the same as passing filter_by_keys={"stepRange": "5-6"}
                when reading fhr 6 data using xarray and cfgrib
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        self.resolution = resolution
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
            accum_hrs=accum_hrs,
        )


        # make sure the default variable set "just works"
        # by pulling out the variable names only available at forecast time
        if tuple(self.variables) == self.available_variables:
            varlist = []
            for key, meta in self._varmeta.items():
                if not self._varmeta[key]["forecast_only"]:
                    varlist.append(key)
            self.variables = varlist

        # for GFS, plenty of variables only exist in the forecast, not analysis grib files
        # make sure the user doesn't ask for these before we get started
        if any(self.fhr == 0):
            requested_vars_not_in_analysis = []
            for varname in self.variables:
                if self._varmeta[varname]["forecast_only"]:
                    requested_vars_not_in_analysis.append(varname)
            if len(requested_vars_not_in_analysis) > 0:
                msg = f"{self.name}.__init__: the following requested variables only exist in forecast timesteps"
                msg += f"\n\n{requested_vars_not_in_analysis}\n\n"
                msg += "these should be requested separately with only fhr > 0 (i.e., fhr start >0 in your yaml)"
                raise Exception(msg)

    def _build_path(
        self,
        t0: pd.Timestamp,
        fhr: int,
        file_suffix: str,
    ) -> str:
        """
        Build the file path to a GFS GRIB file on RDA or AWS.

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            fhr (int): The forecast hour.
            file_suffix (str): For GFS, either '' or 'b'.

        Returns:
            str: The constructed file path.
        """
        res = self.resolution
        bucket = "s3://ecmwf-forecasts"
        outer = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}/{t0.hour:02d}z/ifs/{res}/oper"
        fname = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}{t0.hour:02d}0000-{fhr}h-oper-fc.grib2"


        fullpath = f"filecache::{bucket}/{outer}/{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath
