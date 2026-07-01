#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

EDH_SCRIPT = Path(__file__).with_name("download_era5_edh_buffer_chmpre.py")
if not EDH_SCRIPT.exists():
    EDH_SCRIPT = Path("/home/daxiniu12/lsy/unet/scripts/download_era5_edh_buffer_chmpre.py")

spec = importlib.util.spec_from_file_location("era5_edh_buffer_chmpre", EDH_SCRIPT)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import EDH helper script from {EDH_SCRIPT}")
edh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edh)

DEFAULT_ARCO_ZARRS = [
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
]

ROBUST_ALIASES = {
    "t2m": ["2m_temperature", "2 metre temperature", "2t"],
    "d2m": ["2m_dewpoint_temperature", "2 metre dewpoint temperature", "2d"],
    "skt": ["skin_temperature", "skin temperature"],
    "u10": ["10m_u_component_of_wind", "10 metre u wind component", "10u"],
    "v10": ["10m_v_component_of_wind", "10 metre v wind component", "10v"],
    "sp": ["surface_pressure", "surface pressure"],
    "msl": ["mean_sea_level_pressure", "mean sea level pressure"],
    "tcwv": ["total_column_water_vapour", "total column water vapour"],
    "tp": ["total_precipitation", "total precipitation"],
    "ssrd": ["surface_solar_radiation_downwards", "surface solar radiation downwards"],
    "cape": ["convective_available_potential_energy", "convective available potential energy"],
    "z": ["geopotential"],
    "t": ["temperature"],
    "u": ["u_component_of_wind", "u component of wind"],
    "v": ["v_component_of_wind", "v component of wind"],
    "q": ["specific_humidity", "specific humidity"],
}

# Extend the EDH helper alias table in-place so all reused functions resolve ARCO names.
for _short, _aliases in ROBUST_ALIASES.items():
    merged = list(dict.fromkeys(list(edh.ALIASES.get(_short, [])) + _aliases))
    edh.ALIASES[_short] = merged


def norm(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def require_zarr_dependencies() -> None:
    missing = []
    for name in ["zarr", "gcsfs"]:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing required packages for Google Cloud ARCO Zarr access: "
            + ", ".join(missing)
            + ". Install them in the Python environment used to run this script, "
            "for example: python -m pip install zarr gcsfs"
        )


def open_zarr(url: str) -> xr.Dataset:
    storage_options = {"token": "anon"}
    errors = []
    for consolidated in [True, None, False]:
        try:
            return xr.open_zarr(
                url,
                chunks={},
                consolidated=consolidated,
                storage_options=storage_options,
            )
        except Exception as exc:
            errors.append(f"xr.open_zarr(consolidated={consolidated!r}) -> {type(exc).__name__}: {exc}")
    try:
        return xr.open_dataset(
            url,
            engine="zarr",
            chunks={},
            backend_kwargs={"storage_options": storage_options},
        )
    except Exception as exc:
        errors.append(f"xr.open_dataset(engine='zarr') -> {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not open {url}\n" + "\n".join(errors))


def arco_var_name(ds: xr.Dataset, short_name: str) -> str:
    if short_name in ds.data_vars:
        return short_name
    aliases = {norm(short_name), *[norm(alias) for alias in edh.ALIASES.get(short_name, [])]}
    for name, da in ds.data_vars.items():
        if norm(name) in aliases:
            return name
        for attr in ["short_name", "GRIB_cfVarName", "GRIB_shortName", "long_name", "GRIB_name"]:
            if norm(da.attrs.get(attr, "")) in aliases:
                return name
    raise KeyError(f"Variable {short_name!r} not found; available={list(ds.data_vars)[:80]}")


# Reused EDH functions call edh.var_name, so replace it after defining the robust resolver.
edh.var_name = arco_var_name


def source_urls(primary: str | None, fallbacks: list[str]) -> list[str]:
    urls = []
    for url in ([primary] if primary else []) + fallbacks:
        if url and url not in urls:
            urls.append(url)
    return urls


def check_source_dataset(ds: xr.Dataset, names: list[str], levels: list[int] | None, label: str) -> None:
    lat = edh.coord_name(ds, ["latitude", "lat"])
    lon = edh.coord_name(ds, ["longitude", "lon"])
    time_name = edh.coord_name(ds, ["time", "valid_time"])
    if lat not in ds.dims or lon not in ds.dims:
        raise ValueError(
            f"{label} source is not a regular latitude/longitude grid: "
            f"{lat} dims={ds[lat].dims}, {lon} dims={ds[lon].dims}. "
            "The co/ single-level products use a reduced Gaussian values dimension and are not used by this backend."
        )
    for name in names:
        arco_var_name(ds, name)
    if levels is not None:
        level = edh.coord_name(ds, ["level", "pressure_level", "isobaricInhPa"])
        found = set(np.asarray(ds[level].values).astype(int).tolist())
        missing = [level_value for level_value in levels if level_value not in found]
        if missing:
            raise ValueError(f"{label} source missing pressure levels {missing}; found={sorted(found)}")
    time_values = ds[time_name].values.astype("datetime64[ns]")
    print(
        f"{label}: time={time_values[0]}..{time_values[-1]}, "
        f"{lat}={float(ds[lat].values[0]):g}..{float(ds[lat].values[-1]):g}, "
        f"{lon}={float(ds[lon].values[0]):g}..{float(ds[lon].values[-1]):g}",
        flush=True,
    )
    print(f"{label}: variables " + ", ".join(f"{name}->{arco_var_name(ds, name)}" for name in names), flush=True)


def ensure_time_coverage(ds: xr.Dataset, start: np.datetime64, end: np.datetime64, label: str) -> None:
    time_name = edh.coord_name(ds, ["time", "valid_time"])
    values = ds[time_name].values.astype("datetime64[ns]")
    start_ns = start.astype("datetime64[ns]")
    end_ns = end.astype("datetime64[ns]")
    if values[0] > start_ns or values[-1] < end_ns:
        raise ValueError(f"{label} time coverage {values[0]}..{values[-1]} does not cover {start_ns}..{end_ns}")


def open_first_usable_zarr(urls: list[str], names: list[str], levels: list[int] | None, label: str) -> tuple[str, xr.Dataset]:
    errors = []
    for url in urls:
        try:
            print(f"Opening {label} ARCO Zarr: {url}", flush=True)
            ds = open_zarr(url)
            check_source_dataset(ds, names, levels, label)
            return url, ds
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            print(f"  unusable {label} source {url}: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"No usable {label} ARCO Zarr source found:\n" + "\n".join(errors))


def expected_hourly_times(start: np.datetime64, end: np.datetime64) -> np.ndarray:
    return np.arange(start, end + np.timedelta64(1, "h"), np.timedelta64(1, "h")).astype("datetime64[ns]")


def expected_times_for_raw_month(
    year: int,
    month: int,
    raw_start: np.datetime64,
    raw_end: np.datetime64,
) -> np.ndarray | None:
    window = edh.monthly_window(year, month, raw_start, raw_end)
    if window is None:
        return None
    return expected_hourly_times(*window)


def validate_raw_file_strict(
    path: Path,
    expected_vars: list[str],
    pressure: bool,
    levels: list[int] | None = None,
    expected_times: np.ndarray | None = None,
) -> None:
    edh.validate_raw_file(path, expected_vars, pressure=pressure, levels=levels)
    with xr.open_dataset(path) as ds:
        if expected_times is not None:
            actual = ds["time"].values.astype("datetime64[ns]")
            if actual.shape != expected_times.shape:
                raise ValueError(f"{path} time={actual.shape[0]} expected={expected_times.shape[0]}")
            if actual.size and not np.array_equal(actual, expected_times):
                raise ValueError(
                    f"{path} has unexpected hourly times; found {actual[0]}..{actual[-1]}, "
                    f"expected {expected_times[0]}..{expected_times[-1]}"
                )
        if np.any(np.diff(ds["latitude"].values) <= 0):
            raise ValueError(f"{path} latitude must be increasing after normalize()")
        if np.any(np.diff(ds["longitude"].values) <= 0):
            raise ValueError(f"{path} longitude must be increasing")


def write_raw(
    source: xr.Dataset,
    path: Path,
    year: int,
    month: int,
    raw_start: np.datetime64,
    raw_end: np.datetime64,
    args: argparse.Namespace,
    names: list[str],
    pressure: bool,
    source_url: str,
) -> None:
    if args.raw_write_mode == "var-sharded":
        edh.write_raw_month_by_variable(
            source,
            path,
            year,
            month,
            raw_start,
            raw_end,
            args.north,
            args.south,
            args.west,
            args.east,
            names,
            args.overwrite,
            levels=args.levels if pressure else None,
            lat_tile_deg=args.raw_lat_tile_deg,
            tile_retries=args.raw_tile_retries,
            retry_sleep=args.retry_sleep,
        )
    else:
        ds = edh.build_raw_month(
            source,
            year,
            month,
            raw_start,
            raw_end,
            args.north,
            args.south,
            args.west,
            args.east,
            names,
            levels=args.levels if pressure else None,
        )
        ds.attrs["era5_source"] = "Google Cloud ARCO ERA5 Zarr"
        ds.attrs["arco_zarr_url"] = source_url
        edh.write_netcdf(ds, path, args.overwrite)


def print_accumulation_note(ds: xr.Dataset) -> None:
    for short in ["tp", "ssrd", "cape"]:
        try:
            name = arco_var_name(ds, short)
        except Exception:
            continue
        attrs = ds[name].attrs
        pieces = {key: attrs.get(key) for key in ["units", "short_name", "long_name", "GRIB_stepType", "GRIB_stepUnits"] if key in attrs}
        print(f"{short} source metadata: {json.dumps(pieces, ensure_ascii=False, sort_keys=True)}", flush=True)
    print(
        "Note: tp and ssrd are summed using the same rule as the EDH backend. "
        "ARCO regular-grid metadata exposes units but may not fully encode ERA5 accumulation-window semantics; "
        "treat tp_mm/ssrd as diagnostic until checked against the intended CDS convention.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ERA5 hourly buffer raw data from Google Cloud ARCO Zarr and derive CHM_PRE 20 BJT daily files."
    )
    parser.add_argument("--out-dir", default="/home/daxiniu12/lsy/unet/data/era5")
    parser.add_argument("--single-zarr")
    parser.add_argument("--pressure-zarr")
    parser.add_argument("--fallback-zarr", nargs="+", default=DEFAULT_ARCO_ZARRS)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--raw-start", default="1978-12-31T13:00:00")
    parser.add_argument("--raw-end", default="2022-12-31T12:00:00")
    parser.add_argument("--north", type=float, default=60.0)
    parser.add_argument("--south", type=float, default=5.0)
    parser.add_argument("--west", type=float, default=60.0)
    parser.add_argument("--east", type=float, default=150.0)
    parser.add_argument("--levels", type=int, nargs="+", default=edh.DEFAULT_LEVELS)
    parser.add_argument("--single-vars", nargs="+", default=edh.SINGLE_RAW_VARS)
    parser.add_argument("--pressure-vars", nargs="+", default=edh.PRESSURE_VARS)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["raw-single", "raw-pressure", "daily", "inventory"],
        default=["raw-single", "raw-pressure", "daily", "inventory"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-year", type=int)
    parser.add_argument("--smoke-month", type=int)
    parser.add_argument("--raw-write-mode", choices=["var-sharded", "month"], default="var-sharded")
    parser.add_argument("--raw-lat-tile-deg", type=float, default=10.0)
    parser.add_argument("--raw-tile-retries", type=int, default=5)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=int, default=60)
    parser.add_argument("--tibet-north", type=float, default=40.0)
    parser.add_argument("--tibet-south", type=float, default=25.0)
    parser.add_argument("--tibet-west", type=float, default=75.0)
    parser.add_argument("--tibet-east", type=float, default=105.0)
    args = parser.parse_args()

    require_zarr_dependencies()

    out_dir = Path(args.out_dir)
    raw_start = np.datetime64(args.raw_start)
    raw_end = np.datetime64(args.raw_end)
    years = [args.smoke_year] if args.smoke_year else list(range(args.start_year, args.end_year + 1))
    months = edh.iter_months(raw_start, raw_end)
    if args.smoke_year and args.smoke_month:
        months = [(args.smoke_year, args.smoke_month)]

    datasets: dict[str, xr.Dataset] = {}
    opened_urls: dict[str, str] = {}

    if "raw-single" in args.stages:
        url, ds = open_first_usable_zarr(source_urls(args.single_zarr, args.fallback_zarr), args.single_vars, None, "single")
        ensure_time_coverage(ds, raw_start, raw_end, "single")
        datasets["single"] = ds
        opened_urls["single"] = url
        print_accumulation_note(ds)

    if "raw-pressure" in args.stages:
        if "single" in datasets and args.pressure_zarr is None:
            try:
                check_source_dataset(datasets["single"], args.pressure_vars, args.levels, "pressure")
                datasets["pressure"] = datasets["single"]
                opened_urls["pressure"] = opened_urls["single"]
            except Exception as exc:
                print(f"single source cannot also serve pressure variables: {type(exc).__name__}: {exc}", flush=True)
        if "pressure" not in datasets:
            url, ds = open_first_usable_zarr(source_urls(args.pressure_zarr, args.fallback_zarr), args.pressure_vars, args.levels, "pressure")
            ensure_time_coverage(ds, raw_start, raw_end, "pressure")
            datasets["pressure"] = ds
            opened_urls["pressure"] = url

    if "raw-single" in args.stages:
        for year, month in months:
            path = edh.raw_single_path(out_dir, year, month)
            expected_times = expected_times_for_raw_month(year, month, raw_start, raw_end)
            if path.exists() and not args.overwrite:
                print(f"skip existing {path}", flush=True)
                validate_raw_file_strict(path, args.single_vars, pressure=False, expected_times=expected_times)
                continue

            def write_single(year=year, month=month, path=path) -> None:
                write_raw(datasets["single"], path, year, month, raw_start, raw_end, args, args.single_vars, False, opened_urls["single"])
                validate_raw_file_strict(path, args.single_vars, pressure=False, expected_times=expected_times)

            edh.run_with_retries(write_single, f"raw-single {year:04d}-{month:02d}", args.retries, args.retry_sleep)

    if "raw-pressure" in args.stages:
        for year, month in months:
            path = edh.raw_pressure_path(out_dir, year, month)
            expected_times = expected_times_for_raw_month(year, month, raw_start, raw_end)
            if path.exists() and not args.overwrite:
                print(f"skip existing {path}", flush=True)
                validate_raw_file_strict(path, args.pressure_vars, pressure=True, levels=args.levels, expected_times=expected_times)
                continue

            def write_pressure(year=year, month=month, path=path) -> None:
                write_raw(datasets["pressure"], path, year, month, raw_start, raw_end, args, args.pressure_vars, True, opened_urls["pressure"])
                validate_raw_file_strict(path, args.pressure_vars, pressure=True, levels=args.levels, expected_times=expected_times)

            edh.run_with_retries(write_pressure, f"raw-pressure {year:04d}-{month:02d}", args.retries, args.retry_sleep)

    if "daily" in args.stages:
        if args.single_vars != edh.SINGLE_RAW_VARS or args.pressure_vars != edh.PRESSURE_VARS:
            raise ValueError("--stages daily requires the full default --single-vars and --pressure-vars sets")
        for year in years:
            path = edh.daily_path(out_dir, year)
            if path.exists() and not args.overwrite:
                print(f"skip existing {path}", flush=True)
                edh.validate_daily_file(path, year, args.levels)
                continue

            def write_daily(year=year, path=path) -> None:
                ds = edh.build_daily_from_raw(out_dir, year, args.levels)
                ds.attrs["era5_source"] = "Google Cloud ARCO ERA5 Zarr"
                ds.attrs["accumulation_note"] = (
                    "tp and ssrd are summed exactly as in the EDH backend. Confirm ARCO accumulation semantics "
                    "before using these diagnostic variables as model inputs."
                )
                edh.write_netcdf(ds, path, args.overwrite)
                edh.validate_daily_file(path, year, args.levels)

            edh.run_with_retries(write_daily, f"daily {year}", args.retries, args.retry_sleep)

    if "inventory" in args.stages:
        inventory = edh.build_inventory(
            out_dir,
            years,
            args.levels,
            args.tibet_north,
            args.tibet_south,
            args.tibet_west,
            args.tibet_east,
        )
        edh.write_inventory_reports(inventory, out_dir)
        print(f"wrote {edh.inventory_json_path(out_dir)}", flush=True)
        print(f"wrote {edh.inventory_markdown_path(out_dir)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
