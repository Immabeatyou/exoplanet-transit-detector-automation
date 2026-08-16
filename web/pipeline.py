import os
from astropy.timeseries import BoxLeastSquares
import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.signal import medfilt, find_peaks, peak_widths
import argparse
from datetime import datetime, timezone
import requests
import re
from urllib.parse import urljoin
import random
import uuid
import json
import time
"""
Exoplanet Transit Detection Research Pipeline
---------------------------------------------------------------------------------------------------------
 This JupyterLab module allows for detection of exoplanet transits in Kepler light curve data using:
 1. Filtering to remove extraneous noise while preserving sharper features
 2. Numerical peak detection on inverted flux to identify dips
 3. Analysis of transit spacing to estimate orbital periods
 4. Period stability metrics to assess reliability
 5. Multi-faceted scoring rubric that judges exoplanet candidates by scientific quality
---------------------------------------------------------------------------------------------------------
 Scoring rubric:
 1. Quality flag (low/medium/high confidence based on number of peaks)
 2. Period presence and realism (0.5-50 day range)
 3. Period stability (CV <= 0.10 = stable, <= 0.25 = moderate, > 0.25 = unstable)
 4. Transit depth measurements (mean, median, max)
---------------------------------------------------------------------------------------------------------
 Results are then exported to 3 CSVs:
 1. transit_results.csv: Candidate list with all metrics previously stated
 2. top_candidates.csv: Candidates with review_now status attached to them (high priority)
 3. caution_candidates.csv: Candidates with review_with_caution status attached (inspection needed)
---------------------------------------------------------------------------------------------------------
 Data source: Kepler long-cadence (LLC) FITS files from
 https://archive.stsci.edu/pub/kepler/lightcurves/
---------------------------------------------------------------------------------------------------------
 Author: Exoplanet Research Pipeline
 Date: 2026
"""
BASE_URL = "https://archive.stsci.edu/pub/kepler/lightcurves/"
DOWNLOAD_DIR = "./kepler_llc_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Download folder:", os.path.abspath(DOWNLOAD_DIR))
def parse_args(argv=None):
    """
    Parse command-line arguments for the transit detection pipeline.
    ---------------------------------------------------------------------------------------------------------
    Args: argv (list, optional): Command-line arguments to parse. If None, uses sys.argv[1:].
    ---------------------------------------------------------------------------------------------------------
    Returns: argparse.Namespace: Parsed arguments with all flags and options.
    ---------------------------------------------------------------------------------------------------------
    Raises: ValueError: If --strict-args is set and unknown arguments are encountered.
    """
    parser = argparse.ArgumentParser(description="Run exoplanet transit pipeline")
    parser.add_argument("--output-csv", default="transit_results.csv", help="CSV output path")
    parser.add_argument("--data-dir", default="/Users/adarsh/Downloads/", help="Directory containing FITS files")
    parser.add_argument("--show-plot", action="store_true", help="Display plots during processing")
    parser.add_argument("--targets-file", default=None, help="Text file with one FITS filename per line")
    parser.add_argument("--targets", nargs="+", default=None, help="FITS filename passed directly in CLI")
    parser.add_argument("--strict-args", action="store_true", help="Fail on unknown arguments")
    parser.add_argument("--kernel-size", type=int, default=101, help="Median filter kernel size (must be odd and >= 3)")
    parser.add_argument("--prominence", type=float, default=0.0002, help="Peak prominence for dip detection (> 0)")
    parser.add_argument("--top-candidates-csv", default="top_candidates.csv", help="Output path for shortlisted top candidates CSV")
    parser.add_argument("--review-threshold", type=float, default=70.0, help="Minimum ranking score for review_with_caution")
    parser.add_argument("--review-now-threshold", type=float, default=75.0, help="Minimum ranking score for review_now")
    parser.add_argument("--run-history-csv", default="run_history.csv", help="Path for appending run history records")
    parser.add_argument("--caution-candidates-csv", default="caution_candidates.csv", help="Output path for caution candidate CSV")
    parser.add_argument("--disable-run-history", action="store_true", help="Skip writing run history log")
    parser.add_argument("--min-peaks", type=int, default=20, help="Minimum peaks for high_confidence quality flag")
    parser.add_argument("--max-peaks", type=int, default=400, help="Maximum peaks for high_confidence quality flag")
    parser.add_argument("--explain-top", type=int, default=3, help="Explanation for top ranked candidates")
    args, unknown = parser.parse_known_args(argv)
    if args.strict_args and unknown:
        raise ValueError(f"Unknown args: {unknown}")
    if unknown:
        print(f"Ignoring unknown args: {unknown}")
    return args
def run_detrending(time, flux, kernel_size=101, prominence=0.0002, show_plot=True):
    """
    Detrends a light curve and detects transit dips.
    Algorithm:
    1. Remove NaN/infinite values from time and flux arrays
    2. Apply median filter to smooth out noise
    3. Normalize flux by dividing by smoothed baseline
    4. Invert normalized flux and detect peaks (which correspond to dips)
    ---------------------------------------------------------------------------------------------------------
    Args:
    time (array-like): Time values (days)
    flux (array-like): Flux values (normalized to about 1.0)
    kernel_size (int): Median filter kernel size (must be odd, >= 3). Default: 101
    prominence (float): Peak prominence threshold for dip detection. Default: 0.0002
    show_plot (bool): Display detrended light curve plot. Default: True
    ---------------------------------------------------------------------------------------------------------
    Returns:
    dict: Contains keys:
        - time_clean: Cleaned time array
        - flux_detrended: Detrended normalized flux
        - peaks: Array of peak indices (transit locations)
        - transit_times: Time values at peaks
        - peak_properties: scipy peak detection properties
    ---------------------------------------------------------------------------------------------------------
    Side effects:
    Prints number of candidate dips and first 10 transit times.
    Displays matplotlib plot if show_plot=True.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)
    mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_clean = time_arr[mask]
    flux_clean = flux_arr[mask]
    flux64 = flux_clean.astype(np.float64)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = medfilt(flux64, kernel_size=kernel_size)
    smooth[smooth == 0] = np.nan
    flux_detrended = flux64 / smooth
    finite = np.isfinite(time_clean) & np.isfinite(flux_detrended)
    bls_time = time_clean[finite]
    bls_flux = flux_detrended[finite]

    if len(bls_time) >= 100:
        bls = BoxLeastSquares(bls_time, bls_flux)
        periods = np.linspace(0.5, 50.0, 2000)
        durations = np.linspace(0.04, 0.3, 10)

        bls_result = bls.power(periods, durations)
        best_index = int(np.argmax(bls_result.power))

        bls_period_days = float(bls_result.period[best_index])
        bls_duration_days = float(bls_result.duration[best_index])
        bls_depth = float(bls_result.depth[best_index])
        bls_power = float(bls_result.power[best_index])
    else:
        bls_period_days = None
        bls_duration_days = None
        bls_depth = None
        bls_power = None

    total_samples = len(time)
    valid_samples = int(np.sum(np.isfinite(time) & np.isfinite(flux)))
    valid_fraction = valid_samples / total_samples if total_samples > 0 else 0.0

    finite_times = time[np.isfinite(time)]
    observation_baseline_days = (
        float(np.max(finite_times) - np.min(finite_times))
        if len(finite_times) >= 2
        else None
    )

    inv_flux = 1.0 - flux_detrended
    peaks, props = find_peaks(inv_flux, prominence=prominence)
    transit_times = time_clean[peaks]
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.scatter(time_clean, flux_detrended, s=1)
        plt.xlabel("Time (days)")
        plt.ylabel("Detrended Flux")
        plt.title("Detrended Light Curve")
        plt.show()
    print("Number of candidate dips:", len(peaks))
    print("First 10 dip times:", transit_times[:10])
    return {
        "time_clean": time_clean,
        "flux_detrended": flux_detrended,
        "peaks": peaks,
        "transit_times": transit_times,
        "peak_properties": props,
    }
def validate_fits_file(file_path, required_columns=["TIME", "PDCSAP_FLUX"]):
    """
    Validate FITS file structure before processing.
    ---------------------------------------------------------------------------------------------------------
    Args:
    file_path (str): Path to FITS file
    required_columns (list): Required column names in HDU[1]. Default: ["TIME", "PDCSAP_FLUX"]
    ---------------------------------------------------------------------------------------------------------
    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if file is valid
            - error_message (str): None if valid, error description if invalid
    ---------------------------------------------------------------------------------------------------------
    Checks:
    - File exists
    - File is readable FITS
    - HDU[1] has required columns
    - TIME and PDCSAP_FLUX have same length
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        with fits.open(file_path) as hdul:
            if len(hdul) < 2:
                return False, f"FITS file has fewer than 2 HDUs: {file_path}"

            data = hdul[1].data
            if data is None:
                return False, f"HDU[1] is empty: {file_path}"

            for col in required_columns:
                if col not in data.dtype.names:
                    return False, f"Missing required column '{col}': {file_path}"

            time = data["TIME"]
            flux = data["PDCSAP_FLUX"]
            if len(time) != len(flux):
                return False, f"TIME and PDCSAP_FLUX have different lengths: {file_path}"

        return True, None
    except Exception as e:
        return False, f"FITS validation error: {str(e)}"
def run_pipeline(
    targets, 
    data_dir="/Users/adarsh/Downloads/", 
    export_csv=True, 
    output_csv="transit_results.csv", 
    show_plot=True, 
    kernel_size=101, 
    prominence=0.0002, 
    top_candidates_csv="top_candidates.csv", 
    caution_candidates_csv="caution_candidates.csv", 
    review_threshold=70.0, 
    review_now_threshold=75.0,
    min_peaks=20,
    max_peaks=400,
    downloads_df=None,
):
    """
    Main orchestration function: process targets, detect transits, compute metrics, rank candidates.
    ---------------------------------------------------------------------------------------------------------
    For each target FITS file:
    1. Load TIME and PDCSAP_FLUX columns
    2. Run detrending to detect transit dips
    3. Compute period estimate (median spacing of transits)
    4. Compute period stability (coefficient of variation)
    5. Measure transit depths (mean, median, max)
    6. Assign quality flag based on peak count
    7. Compute final ranking score (0-100)
    8. Assign review status (review_now / review_with_caution / low_priority)
    9. Append results to output DataFrames
    ---------------------------------------------------------------------------------------------------------
    Args:
    targets (list): FITS filenames to process
    data_dir (str): Directory containing FITS files
    export_csv (bool): Write CSVs to disk. Default: True
    output_csv (str): Path for main results CSV. Default: transit_results.csv
    show_plot (bool): Display plots during processing. Default: False
    kernel_size (int): Median filter kernel size. Default: 101
    prominence (float): Peak prominence threshold. Default: 0.0002
    top_candidates_csv (str): Path for review_now CSV. Default: top_candidates.csv
    caution_candidates_csv (str): Path for review_with_caution CSV. Default: caution_candidates.csv
    review_threshold (float): Minimum score for review_with_caution. Default: 70.0
    review_now_threshold (float): Minimum score for review_now. Default: 75.0
    min_peaks (int): Minimum peaks for high_confidence. Default: 20
    max_peaks (int): Maximum peaks for high_confidence. Default: 400
    --------------------------------------------------------------------------------------------------------
    Returns:
    tuple: (results_df, failures, summary)
        - results_df (DataFrame): Ranked candidates with all metrics
        - failures (list): Failed targets with error messages
        - summary (dict): Keys: processed, succeeded, failed, total_candidate_dips
    ---------------------------------------------------------------------------------------------------------
    Output columns in results_df:
        Scientific: target, num_peaks, quality_flag, estimated_period_days, period_stability_cv, period_stability_flag, mean_transit_depth, median_transit_depth, max_transit_depth
        Provenance: run_id, fetched_at_utc, source_url, local_path
        Metadata: kernel_size, prominence, data_dir, first_transit_time, mean_detrended_flux, std_detrended_flux
        Scoring: final_ranking_score, review_status, review_reason
    """
    rows = []
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.now(timezone.utc).isoformat()
    if downloads_df is None:
        downloads_df = pd.DataFrame()
    failures = []
    total_candidate_dips = 0
    
    for target in targets:
        try:
            print(f"[{len(rows) + len(failures) + 1}/{len(targets)}] processing {target}")
            
            matching_download = (
                downloads_df[downloads_df["filename"] == target]
                if not downloads_df.empty and "filename" in downloads_df.columns
                else pd.DataFrame()
            )
            file_path = (
                matching_download.iloc[0]["local_path"]
                if not matching_download.empty and pd.notna(matching_download.iloc[0].get("local_path"))
                else os.path.join(data_dir, target)
            )
            
            is_valid, error_msg = validate_fits_file(file_path)
            if not is_valid:
                failures.append({
                    "target": target,
                    "error": f"FITS validation failed: {error_msg}",
                })
                continue
                
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing FITS file: {file_path}")
                
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                time = data["TIME"]
                flux = data["PDCSAP_FLUX"]

            total_samples = len(time)
            valid_samples = int(np.sum(np.isfinite(time) & np.isfinite(flux)))
            valid_fraction = valid_samples / total_samples if total_samples else 0.0
            finite_times = time[np.isfinite(time)]
            observation_baseline_days = (
                float(np.max(finite_times) - np.min(finite_times))
                if len(finite_times) >= 2
                else None
            )
                
            result = run_detrending(
                time=time, 
                flux=flux, 
                kernel_size=kernel_size, 
                prominence=prominence, 
                show_plot=show_plot
            )
            peaks = result["peaks"]
            num_peaks = len(peaks)
            total_candidate_dips += len(peaks)
            transit_times = result["transit_times"]
            if len(transit_times) >= 3:
                spacings = np.diff(transit_times)
                spacings = spacings[np.isfinite(spacings)]
                estimated_period_days = float(np.median(spacings)) if len(spacings) > 0 else None

                if len(spacings) >= 2 and np.nanmean(spacings) > 0:
                    period_stability_cv = float(np.nanstd(spacings) / np.nanmean(spacings))
                else:
                    period_stability_cv = None
            else:
                estimated_period_days = None
                period_stability_cv = None

            if period_stability_cv is None:
                period_stability_flag = "unknown"
            elif period_stability_cv <= 0.20:
                period_stability_flag = "stable"
            elif period_stability_cv <= 0.50:
                period_stability_flag = "moderate"
            else:
                period_stability_flag = "unstable"
                
            flux_detrended = result["flux_detrended"]
            bls_period_days = None
            bls_duration_days = None
            bls_depth = None
            bls_power = None
            flux_noise = float(np.nanstd(flux_detrended - 1.0))
            transit_snr = None
            depth_consistency_cv = None

            finite = np.isfinite(result["time_clean"]) & np.isfinite(flux_detrended)
            bls_time = result["time_clean"][finite]
            bls_flux = flux_detrended[finite]
            if len(bls_time) >= 100:
                bls = BoxLeastSquares(bls_time, bls_flux)
                periods = np.linspace(0.5, 50.0, 2000)
                durations = np.linspace(0.04, 0.3, 10)
                bls_result = bls.power(periods, durations)
                best_index = int(np.argmax(bls_result.power))
                bls_period_days = float(bls_result.period[best_index])
                bls_duration_days = float(bls_result.duration[best_index])
                bls_depth = float(bls_result.depth[best_index])
                bls_power = float(bls_result.power[best_index])

            if len(peaks) > 0:
                dip_depths = 1.0 - flux_detrended[peaks]
                dip_depths = dip_depths[np.isfinite(dip_depths)]
                mean_transit_depth = float(np.mean(dip_depths)) if len(dip_depths) > 0 else None
                median_transit_depth = float(np.median(dip_depths)) if len(dip_depths) > 0 else None
                max_transit_depth = float(np.max(dip_depths)) if len(dip_depths) > 0 else None
                transit_snr = (
                    float(median_transit_depth / flux_noise * np.sqrt(len(peaks)))
                    if median_transit_depth is not None and flux_noise > 0 
                    else None
                )
                
                depth_consistency_cv = (
                    float(np.std(dip_depths) / np.mean(dip_depths))
                    if len(dip_depths) >= 2 and np.mean(dip_depths) > 0
                    else None
                )
                                
            else:
                mean_transit_depth = None
                median_transit_depth = None
                max_transit_depth = None

            peak_width_values = peak_widths(
                1.0 - flux_detrended,
                peaks,
                rel_height=0.5,
            )[0] if len(peaks) > 0 else []
            median_transit_duration = (
                float(np.median(peak_width_values))
                if len(peak_width_values) > 0
                else None
            )

            if transit_snr is None:
                snr_flag = "unknown"
            elif transit_snr >= 10:
                snr_flag = "strong"
            elif transit_snr >= 5:
                snr_flag = "moderate"
            else:
                snr_flag = "weak"

            period_agreement = None
            if estimated_period_days and bls_period_days:
                period_agreement = abs(
                    estimated_period_days - bls_period_days
                ) / estimated_period_days
            
            if num_peaks == 0 or num_peaks > max_peaks * 5:
                quality_flag = "low_confidence"
            elif min_peaks <= num_peaks <= max_peaks:
                quality_flag = "high_confidence"
            else:
                quality_flag = "medium_confidence"
                
            def compute_final_ranking_score(
                num_peaks, 
                quality_flag, 
                estimated_period_days,
                min_peaks=20,
                max_peaks=400
            ):
                score = 0.0
                if quality_flag == "high_confidence":
                    score += 55.0
                elif quality_flag == "medium_confidence":
                    score += 35.0
                else:
                    score += 10.0

                if estimated_period_days is not None:
                    score += 25.0

                    if 0.5 <= estimated_period_days <= 50:
                        score += 10.0
                else:
                    score -= 5.0

                if num_peaks == 0:
                    score -= 20.0
                elif num_peaks > 2000:
                    score -= 15.0
                elif min_peaks <= num_peaks <= max_peaks:
                    score += 10.0

                return float(max(0.0, min(100.0, score)))
                
            final_ranking_score = compute_final_ranking_score(
                num_peaks=num_peaks,
                quality_flag=quality_flag,
                estimated_period_days=estimated_period_days,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
            )

            if period_stability_flag == "stable":
                final_ranking_score = min(100.0, final_ranking_score + 8.0)
            elif period_stability_flag == "moderate":
                final_ranking_score = min(100.0, final_ranking_score + 3.0)
            elif period_stability_flag == "unstable":
                final_ranking_score = max(0.0, final_ranking_score - 20.0)

            if (
                final_ranking_score >= 90
                and period_stability_flag in ["stable", "moderate"]
                and quality_flag in ["high_confidence", "medium_confidence"]
            ):
                review_status = "review_now"
                review_reason = "very high score with acceptable stability/confidence"
            elif final_ranking_score >= 75:
                review_status = "review_with_caution"
                review_reason = "strong score but needs additional validation"
            elif final_ranking_score >= review_threshold:
                review_status = "review_with_caution"
                review_reason = "above threshold with weaker supporting signals"
            else:
                review_status = "low_priority"
                review_reason = "below review threshold or weak supporting signals"
                
            rows.append({
                "target": target,
                "first_transit_time": float(transit_times[0]) if len(transit_times) > 0 else None,
                "mean_detrended_flux": float(np.nanmean(flux_detrended)),
                "std_detrended_flux": float(np.nanstd(flux_detrended)),
                "kernel_size": kernel_size,
                "prominence": prominence,
                "data_dir": os.path.abspath(data_dir),
                "num_peaks": num_peaks,
                "quality_flag": quality_flag,
                "estimated_period_days": estimated_period_days,
                "final_ranking_score": final_ranking_score,
                "period_stability_cv": period_stability_cv,
                "period_stability_flag": period_stability_flag,
                "review_status": review_status,
                "review_reason": review_reason,
                "mean_transit_depth": mean_transit_depth,
                "median_transit_depth": median_transit_depth,
                "max_transit_depth": max_transit_depth,
                "median_transit_duration": median_transit_duration,
                "snr_flag": snr_flag,
                "period_agreement": period_agreement,
                "run_id": run_id,
                "fetched_at_utc": run_timestamp,
                "source_url": downloads_df[downloads_df["filename"] == target]["source_url"].values[0] if not downloads_df.empty and target in downloads_df["filename"].values else None,
                "local_path": downloads_df[downloads_df["filename"] == target]["local_path"].values[0] if not downloads_df.empty and target in downloads_df["filename"].values else None,
                "bls_period_days": bls_period_days,
                "bls_duration_days": bls_duration_days,
                "bls_depth": bls_depth,
                "bls_power": bls_power,
                "flux_noise": flux_noise,
                "transit_snr": transit_snr,
                "depth_consistency_cv": depth_consistency_cv,
                "valid_fraction": valid_fraction,
                "observation_baseline_days": observation_baseline_days,
            })
        except Exception as e:
            failures.append({
                "target": target,
                "error": str(e),
            })
            continue
    results_df = pd.DataFrame(rows)

    model_path = os.environ.get("XGBOOST_MODEL_PATH", "transit_xgb_model.json")
    metadata_path = os.environ.get(
        "XGBOOST_METADATA_PATH",
        "transit_xgb_metadata.json",
    )

    model_threshold = 0.75

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            model_threshold = float(
                metadata.get("best_threshold", model_threshold)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"ML metadata unavailable; using threshold {model_threshold}: {exc}")

    model_threshold = min(max(model_threshold, 0.41), 0.99)

    results_df["ml_probability"] = np.nan
    ml_model_status = "unavailable"

    if not results_df.empty and os.path.exists(model_path):
        try:
            from xgboost import XGBClassifier

            feature_columns = [
                "num_peaks",
                "estimated_period_days",
                "period_stability_cv",
                "mean_transit_depth",
                "median_transit_depth",
                "max_transit_depth",
                "bls_period_days",
                "bls_duration_days",
                "bls_depth",
                "bls_power",
                "mean_detrended_flux",
                "std_detrended_flux",
                "kernel_size",
                "prominence",
                "transit_snr",
                "depth_consistency_cv",
                "valid_fraction",
                "observation_baseline_days",
                "period_agreement",
            ]

            model = XGBClassifier()
            model.load_model(model_path)
            model_features = model.get_booster().feature_names or feature_columns
            missing_model_features = [
                column for column in model_features
                if column not in results_df.columns
            ]
            if missing_model_features:
                raise ValueError(
                    f"Model features missing from results: {missing_model_features}"
                )

            results_df["ml_probability"] = model.predict_proba(
                results_df[model_features]
            )[:, 1]
            ml_model_status = "active"
        except Exception as exc:
            print(f"ML scoring unavailable: {exc}")

    results_df["ml_model_status"] = ml_model_status
    results_df["ml_review_status"] = "model_unavailable"

    valid_ml = results_df["ml_probability"].notna()
    results_df.loc[valid_ml, "ml_review_status"] = pd.cut(
        results_df.loc[valid_ml, "ml_probability"],
        bins=[-0.01, 0.40, model_threshold, 1.0],
        labels=["low_priority", "review_with_caution", "review_now"],
    ).astype(str)

    if not results_df.empty and "final_ranking_score" in results_df.columns:
        results_df = results_df.sort_values("final_ranking_score", ascending=False).reset_index(drop=True)
    summary = {
        "processed": len(targets),
        "succeeded": len(rows),
        "failed": len(failures),
        "total_candidate_dips": int(total_candidate_dips),
    }
    if export_csv and not results_df.empty:
        results_df.to_csv(output_csv, index=False)
    if export_csv:
        if results_df.empty:
            print("No successful rows to export.")
        else:
            print(f"Exported {len(results_df)} rows to {output_csv}")
    
    review_now_df = results_df.loc[results_df['ml_review_status'] == "review_now"].copy()
    review_caution_df = results_df.loc[results_df['ml_review_status'] == "review_with_caution"].copy()
    
    if "final_ranking_score" in review_now_df.columns:
        review_now_df = review_now_df.sort_values("final_ranking_score", ascending=False).reset_index(drop=True)
    
    if "final_ranking_score" in review_caution_df.columns:
        review_caution_df = review_caution_df.sort_values("final_ranking_score", ascending=False).reset_index(drop=True)
            
    review_now_df.to_csv(top_candidates_csv, index=False)
    review_caution_df.to_csv(caution_candidates_csv, index=False)

    print(f"Exported {len(review_now_df)} review-now candidates to {top_candidates_csv}")
    print(f"Exported {len(review_caution_df)} caution candidates to {caution_candidates_csv}")
    
    return results_df, failures, summary
def explain_ranking(row, min_peaks=20, max_peaks=400):
    print(f"Ranking Explanation for {row['target']}")
    print(f"Final Score: {row['final_ranking_score']:.1f}/100")
    print()

    print(f"Quality Flag: {row['quality_flag']}")
    if row['quality_flag'] == "high_confidence":
        print(f"+55 pts (num_peaks={row['num_peaks']} in range [{min_peaks}, {max_peaks}])")
    elif row['quality_flag'] == "medium_confidence":
        print(f"+35 pts (num_peaks={row['num_peaks']} outside ideal range)")
    else:
        print(f"+10 pts (LOW: num_peaks={row['num_peaks']} extreme)")
    print()

    print(f"Period: {row['estimated_period_days']:.2f} days" if row['estimated_period_days'] else "Period: None")
    if row['estimated_period_days'] is not None:
        print(f"+25 pts (period detected)")
        if 0.5 <= row['estimated_period_days'] <= 50:
            print(f"+10 pts (period in realistic range [0.5-50] days)")
        else:
            print(f"0 pts (period outside realistic range)")
    else:
        print(f"-5 pts (no period detected)")
    print()

    print(f"Period Stability: {row['period_stability_flag']} (CV={row['period_stability_cv']:.3f})" if row['period_stability_cv'] else "Period Stability: unknown")
    if row['period_stability_flag'] == "stable":
        print(f"+8 pts (CV <= 0.10, consistent transit timing)")
    elif row['period_stability_flag'] == "moderate":
        print(f"+3 pts (CV <= 0.25, somewhat variable)")
    elif row['period_stability_flag'] == "unstable":
        print(f"-8 pts (CV > 0.25, high jitter)")
    else:
        print(f"0 pts (unknown, insufficient transits)")
    print()

    print(f"Transit Depth: mean={row['mean_transit_depth']:.4f}, median={row['median_transit_depth']:.4f}, max={row['max_transit_depth']:.4f}")
    print(f"(Raw signal strength; no direct bonus but informs quality)")
    print()
    
    print(f"Review Status: {row['review_status']}")
    print(f"Reason: {row['review_reason']}")
def append_run_history(args, summary, success_rate, history_csv="run_history.csv"):
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processed": int(summary["processed"]),
        "succeeded": int(summary["succeeded"]),
        "failed": int(summary["failed"]),
        "success_rate": float(success_rate),
        "total_candidate_dips": int(summary["total_candidate_dips"]),
        "kernel_size": int(args.kernel_size),
        "prominence": float(args.prominence),
        "output_csv": os.path.abspath(args.output_csv),
        "top_candidates_csv": os.path.abspath(args.top_candidates_csv),
    }

    history_df = pd.DataFrame([row])
    write_header = not os.path.exists(history_csv)
    history_df.to_csv(history_csv, mode="a", header=write_header, index=False)
SEEN_TARGETS_CSV = "./seen_targets.csv"
def load_seen_targets(path=SEEN_TARGETS_CSV):
    if not os.path.exists(path):
        return set()

    df = pd.read_csv(path)
    if "filename" not in df.columns:
        return set()
    return set(df["filename"].dropna().astype(str).tolist())

def save_seen_targets(filenames, path=SEEN_TARGETS_CSV):
    if not filenames:
        return

    old = load_seen_targets(path)
    merged = sorted(old.union(set(filenames)))
    pd.DataFrame({"filename": merged}).to_csv(path, index=False)

def clear_seen_targets(path=SEEN_TARGETS_CSV):
    if os.path.exists(path):
        os.remove(path)
def fetch_links(url, timeout=30, retries=3):
    """Fetch archive links with retries so one slow response does not look empty."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=(10, timeout))
            resp.raise_for_status()
            hrefs = re.findall(r'href="([^"]+)"', resp.text, flags=re.IGNORECASE)
            return [urljoin(url, h) for h in hrefs if h not in ("../", "./")]
        except requests.RequestException as exc:
            last_error = exc
            print(f"Archive request {attempt}/{retries} failed for {url}: {exc}")
            if attempt < retries:
                time.sleep(1)

    raise RuntimeError(f"Could not fetch archive index after {retries} attempts: {last_error}")

def list_bucket_dirs(base_url=BASE_URL):
    links = fetch_links(base_url)
    buckets = [u for u in links if re.search(r"/\d{4}/$", u)]
    if not buckets:
        raise RuntimeError(f"No Kepler archive buckets found at {base_url}")
    return buckets

def list_target_dirs(bucket_url):
    links = fetch_links(bucket_url)
    return [u for u in links if re.search(r"/\d{9}/$", u)]

def list_llc_files(target_dir_url):
    links = fetch_links(target_dir_url)
    return [u for u in links if u.lower().endswith("_llc.fits")]

def download_file(url, out_path, timeout=30):
    """Download file with aggressive timeout."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.Timeout:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise TimeoutError(f"Download timeout (>{timeout}s): {url}")
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise

def _discover_local_llc_files(download_dir, max_results=None):
    """Find any cached FITS files available locally when the archive is unreachable."""
    search_dirs = []
    for base_dir in [
        download_dir,
        os.path.abspath(os.path.join(os.getcwd(), "kepler_llc_downloads")),
        os.path.abspath(os.path.join(os.getcwd(), "data")),
        os.path.abspath(os.path.join(os.getcwd(), "uploads")),
    ]:
        if base_dir and os.path.isdir(base_dir) and base_dir not in search_dirs:
            search_dirs.append(base_dir)

    local_rows = []
    seen_names = set()

    for base_dir in search_dirs:
        try:
            for filename in sorted(os.listdir(base_dir)):
                lower_name = filename.lower()
                if not (lower_name.endswith(".fits") or lower_name.endswith(".fits.gz")):
                    continue
                if filename in seen_names:
                    continue
                seen_names.add(filename)

                local_path = os.path.abspath(os.path.join(base_dir, filename))
                if not os.path.isfile(local_path):
                    continue
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                local_rows.append({
                    "filename": filename,
                    "local_path": local_path,
                    "source_url": f"local://{local_path}",
                    "size_mb": round(size_mb, 2),
                })
                if max_results is not None and len(local_rows) >= max_results:
                    return local_rows
        except OSError:
            continue

    return local_rows


def fetch_kepler_llc_from_archive(
    target_count, 
    download_dir=DOWNLOAD_DIR, 
    max_buckets=500, 
    randomize=True, 
    random_seed=None,
    exclude_filenames=None,
):
    """
    Crawl Kepler archive website and download LLC FITS files.
    ---------------------------------------------------------------------------------------------------------
    Args:
    target_count (int): Number of FITS files to fetch
    download_dir (str): Local directory to save files
    max_buckets (int): Maximum archive buckets to scan
    randomize (bool): Shuffle bucket/target selection for variety
    random_seed (int, optional): Seed for reproducible selection. None = different each run.
    exclude_filenames (set, optional): Filenames to skip (for non-repeating runs)
    ---------------------------------------------------------------------------------------------------------
    Returns:
    DataFrame: Columns: filename, local_path, source_url, size_mb
    ---------------------------------------------------------------------------------------------------------
    Crawl strategy:
    1. List archive buckets (0007/, 0008/, ...)
    2. For each bucket, list KIC target directories (9-digit KICs)
    3. For each target, list LLC FITS files
    4. Download until target_count reached
    5. Track filenames in exclude_filenames to avoid repeats
    """
    if target_count <= 0:
        return pd.DataFrame(columns=["filename", "local_path", "source_url", "size_mb"])

    if exclude_filenames is None:
        exclude_filenames = set()

    os.makedirs(download_dir, exist_ok=True)
    if not randomize:
        local_rows = _discover_local_llc_files(download_dir, max_results=target_count)
        if local_rows:
            filtered_local = []
            seen = set()
            for row in local_rows:
                fname = row["filename"]
                if fname in seen or fname in exclude_filenames:
                    continue
                seen.add(fname)
                filtered_local.append(row)
                if len(filtered_local) >= target_count:
                    break
            return pd.DataFrame(filtered_local)

    rows = []
    seen = set()

    rng = random.Random(random_seed)

    try:
        bucket_dirs = list_bucket_dirs()
        step = max(1, len(bucket_dirs) // max_buckets)
        bucket_dirs = bucket_dirs[::step][:max_buckets]
        rng.shuffle(bucket_dirs)

        bucket_targets = []
        for bucket in bucket_dirs:
            try:
                target_dirs = list_target_dirs(bucket)
                rng.shuffle(target_dirs)
                bucket_targets.append(target_dirs)
            except Exception as e:
                print("Skipping bucket:", bucket, "|", e)

        # Rotate through buckets so the first pass covers as much of MAST as possible.
        for round_index in range(max((len(targets) for targets in bucket_targets), default=0)):
            for target_dirs in bucket_targets:
                if len(rows) >= target_count:
                    break
                if round_index >= len(target_dirs):
                    continue

                tdir = target_dirs[round_index]
                kic_name = os.path.basename(tdir.rstrip("/"))
                if any(existing_kic == kic_name for existing_kic in seen):
                    continue

                try:
                    llc_urls = list_llc_files(tdir)
                    rng.shuffle(llc_urls)
                except Exception as e:
                    print("Skipping target dir:", tdir, "|", e)
                    continue

                for file_url in llc_urls:
                    if len(rows) >= target_count:
                        break

                    fname = os.path.basename(file_url)
                    if fname in seen or fname in exclude_filenames:
                        continue
                    seen.add(kic_name)
                    seen.add(fname)

                    local_path = os.path.abspath(os.path.join(download_dir, fname))
                    try:
                        if not os.path.exists(local_path):
                            download_file(file_url, local_path)
                            print(f"Downloaded {len(rows)+1}/{target_count}: {fname}")
                        else:
                            print(f"Using existing {len(rows)+1}/{target_count}: {fname}")

                        size_mb = os.path.getsize(local_path) / (1024 * 1024)
                        rows.append({
                            "filename": fname,
                            "local_path": local_path,
                            "source_url": file_url,
                            "size_mb": round(size_mb, 2),
                        })
                    except Exception as e:
                        print("Failed file:", file_url, "|", e)
                    break
    except Exception as exc:
        print(f"Kepler archive unreachable; falling back to local cache: {exc}")

    if len(rows) < target_count:
        local_rows = _discover_local_llc_files(download_dir, max_results=target_count)
        if local_rows:
            print(
                f"Using {len(local_rows)} cached local FITS file(s) because the Kepler archive is unavailable."
            )
        for row in local_rows:
            fname = row["filename"]
            if fname in seen or fname in exclude_filenames:
                continue
            seen.add(fname)
            rows.append(row)
            if len(rows) >= target_count:
                break

    return pd.DataFrame(rows)
def main(argv=None):
    args = parse_args(argv)
    if args.kernel_size < 3:
        raise ValueError("--kernel-size must be >= 3")
    if args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be odd")
    if args.prominence <= 0:
        raise ValueError("--prominence must be > 0")    
    if args.review_threshold < 0 or args.review_threshold > 100:
        raise ValueError("--review-threshold must be between 0 and 100")
    if args.review_now_threshold < 0 or args.review_now_threshold > 100:
        raise ValueError("--review-now-threshold must be between 0 and 100")
    if args.review_now_threshold < args.review_threshold:
        raise ValueError("--review-now-threshold must be >= --review-threshold")
    if args.min_peaks < 1:
        raise ValueError("--min-peaks must be >= 1")
    if args.max_peaks <= args.min_peaks:
        raise ValueError("--max-peaks must be > --min-peaks")

    TARGET_COUNT = 10
    already_seen = load_seen_targets()

    downloads_df = fetch_kepler_llc_from_archive(
        TARGET_COUNT,
        max_buckets=20,
        randomize=True, 
        random_seed=None,
        exclude_filenames=already_seen,
    )
    
    if downloads_df.empty:
        print("No unseen files found. Resetting list of seen targets and retrying.")
        clear_seen_targets()
        downloads_df = fetch_kepler_llc_from_archive(
            target_count=TARGET_COUNT,
            max_buckets=20,
            randomize=True,
            random_seed=None,
            exclude_filenames=set(),
        )
        
    if not downloads_df.empty:
        save_seen_targets(downloads_df["filename"].tolist())

    min_target_warning = 5
    if len(downloads_df) < min_target_warning:
        print(f"\nOnly {len(downloads_df)} targets fetched (expected >= {min_target_warning})")
        print("This can produce unreliable results. Consider retrying or checking archive connectivity.\n")
    print("Total files ready:", len(downloads_df))
    print(downloads_df.head(20))

    targets = downloads_df["filename"].tolist()
    
    results_df, failures, summary = run_pipeline(
        targets=targets,
        data_dir=os.path.abspath(DOWNLOAD_DIR),
        export_csv=True,
        output_csv=args.output_csv,
        show_plot=args.show_plot,
        kernel_size=args.kernel_size,
        prominence=args.prominence,
        top_candidates_csv=args.top_candidates_csv,
        caution_candidates_csv=args.caution_candidates_csv,
        review_threshold=args.review_threshold,
        review_now_threshold=args.review_now_threshold,
        min_peaks=args.min_peaks,
        max_peaks=args.max_peaks,
        downloads_df=downloads_df,
    )

    success_rate = (summary["succeeded"] / summary["processed"] * 100) if summary["processed"] else 0.0
    if not args.disable_run_history:
        append_run_history(args, summary, success_rate, history_csv=args.run_history_csv)
        print(f"Appended run metadata to {args.run_history_csv}")
    else:
        print("Run history logging disabled for this execution.")

    print("\nRun summary:")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total candidate dips: {summary['total_candidate_dips']}")
    print(f"Using data directory: {args.data_dir}")
    print(f"Targets requested: {len(targets)}")
    print(
        f"Summary: processed={summary['processed']}, "
        f"succeeded={summary['succeeded']}, "
        f"failed={summary['failed']}"
    )
    
    if not results_df.empty:
        print("\nQuality flag counts:")
        print(results_df["quality_flag"].value_counts(dropna=False))

        print("\nPeriod stability counts:")
        print(results_df["period_stability_flag"].value_counts(dropna=False))

    review_threshold = args.review_threshold

    if not results_df.empty:
        review_df = results_df[results_df["final_ranking_score"] >= review_threshold].copy()
        print(f"\nCandidates meeting review threshold ({review_threshold}): {len(review_df)}")

        if not review_df.empty:
            preview_cols = [
                "target",
                "final_ranking_score",
                "estimated_period_days",
                "period_stability_flag",
                "quality_flag",
            ]
            print(review_df[preview_cols].head(5))
            
    if not results_df.empty:
        suspicious_df = results_df[
            (results_df["final_ranking_score"] >= review_threshold)
            & (
                (results_df["period_stability_flag"] == "unstable")
                | (results_df["estimated_period_days"].isna())
            )
            ].copy()
        if not suspicious_df.empty:
            print("\nWarning: high-scoring candidates with caution flags detected.")
            warning_cols = [
                "target",
                "final_ranking_score",
                "estimated_period_days",
                "period_stability_flag",
                "quality_flag",
            ]
            print(suspicious_df[warning_cols].head(5))
    
    if args.explain_top > 0 and not results_df.empty:
        print("\n- Explaining Top Candidates -")
        for i in range(min(args.explain_top, len(results_df))):
            explain_ranking(results_df.iloc[i], min_peaks=args.min_peaks, max_peaks=args.max_peaks)
            print()
            
    if summary["succeeded"] == 0:
        print("No targets processed successfully.")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f" - {failure['target']}: {failure['error']}")

    if not results_df.empty:
        print(results_df.head())

if __name__ == "__main__":
    main()