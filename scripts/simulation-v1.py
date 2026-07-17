from scipy.special import logsumexp
from scipy.stats import lognorm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd
from scipy.integrate import solve_ivp

# ============================================================
# PHYSICS HELPERS
# ============================================================

# Boltzmann constant in eV/K
kB = 8.617333262145e-5


def diffusion_coeff(D0, Em, T_K):
    """
    Arrhenius diffusion coefficient.

    D = D0 * exp(-Em / kB*T)
    """
    return D0 * np.exp(-Em / (kB * T_K))


def coalescence_rate(P0, Ea, T_K):
    """
    Thermally activated coalescence coefficient.

    Pcs = P0 * exp(-Ea / kB*T)
    """
    return P0 * np.exp(-Ea / (kB * T_K))


def compute_Rx(Nx, Cx, b, Omega0, eps=1e-300):
    """
    Paper-consistent loop radius.

    The factor 1/3 is already included in the Frank-loop
    Burgers vector magnitude b = |a/3 <111>|.

    N = (pi*b/Omega0) * R^2 * C

    Therefore:
    R = sqrt(Omega0*N / (pi*b*C))
    """

    Nx_eff = max(float(Nx), eps)
    Cx_eff = max(float(Cx), eps)

    return np.sqrt(Omega0 * Nx_eff / (np.pi * b * Cx_eff))


def logterminv(R, r0):
    """
    Safe version of 1 / ln(8R/r0).
    """

    R_min = 1.01 * r0 / 8.0
    R_eff = max(float(R), R_min)

    return 1.0 / np.log(8.0 * R_eff / r0)


def j_x_L(Rx, Di, Ci, r0):
    """
    Interstitial flux to a toroidal dislocation loop.
    """

    return (
        2.0
        * np.pi**2
        * Rx
        * Di
        * max(float(Ci), 0.0)
        * logterminv(Rx, r0)
    )

# ============================================================
# LOAD LOOP-SIZE DATA FROM CSV FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change this if your CSV files are stored somewhere else.
DATA_DIR = PROJECT_ROOT / "data"


DATASET_SPECS = [
    # filename, temperature_C, mode, irradiated
    ("RT-BF-irr.csv",    25.0,   "BF", True),
    ("RT-BF.csv",        25.0,   "BF", False),
    ("RT-DF-irr.csv",    25.0,   "DF", True),
    ("RT-DF.csv",        25.0,   "DF", False),

    ("900-BF-irr.csv",   900.0,  "BF", True),
    ("900-BF.csv",       900.0,  "BF", False),
    ("900-DF-irr.csv",   900.0,  "DF", True),
    ("900-DF.csv",       900.0,  "DF", False),

    ("1100-BF-irr.csv",  1100.0, "BF", True),
    ("1100-DF-irr.csv",  1100.0, "DF", True),
]


def load_one_dataset(filename, temperature_C, mode, irradiated):
    """
    Read one CSV file containing one or more image columns.

    Each numerical value is interpreted as one measured loop diameter in nm.
    """

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{file_path}\n\n"
            "Check DATA_DIR and confirm the CSV files are in that folder."
        )

    raw = pd.read_csv(file_path)

    # Convert wide image-column table into one long table:
    # Image 0921 | Image 1529  ->  image | size
    long_data = raw.melt(
        var_name="image",
        value_name="size",
    )

    long_data["size"] = pd.to_numeric(
        long_data["size"],
        errors="coerce",
    )

    long_data = long_data.dropna(subset=["size"]).copy()
    long_data = long_data[long_data["size"] > 0].copy()

    long_data["temperature_C"] = float(temperature_C)
    long_data["mode"] = mode
    long_data["irradiated"] = bool(irradiated)
    long_data["source_file"] = filename

    return long_data[
        [
            "size",
            "temperature_C",
            "mode",
            "irradiated",
            "image",
            "source_file",
        ]
    ]


def load_all_loop_data():
    datasets = []

    for filename, temperature_C, mode, irradiated in DATASET_SPECS:
        df = load_one_dataset(
            filename=filename,
            temperature_C=temperature_C,
            mode=mode,
            irradiated=irradiated,
        )
        datasets.append(df)

    loop_data = pd.concat(datasets, ignore_index=True)

    return loop_data


loop_data = load_all_loop_data()

print("\nLoaded loop-size datasets:")
print(
    loop_data
    .groupby(["temperature_C", "mode", "irradiated"])
    .size()
    .rename("n_loops")
)

print("\nTotal measured loops:", len(loop_data))

# ============================================================
# RANDOM PHYSICAL INITIAL CONDITION
# ============================================================

def loguniform(low, high, rng):
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def make_random_y0(b, seed=None):
    """
    Random but physically reasonable initial state.

    State:
        y0 = [Ci0, Cv0, Nf0, Np0, Cf0, Cp0]

    Units:
        R in cm
        loop densities in cm^-3
    """

    rng = np.random.default_rng(seed)

    # Mobile point defects
    Ci0 = loguniform(1e-18, 1e-6, rng)
    Cv0 = 0.0

    # Loop densities
    Cf0 = loguniform(1e14, 1e18, rng)
    Cp0 = loguniform(1e12, 1e18, rng)

    # Initial radii in nm
    Rf0_nm = loguniform(0.5, 10.0, rng)
    Rp0_nm = loguniform(0.5, 20.0, rng)

    # Convert nm -> cm
    Rf0 = Rf0_nm * 1e-7
    Rp0 = Rp0_nm * 1e-7

    # Your convention:
    # R = sqrt(N / (pi*b*C))
    # therefore N = pi*b*R^2*C
    Nf0 = np.pi * b * Rf0**2 * Cf0
    Np0 = np.pi * b * Rp0**2 * Cp0

    y0 = np.array([Ci0, Cv0, Nf0, Np0, Cf0, Cp0], dtype=float)

    print("Random physical y0:")
    print(f"  Ci0 = {Ci0:.3e}")
    print(f"  Cv0 = {Cv0:.3e}")
    print(f"  Rf0 = {Rf0_nm:.3f} nm")
    print(f"  Rp0 = {Rp0_nm:.3f} nm")
    print(f"  Cf0 = {Cf0:.3e} cm^-3")
    print(f"  Cp0 = {Cp0:.3e} cm^-3")
    print(f"  Nf0 = {Nf0:.3e}")
    print(f"  Np0 = {Np0:.3e}")
    print(f"  len(y0) = {len(y0)}")

    return y0


# ============================================================
# DISTRIBUTION MODEL
# ============================================================

def lognormal_shape_from_mean_std(mean, std):
    """
    Convert physical mean and physical standard deviation
    into scipy's lognormal shape parameter.
    """

    mean = max(float(mean), 1e-30)
    std = max(float(std), 1e-30)

    return np.sqrt(np.log(1.0 + (std / mean)**2))


def lognormal_logpdf_from_mean_and_k(x, mean, k):
    """
    Lognormal log-PDF where the physical standard deviation is:

        std = k * mean

    Here k controls the width of the distribution.
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    mean = max(float(mean), 1e-30)
    k = max(float(k), 1e-8)

    std = k * mean
    sigma_logn = lognormal_shape_from_mean_std(mean, std)
    mu_logn = np.log(mean) - 0.5 * sigma_logn**2

    return lognorm.logpdf(x, s=sigma_logn, scale=np.exp(mu_logn))


def predicted_loop_logpdf(values_nm, mode, prediction, fit_theta, radius_unit_to_nm=1e7):
    """
    Predicted log-PDF for BF or DF data.

    DF = faulted loops only
    BF = mixture of faulted + perfect loops
    """

    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm)]
    values_nm = values_nm[values_nm > 0]

    if len(values_nm) == 0:
        return np.array([])

    Rf_nm = prediction["Rf"] * radius_unit_to_nm
    Rp_nm = prediction["Rp"] * radius_unit_to_nm

    Df_nm = 2.0 * Rf_nm
    Dp_nm = 2.0 * Rp_nm

    Cf = max(float(prediction["Cf"]), 1e-30)
    Cp = max(float(prediction["Cp"]), 1e-30)

    k_f = fit_theta["k_f"]
    k_p = fit_theta["k_p"]

    logpdf_f = lognormal_logpdf_from_mean_and_k(
        x=values_nm,
        mean=Df_nm,
        k=k_f,
    )

    logpdf_p = lognormal_logpdf_from_mean_and_k(
        x=values_nm,
        mean=Dp_nm,
        k=k_p,
    )

    if mode == "DF":
        return logpdf_f

    if mode == "BF":
        wF = Cf / (Cf + Cp)
        wP = Cp / (Cf + Cp)

        return logsumexp(
            np.vstack([
                np.log(wF) + logpdf_f,
                np.log(wP) + logpdf_p,
            ]),
            axis=0,
        )

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_pdf(x_nm, mode, prediction, fit_theta, radius_unit_to_nm=1e7):
    """
    PDF version used only for plotting.
    """

    logpdf = predicted_loop_logpdf(
        values_nm=x_nm,
        mode=mode,
        prediction=prediction,
        fit_theta=fit_theta,
        radius_unit_to_nm=radius_unit_to_nm,
    )

    return np.exp(logpdf)


# ============================================================
# FITTING PARAMETERS
# ============================================================

def unpack_theta(theta_vec, temperatures):
    """
    theta = [
        Em,
        Ea,
        log(P0),
        log(Puf_T1), ..., log(Puf_Tn),
        log(k_f),
        log(k_p)
    ]

    k_f and k_p control the distribution width:
        std_f = k_f * Df
        std_p = k_p * Dp
    """

    Em = theta_vec[0]
    Ea = theta_vec[1]
    P0 = np.exp(theta_vec[2])

    Puf_by_T = {}
    idx = 3

    for T in temperatures:
        Puf_by_T[float(T)] = np.exp(theta_vec[idx])
        idx += 1

    k_f = np.exp(theta_vec[idx])
    k_p = np.exp(theta_vec[idx + 1])

    return {
        "Em": Em,
        "Ea": Ea,
        "P0": P0,
        "Puf_by_T": Puf_by_T,
        "k_f": k_f,
        "k_p": k_p,
    }


def build_theta0_and_bounds(temperatures):
    """
    Build initial guesses and bounds.
    """

    Em0 = 2.8
    Ea0 = 1.9

    # IMPORTANT:
    # P0 must be small because Pcs multiplies Cp^2.
    P0_0 = 1e-12

    Puf0_by_T = [1e-5 for _ in temperatures]

    # Distribution width factors.
    # k = 0.5 means std = 0.5 * predicted mean diameter.
    k_f0 = 0.5
    k_p0 = 0.5

    theta0 = np.array(
        [Em0, Ea0, np.log(P0_0)]
        + [np.log(x) for x in Puf0_by_T]
        + [np.log(k_f0), np.log(k_p0)],
        dtype=float,
    )

    bounds = []

    # Em, eV
    bounds.append((0.1, 6.0))

    # Ea, eV
    bounds.append((0.1, 6.0))

    # log(P0)
    bounds.append((np.log(1e-30), np.log(1e-6)))

    # log(Puf_T)
    for _ in temperatures:
        bounds.append((np.log(1e-10), np.log(1e-2)))

    # log(k_f), log(k_p)
    bounds.append((np.log(0.05), np.log(3.0)))
    bounds.append((np.log(0.05), np.log(3.0)))

    return theta0, bounds


# ============================================================
# SIMULATION
# ============================================================

def simulate_one_temperature(T_C, theta, base_params, y0, t_end_s=3600):
    """
    Simulate RT+ model for one temperature.

    State:
        y = [Ci, Cv, Nf, Np, Cf, Cp]
    """

    T_C = float(T_C)
    T_K = T_C + 273.15

    Em = theta["Em"]
    Ea = theta["Ea"]
    P0 = theta["P0"]
    Puf = theta["Puf_by_T"][T_C]

    params = base_params.copy()
    params["Di"] = diffusion_coeff(params["D0"], Em, T_K)
    params["Pcs"] = coalescence_rate(P0, Ea, T_K)
    params["Puf"] = Puf

    sol = solve_ivp(
        fun=lambda t, y: ODE(t, y, params),
        t_span=(0.0, t_end_s),
        y0=y0,
        method="BDF",
        rtol=1e-6,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    y_final = sol.y[:, -1]

    Ci, Cv, Nf, Np, Cf, Cp = y_final

    Rf = compute_Rx(Nf, Cf, params["b"])
    Rp = compute_Rx(Np, Cp, params["b"])

    return {
        "y_final": y_final,
        "Rf": Rf,
        "Rp": Rp,
        "Cf": Cf,
        "Cp": Cp,
        "Di": params["Di"],
        "Pcs": params["Pcs"],
        "Puf": Puf,
    }


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(
    theta_vec,
    loop_data,
    base_params,
    y0_initial,
    fit_temperatures,
    radius_unit_to_nm=1e7,
    t_end_s=3600,
):
    """
    Negative log-likelihood objective.

    Uses only irradiated data at fit_temperatures.
    """

    theta = unpack_theta(theta_vec, fit_temperatures)

    total_nll = 0.0
    predictions = {}

    for T_C in fit_temperatures:
        try:
            predictions[float(T_C)] = simulate_one_temperature(
                T_C=float(T_C),
                theta=theta,
                base_params=base_params,
                y0=y0_initial,
                t_end_s=t_end_s,
            )
        except Exception:
            return 1e100

    data_to_fit = loop_data[
        (loop_data["irradiated"] == True) &
        (loop_data["temperature_C"].isin(fit_temperatures))
    ].copy()

    if len(data_to_fit) == 0:
        return 1e100

    for (T_C, mode), group in data_to_fit.groupby(["temperature_C", "mode"]):
        values_nm = group["size"].to_numpy(dtype=float)

        logpdf = predicted_loop_logpdf(
            values_nm=values_nm,
            mode=mode,
            prediction=predictions[float(T_C)],
            fit_theta=theta,
            radius_unit_to_nm=radius_unit_to_nm,
        )

        if len(logpdf) == 0:
            return 1e100

        if not np.all(np.isfinite(logpdf)):
            return 1e100

        # Use mean NLL so one dataset does not dominate only because it has more loops.
        total_nll += -np.mean(logpdf)

    if not np.isfinite(total_nll):
        return 1e100

    return total_nll


# ============================================================
# PLOTTING
# ============================================================

def plot_model_vs_data(
    values_nm,
    mode,
    prediction,
    fit_theta,
    radius_unit_to_nm=1e7,
    title="",
    bins=20,
):
    """
    Plot experimental histogram and fitted RT+ predicted distribution.
    """

    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm)]
    values_nm = values_nm[values_nm > 0]

    if len(values_nm) == 0:
        print(f"No valid data for {title}")
        return

    Rf_nm = prediction["Rf"] * radius_unit_to_nm
    Rp_nm = prediction["Rp"] * radius_unit_to_nm

    Df_nm = 2.0 * Rf_nm
    Dp_nm = 2.0 * Rp_nm

    x_max = max(values_nm.max() * 1.2, Df_nm * 1.5, Dp_nm * 1.5)
    x = np.linspace(1e-9, x_max, 500)

    pdf_model = predicted_loop_pdf(
        x_nm=x,
        mode=mode,
        prediction=prediction,
        fit_theta=fit_theta,
        radius_unit_to_nm=radius_unit_to_nm,
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        values_nm,
        bins=bins,
        density=True,
        alpha=0.65,
        edgecolor="black",
        label="Experimental data",
    )

    plt.plot(
        x,
        pdf_model,
        linewidth=2.5,
        label="RT+ fitted distribution",
    )

    plt.axvline(
        Df_nm,
        linestyle="--",
        linewidth=1.5,
        label=f"Faulted diameter = {Df_nm:.2f} nm",
    )

    if mode == "BF":
        plt.axvline(
            Dp_nm,
            linestyle=":",
            linewidth=1.5,
            label=f"Perfect diameter = {Dp_nm:.2f} nm",
        )

    plt.title(title)
    plt.xlabel("Loop diameter (nm)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_model_vs_data(
    loop_data,
    fit_theta,
    base_params,
    y0_initial,
    fit_temperatures,
    radius_unit_to_nm=1e7,
    t_end_s=3600,
    bins=20,
):
    """
    Plot fitted model against irradiated datasets.
    """

    predictions = {}

    for T_C in fit_temperatures:
        predictions[float(T_C)] = simulate_one_temperature(
            T_C=float(T_C),
            theta=fit_theta,
            base_params=base_params,
            y0=y0_initial,
            t_end_s=t_end_s,
        )

    data_to_plot = loop_data[
        (loop_data["irradiated"] == True) &
        (loop_data["temperature_C"].isin(fit_temperatures))
    ].copy()

    for (T_C, mode), group in data_to_plot.groupby(["temperature_C", "mode"]):
        T_C = float(T_C)
        values_nm = group["size"].to_numpy(dtype=float)

        title = f"{T_C:g} °C - {mode} - irradiated"

        plot_model_vs_data(
            values_nm=values_nm,
            mode=mode,
            prediction=predictions[T_C],
            fit_theta=fit_theta,
            radius_unit_to_nm=radius_unit_to_nm,
            title=title,
            bins=bins,
        )

    return predictions

# ============================================================
# RUN FIT
# ============================================================

# Fixed material / geometric constants
a = 5.41e-8
b = a * np.sqrt(3.0) / 3.0
Omega0 = a**3 / 4.0
r0 = 2.0 * a
Rii = (np.sqrt(3.0) / 2.0) * a
Zii = 12.0


# Choose only temperatures that actually exist in your irradiated data.
available_temperatures = sorted(
    float(T) for T in loop_data[loop_data["irradiated"]]["temperature_C"].dropna().unique()
)

print("Available irradiated temperatures:", available_temperatures)

# Use all available irradiated temperatures.
# Or manually set, e.g. FIT_TEMPERATURES = [900.0, 1100.0, 1300.0]
FIT_TEMPERATURES = available_temperatures

# Random physical initial condition
y0_initial = make_random_y0(b=b, seed=10)

# Fixed model parameters
base_params = {
    "b": b,
    "Omega0": Omega0,
    "r0": r0,
    "D0": 1e6,
    "Rii": Rii,
    "Zii": Zii,
}

theta0, bounds = build_theta0_and_bounds(FIT_TEMPERATURES)

result = minimize(
    objective,
    theta0,
    args=(loop_data, base_params, y0_initial, FIT_TEMPERATURES),
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 5000,
        "ftol": 1e-9,
        "gtol": 1e-6,
    },
)

fit_theta = unpack_theta(result.x, FIT_TEMPERATURES)

print("Fit success:", result.success)
print("Message:", result.message)
print("Final objective:", result.fun)
print("Fit parameters:")
print(fit_theta)

predictions = plot_all_model_vs_data(
    loop_data=loop_data,
    fit_theta=fit_theta,
    base_params=base_params,
    y0_initial=y0_initial,
    fit_temperatures=FIT_TEMPERATURES,
    radius_unit_to_nm=1e7,
    t_end_s=3600,
    bins=20,
)