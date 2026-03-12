from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import streamlit as st

PRODUCTS = ["Rolls", "Croissant", "Cake", "Sandwich", "Coffee"]

DAYS_IN_MONTH = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

BASE_MONTH_LAMBDA = {
    1: 230,
    2: 240,
    3: 280,
    4: 300,
    5: 320,
    6: 360,
    7: 290,
    8: 260,
    9: 340,
    10: 330,
    11: 310,
    12: 370,
}

RAIN_PROB = {
    1: 0.36,
    2: 0.35,
    3: 0.33,
    4: 0.36,
    5: 0.37,
    6: 0.34,
    7: 0.30,
    8: 0.24,
    9: 0.28,
    10: 0.33,
    11: 0.38,
    12: 0.37,
}


@dataclass
class Product:
    price: float
    unit_cost: float
    salvage: float
    waste_cost: float


@dataclass
class Costs:
    fixed_cost_per_day: float
    extra_staff_cost: float
    energy_cost_per_day: float


@dataclass
class DemandCfg:
    demand_noise_sd: float
    dispersion_k: float
    event_prob_weekday: float


DEFAULT_PRODUCTS: Dict[str, Product] = {
    "Rolls": Product(price=0.55, unit_cost=0.20, salvage=0.10, waste_cost=0.03),
    "Croissant": Product(price=1.70, unit_cost=0.62, salvage=0.40, waste_cost=0.04),
    "Cake": Product(price=3.30, unit_cost=1.30, salvage=0.90, waste_cost=0.07),
    "Sandwich": Product(price=2.80, unit_cost=1.15, salvage=0.65, waste_cost=0.06),
    "Coffee": Product(price=2.10, unit_cost=0.70, salvage=0.20, waste_cost=0.05),
}


def poisson_knuth(lmbda: float, rng: random.Random) -> int:
    if lmbda <= 0:
        return 0
    if lmbda > 60:
        return max(0, int(round(rng.gauss(lmbda, math.sqrt(lmbda)))))
    l_cut = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > l_cut:
        k += 1
        p *= rng.random()
    return k - 1


def draw_customers(lmbda: float, k: float, rng: random.Random) -> int:
    if lmbda <= 0:
        return 0
    if k <= 0:
        return poisson_knuth(lmbda, rng)
    lam_draw = rng.gammavariate(k, lmbda / k)
    return poisson_knuth(lam_draw, rng)


def weather_factor(month: int, rng: random.Random) -> Tuple[str, float]:
    rain_prob = max(0.0, min(0.95, RAIN_PROB[month]))
    storm_prob = min(0.08, rain_prob * 0.2)
    u = rng.random()
    if u < storm_prob:
        return "storm", 0.68
    if u < rain_prob:
        return "rain", 0.84
    if u < rain_prob + 0.22:
        return "cloudy", 0.95
    return "clear", 1.04


def items_weights(month: int, is_weekend: bool, exam_day: bool, event_day: bool) -> List[float]:
    w = [0.24, 0.34, 0.27, 0.12, 0.03] if not is_weekend else [0.34, 0.36, 0.20, 0.08, 0.02]
    if month == 12:
        w = [0.17, 0.29, 0.30, 0.18, 0.06]
    if exam_day:
        w = [0.16, 0.30, 0.26, 0.22, 0.06]
    if event_day:
        w = [0.14, 0.26, 0.26, 0.24, 0.10]
    s = sum(w)
    return [x / s for x in w]


def basket_probs(month: int, is_weekend: bool, exam_day: bool, event_day: bool) -> Dict[str, float]:
    p = {"Rolls": 0.33, "Croissant": 0.22, "Cake": 0.12, "Sandwich": 0.16, "Coffee": 0.17}
    if month in (11, 12, 1, 2):
        p["Cake"] += 0.07
        p["Rolls"] -= 0.03
        p["Sandwich"] -= 0.01
        p["Coffee"] -= 0.03
    if is_weekend:
        p["Cake"] += 0.05
        p["Coffee"] -= 0.02
        p["Sandwich"] -= 0.03
    if exam_day:
        p["Coffee"] += 0.08
        p["Sandwich"] += 0.04
        p["Cake"] -= 0.04
        p["Rolls"] -= 0.04
        p["Croissant"] -= 0.04
    if event_day:
        p["Coffee"] += 0.04
        p["Sandwich"] += 0.06
        p["Cake"] -= 0.03
        p["Rolls"] -= 0.04
        p["Croissant"] -= 0.03
    s = sum(max(0.001, x) for x in p.values())
    return {k: max(0.001, v) / s for k, v in p.items()}


def q(xs: Sequence[float], level: float) -> float:
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * level))))
    return sorted(xs)[k]


def simulate_one_year(
    rng: random.Random,
    products: Dict[str, Product],
    costs: Costs,
    demand: DemandCfg,
    plan: Dict[str, int],
    extra_staff: int,
    safety: float,
    cap_gain: float,
) -> Tuple[float, List[float], Dict[str, float]]:
    monthly = []
    total_demand = 0
    total_sold = 0
    stockout_total = 0
    waste_total = 0
    total_revenue = 0.0
    total_prod_cost = 0.0
    total_fixed = 0.0
    total_salvage = 0.0
    total_waste_cost = 0.0
    dow = 0
    weekday_factor = (1.05, 1.08, 1.10, 1.12, 1.18, 0.76, 0.70)

    for month in range(1, 13):
        month_profit = 0.0
        for _ in range(DAYS_IN_MONTH[month]):
            is_weekend = dow in (5, 6)
            exam_day = (month in (1, 6, 9)) and (not is_weekend) and rng.random() < 0.35
            event_day = (not is_weekend) and rng.random() < demand.event_prob_weekday
            weather, weather_mult = weather_factor(month, rng)
            noise_mult = max(0.55, min(1.55, 1.0 + rng.gauss(0.0, demand.demand_noise_sd)))

            lam = BASE_MONTH_LAMBDA[month] * weekday_factor[dow] * weather_mult * noise_mult
            if exam_day:
                lam *= 1.12
            if event_day:
                lam *= 1.20

            customers = draw_customers(lam, demand.dispersion_k, rng)
            iw = items_weights(month, is_weekend, exam_day, event_day)
            pw = basket_probs(month, is_weekend, exam_day, event_day)
            expected_items = lam * (1 * iw[0] + 2 * iw[1] + 3 * iw[2] + 4 * iw[3] + 5 * iw[4])

            produced_today = {}
            for name in PRODUCTS:
                cap = int(round(plan[name] * (1.0 + extra_staff * cap_gain)))
                target = expected_items * pw[name] * safety
                produced_today[name] = min(cap, max(0, int(round(target))))

            if customers > 0:
                total_items = sum(rng.choices([1, 2, 3, 4, 5], weights=iw, k=customers))
                chosen = rng.choices(PRODUCTS, weights=[pw[n] for n in PRODUCTS], k=total_items)
                demand_units = Counter(chosen)
            else:
                demand_units = Counter()

            rev = 0.0
            prod_cost = 0.0
            salv = 0.0
            waste_cost = 0.0
            for name in PRODUCTS:
                produced = produced_today[name]
                demanded = demand_units.get(name, 0)
                sold = min(produced, demanded)
                stockout = max(0, demanded - produced)
                waste = max(0, produced - demanded)
                p = products[name]
                rev += sold * p.price
                prod_cost += produced * p.unit_cost
                salv += waste * p.salvage
                waste_cost += waste * p.waste_cost
                total_demand += demanded
                total_sold += sold
                stockout_total += stockout
                waste_total += waste

            fixed = costs.fixed_cost_per_day + extra_staff * costs.extra_staff_cost + costs.energy_cost_per_day * (1.1 if weather in ("rain", "storm") else 1.0)
            day_profit = (rev + salv) - (prod_cost + waste_cost + fixed)
            month_profit += day_profit
            total_revenue += rev
            total_prod_cost += prod_cost
            total_fixed += fixed
            total_salvage += salv
            total_waste_cost += waste_cost
            dow = (dow + 1) % 7
        monthly.append(month_profit)

    service = total_sold / total_demand if total_demand > 0 else 0.0
    return sum(monthly), monthly, {
        "revenue": total_revenue,
        "production_cost": total_prod_cost,
        "fixed_cost": total_fixed,
        "salvage": total_salvage,
        "waste_cost": total_waste_cost,
        "stockout_units": float(stockout_total),
        "waste_units": float(waste_total),
        "service_level": service,
    }


def run_mc(runs: int, seed: int, products: Dict[str, Product], costs: Costs, demand: DemandCfg, plan: Dict[str, int], extra_staff: int, safety: float, cap_gain: float) -> Dict[str, object]:
    rng = random.Random(seed)
    profits = []
    stockouts = []
    services = []
    monthly_avg = [0.0] * 12
    for _ in range(runs):
        y, m, kpi = simulate_one_year(rng, products, costs, demand, plan, extra_staff, safety, cap_gain)
        profits.append(y)
        stockouts.append(kpi["stockout_units"])
        services.append(kpi["service_level"])
        for i in range(12):
            monthly_avg[i] += m[i]
    monthly_avg = [x / runs for x in monthly_avg]
    tail_n = max(1, int(round(0.05 * runs)))
    return {
        "mean_profit": statistics.mean(profits),
        "stdev_profit": statistics.pstdev(profits) if runs > 1 else 0.0,
        "p05": q(profits, 0.05),
        "p50": q(profits, 0.50),
        "p95": q(profits, 0.95),
        "cvar5": statistics.mean(sorted(profits)[:tail_n]),
        "mean_stockout": statistics.mean(stockouts),
        "mean_service": statistics.mean(services),
        "profits": profits,
        "monthly_avg": monthly_avg,
    }


def demand_diagnostic(seed: int, samples: int, demand: DemandCfg) -> Dict[str, float | str]:
    rng = random.Random(seed + 777)
    vals = []
    dow = 0
    m = 1
    d = 0
    wf = (1.05, 1.08, 1.10, 1.12, 1.18, 0.76, 0.70)
    for _ in range(samples):
        if d >= DAYS_IN_MONTH[m]:
            m = m + 1 if m < 12 else 1
            d = 0
        is_weekend = dow in (5, 6)
        exam_day = (m in (1, 6, 9)) and (not is_weekend) and rng.random() < 0.35
        event_day = (not is_weekend) and rng.random() < demand.event_prob_weekday
        _, w = weather_factor(m, rng)
        noise = max(0.55, min(1.55, 1.0 + rng.gauss(0.0, demand.demand_noise_sd)))
        lam = BASE_MONTH_LAMBDA[m] * wf[dow] * w * noise * (1.12 if exam_day else 1.0) * (1.20 if event_day else 1.0)
        vals.append(draw_customers(lam, demand.dispersion_k, rng))
        dow = (dow + 1) % 7
        d += 1
    mean_v = statistics.mean(vals)
    var_v = statistics.pvariance(vals)
    p0_emp = sum(1 for x in vals if x == 0) / len(vals)
    p0_pois = math.exp(-mean_v)
    if var_v > mean_v:
        k_est = (mean_v * mean_v) / (var_v - mean_v)
        p0_nb = (k_est / (k_est + mean_v)) ** k_est
    else:
        k_est = float("inf")
        p0_nb = p0_pois
    reco = "Negative Binomial" if abs(p0_nb - p0_emp) < abs(p0_pois - p0_emp) else "Poisson"
    return {
        "mean": mean_v,
        "variance": var_v,
        "fano": var_v / mean_v if mean_v > 0 else 0.0,
        "p0_emp": p0_emp,
        "p0_pois": p0_pois,
        "p0_nb": p0_nb,
        "k_est": k_est,
        "recommended_model": reco,
    }


def plot_monthly(monthly_avg: Sequence[float]):
    fig = plt.figure()
    x = list(range(1, 13))
    plt.plot(x, monthly_avg, marker="o")
    plt.title("Average monthly profit")
    plt.xlabel("Month")
    plt.ylabel("Profit [EUR]")
    plt.xticks(x)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_hist(values: Sequence[float]):
    fig = plt.figure()
    plt.hist(values, bins=18, edgecolor="black")
    plt.title("Yearly profit distribution")
    plt.xlabel("Profit [EUR]")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.20)
    plt.tight_layout()
    return fig


def score_table(runs: int, seed: int, products: Dict[str, Product], costs: Costs, demand: DemandCfg, plan: Dict[str, int], cap_gain: float, staff_min: int, staff_max: int, safety_values: Sequence[float], penalty_stockout: float) -> List[Tuple[int, float, float, float, float, float]]:
    rows = []
    for staff in range(staff_min, staff_max + 1):
        for i, safety in enumerate(safety_values):
            res = run_mc(runs, seed + 1009 * staff + 17 * i, products, costs, demand, plan, staff, safety, cap_gain)
            score = float(res["mean_profit"]) - penalty_stockout * float(res["mean_stockout"]) - max(0.0, 0.98 - float(res["mean_service"])) * 10000.0
            rows.append((staff, safety, float(res["mean_profit"]), float(res["mean_stockout"]), float(res["mean_service"]), score))
    return rows


st.set_page_config(page_title="INF23 Final Project Simulation", layout="wide")
st.title("INF23 Final Project - Expanded Campus Bakery Simulation")
st.caption("Larger version with stochastic demand drivers, diagnostics, scenarios, and optimization.")

with st.sidebar:
    runs = st.number_input("Runs", 20, 500, 160, 20)
    seed = st.number_input("Seed", 0, 99999, 42, 1)
    st.subheader("Production plan (base max/day)")
    prod_rolls = st.number_input("Rolls/day", 0, 500, 180, 10)
    prod_croissant = st.number_input("Croissant/day", 0, 300, 120, 5)
    prod_cake = st.number_input("Cake/day", 0, 180, 70, 5)
    prod_sandwich = st.number_input("Sandwich/day", 0, 300, 130, 5)
    prod_coffee = st.number_input("Coffee/day", 0, 300, 150, 5)
    st.subheader("Costs")
    fixed_cost = st.number_input("Fixed/day [EUR]", 0.0, 600.0, 130.0, 10.0)
    staff_cost = st.number_input("Extra staff/day [EUR]", 0.0, 300.0, 28.0, 2.0)
    energy_cost = st.number_input("Energy/day [EUR]", 0.0, 200.0, 24.0, 1.0)
    st.subheader("Demand")
    noise_sd = st.slider("Demand noise SD", 0.00, 0.60, 0.14, 0.01)
    dispersion_k = st.slider("Overdispersion k", 0.8, 30.0, 6.0, 0.2)
    event_prob = st.slider("Weekday event probability", 0.00, 0.30, 0.08, 0.01)
    st.subheader("Policy")
    baseline_staff = st.number_input("Extra staff (baseline)", 0, 40, 2, 1)
    safety = st.slider("Safety factor", 0.80, 1.25, 1.05, 0.01)
    cap_gain = st.slider("Capacity gain per staff", 0.00, 0.30, 0.08, 0.01)

plan = {
    "Rolls": int(prod_rolls),
    "Croissant": int(prod_croissant),
    "Cake": int(prod_cake),
    "Sandwich": int(prod_sandwich),
    "Coffee": int(prod_coffee),
}
costs = Costs(float(fixed_cost), float(staff_cost), float(energy_cost))
demand = DemandCfg(float(noise_sd), float(dispersion_k), float(event_prob))
products = DEFAULT_PRODUCTS

tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Scenario Lab", "Optimization", "Requirement Mapping"])

with tab1:
    if st.button("Run baseline simulation"):
        st.session_state["base"] = run_mc(int(runs), int(seed), products, costs, demand, plan, int(baseline_staff), float(safety), float(cap_gain))
    base = st.session_state.get("base")
    if base:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E[Profit]", f"{base['mean_profit']:.2f} EUR")
        c2.metric("Sigma", f"{base['stdev_profit']:.2f}")
        c3.metric("Stockout", f"{base['mean_stockout']:.1f}")
        c4.metric("Service", f"{100.0 * base['mean_service']:.2f}%")
        c5, c6, c7 = st.columns(3)
        c5.metric("P05", f"{base['p05']:.2f}")
        c6.metric("Median", f"{base['p50']:.2f}")
        c7.metric("CVaR5", f"{base['cvar5']:.2f}")
        p1, p2 = st.columns(2)
        p1.pyplot(plot_monthly(base["monthly_avg"]))
        p2.pyplot(plot_hist(base["profits"]))
    else:
        st.info("Click run to compute baseline KPIs.")

with tab2:
    if st.button("Run scenario lab"):
        scenarios = [
            ("Lean", 0, 0.98, max(0.02, float(noise_sd) - 0.02)),
            ("Balanced", 2, 1.04, float(noise_sd)),
            ("Peak", 4, 1.10, min(0.60, float(noise_sd) + 0.03)),
        ]
        rows = []
        for i, (name, staff, sf, n) in enumerate(scenarios):
            d = DemandCfg(n, float(dispersion_k), float(event_prob))
            r = run_mc(int(runs), int(seed) + 701 * i, products, costs, d, plan, staff, sf, float(cap_gain))
            rows.append(
                {
                    "Scenario": name,
                    "Staff": staff,
                    "Safety": sf,
                    "NoiseSD": n,
                    "MeanProfit": round(float(r["mean_profit"]), 2),
                    "P05": round(float(r["p05"]), 2),
                    "Stockout": round(float(r["mean_stockout"]), 1),
                    "ServicePct": round(100.0 * float(r["mean_service"]), 2),
                }
            )
        st.dataframe(rows, use_container_width=True)

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    staff_min = c1.number_input("Staff min", 0, 30, 0, 1)
    staff_max = c2.number_input("Staff max", 0, 30, 10, 1)
    safety_min = c3.number_input("Safety min", 0.80, 1.10, 0.95, 0.01)
    safety_max = c4.number_input("Safety max", 0.95, 1.25, 1.15, 0.01)
    safety_step = st.number_input("Safety step", 0.01, 0.10, 0.04, 0.01)
    penalty = st.number_input("Penalty EUR/stockout", 0.0, 15.0, 0.50, 0.05)
    opt_runs = st.number_input("Optimization runs", 20, 300, max(40, int(runs) // 2), 10)

    sv = []
    x = float(safety_min)
    while x <= float(safety_max) + 1e-9:
        sv.append(round(x, 2))
        x += float(safety_step)

    if st.button("Run optimization grid"):
        if int(staff_max) < int(staff_min):
            st.error("staff_max must be >= staff_min")
        else:
            rows = score_table(int(opt_runs), int(seed), products, costs, demand, plan, float(cap_gain), int(staff_min), int(staff_max), sv, float(penalty))
            best = max(rows, key=lambda t: t[5])
            st.success(f"Best: staff={best[0]}, safety={best[1]:.2f}, score={best[5]:.2f}")
            st.dataframe(
                [{"Staff": n, "Safety": s, "MeanProfit": round(mp, 2), "Stockout": round(sto, 1), "ServicePct": round(100 * sl, 2), "Score": round(sc, 2)} for (n, s, mp, sto, sl, sc) in rows],
                use_container_width=True,
            )

    st.markdown("---")
    diag_days = st.number_input("Diagnostic sample days", 120, 4000, 1000, 20)
    if st.button("Run distribution diagnostics"):
        rep = demand_diagnostic(int(seed), int(diag_days), demand)
        st.write(
            {
                "Mean": round(float(rep["mean"]), 3),
                "Variance": round(float(rep["variance"]), 3),
                "Fano(var/mean)": round(float(rep["fano"]), 3),
                "P0_emp": round(float(rep["p0_emp"]), 5),
                "P0_poisson": round(float(rep["p0_pois"]), 5),
                "P0_neg_bin": round(float(rep["p0_nb"]), 5),
                "k_est": "inf" if math.isinf(float(rep["k_est"])) else round(float(rep["k_est"]), 3),
                "Recommended": str(rep["recommended_model"]),
            }
        )

with tab4:
    st.markdown(
        """
1. Random effects are central (weather, exam days, event days, overdispersion, daily noise).
2. Parameters are explicitly modeled and checked via diagnostics (Poisson vs Negative Binomial).
3. The model supports independent analysis with scenario and optimization workflows.
4. Outputs include risk metrics (P05, CVaR), service level, and monthly dynamics for reporting.
5. This is a larger, report-ready final-project variant of the original simulation.
        """
    )
