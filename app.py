

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our own modules
from data_loader import load_data, get_summary_stats, PRODUCT_FACTORY, FACTORY_COORDS, REGION_COORDS, haversine_distance
from ml_model    import train_models, predict_lead_time
from optimizer   import simulate_factory_options, generate_recommendations, get_risk_summary

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Factory Optimizer",
    page_icon="🍬",
    layout="wide",
)

# ── Custom CSS (makes the app look cleaner) ──────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size: 2rem; font-weight: 700; color: #3E2463; margin-bottom: 0;}
    .sub-title  {font-size: 1rem; color: #666; margin-bottom: 1.5rem;}
    .section-header {font-size: 1.2rem; font-weight: 600; margin: 1rem 0 0.5rem 0;}
    .risk-high   {color: #d63031; font-weight: 600;}
    .risk-medium {color: #e17055; font-weight: 600;}
    .risk-low    {color: #00b894; font-weight: 600;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Candy_corn_straws.jpg/320px-Candy_corn_straws.jpg", use_column_width=True)
    st.markdown("## Nassau Candy")
    st.markdown("**Factory Optimization System**")
    st.markdown("---")
    data_path = st.text_input(
        "CSV file path",
        value="Nassau_Candy_Distributor.csv",
        help="Put your CSV file in the same folder as app.py"
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("Go to", [
        "Overview",
        "Factory Simulator",
        "What-If Analysis",
        "Recommendations",
        "Risk & Impact",
    ])

# ── Load data and train models (cached so it only runs once) ─────────────────
@st.cache_data
def load_and_train(path):
    """
    This function is decorated with @st.cache_data.
    That means Streamlit saves the result after the first run.
    If you don't change anything, it won't re-run — making the app fast.
    """
    df = load_data(path)
    results, encoders, features = train_models(df)
    return df, results, encoders, features

# Show a spinner while loading
with st.spinner("Loading data and training models... (first load takes ~30 seconds)"):
    try:
        df, model_results, encoders, features = load_and_train(data_path)
        best_model_name = min(model_results, key=lambda k: model_results[k]["MAE"])
        best_model      = model_results[best_model_name]["model"]
    except FileNotFoundError:
        st.error(f"Could not find the file: **{data_path}**\n\n"
                  "Please check the file path in the sidebar.")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="main-title">Nassau Candy — Factory Optimization Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Shipping analytics and factory performance at a glance</div>', unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    stats = get_summary_stats(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Orders",      f"{stats['total_orders']:,}")
    c2.metric("Avg Lead Time",     f"{stats['avg_lead_time']} days")
    c3.metric("Products",          stats["total_products"])
    c4.metric("Factories",         stats["total_factories"])
    c5.metric("Avg Gross Profit",  f"${stats['avg_profit']}")
    c6.metric("Total Sales",       f"${stats['total_sales']:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Lead time by factory ─────────────────────────────────────────────────
    with col1:
        st.markdown("#### Average lead time by factory (days)")
        factory_lt = df.groupby("Factory")["Lead Time"].mean().reset_index()
        factory_lt.columns = ["Factory", "Avg Lead Time"]
        fig = px.bar(factory_lt, x="Factory", y="Avg Lead Time",
                     color="Factory", text_auto=".0f",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Days",
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Gross profit by factory ──────────────────────────────────────────────
    with col2:
        st.markdown("#### Average gross profit by factory ($)")
        factory_profit = df.groupby("Factory")["Gross Profit"].mean().reset_index()
        factory_profit.columns = ["Factory", "Avg Profit"]
        fig2 = px.bar(factory_profit, x="Factory", y="Avg Profit",
                      color="Factory", text_auto=".2f",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="$",
                           margin=dict(l=0, r=0, t=10, b=0))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # ── Lead time by ship mode ───────────────────────────────────────────────
    with col3:
        st.markdown("#### Lead time by ship mode")
        ship_lt = df.groupby("Ship Mode")["Lead Time"].mean().reset_index()
        fig3 = px.bar(ship_lt, x="Ship Mode", y="Lead Time",
                      color="Ship Mode", text_auto=".0f",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(showlegend=False, xaxis_title="", yaxis_title="Days",
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Lead time by region ──────────────────────────────────────────────────
    with col4:
        st.markdown("#### Lead time by region")
        region_lt = df.groupby("Region")["Lead Time"].mean().reset_index()
        fig4 = px.bar(region_lt, x="Region", y="Lead Time",
                      color="Region", text_auto=".0f",
                      color_discrete_sequence=px.colors.qualitative.Antique)
        fig4.update_layout(showlegend=False, xaxis_title="", yaxis_title="Days",
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Lead time over time ──────────────────────────────────────────────────
    st.markdown("#### Lead time trend by order month")
    df_trend = df.copy()
    df_trend["Month"] = df_trend["Order Date"].dt.to_period("M").astype(str)
    monthly = df_trend.groupby("Month")["Lead Time"].mean().reset_index()
    fig5 = px.line(monthly, x="Month", y="Lead Time", markers=True)
    fig5.update_layout(xaxis_title="Month", yaxis_title="Avg Lead Time (days)",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – FACTORY SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "Factory Simulator":
    st.markdown("## Factory Simulator")
    st.markdown("Select a product, region, and ship mode to see how lead times would vary if the product were made at each factory.")

    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("Product", sorted(PRODUCT_FACTORY.keys()))
    with col2:
        region  = st.selectbox("Region",  ["All Regions"] + list(REGION_COORDS.keys()))
    with col3:
        ship    = st.selectbox("Ship Mode", df["Ship Mode"].unique().tolist())

    st.markdown("---")

    # Run simulation
    options = simulate_factory_options(df, best_model, encoders, features, product, region, ship)

    current_factory = PRODUCT_FACTORY[product]
    current_lt      = next(o["Avg Lead Time"] for o in options if o["Is Current"])
    best_option     = options[0]  # Sorted by lead time — first = best
    days_saved      = current_lt - best_option["Avg Lead Time"]

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current factory",    current_factory)
    c2.metric("Current lead time",  f"{current_lt} days")
    c3.metric("Best alternative",   best_option["Factory"])
    c4.metric("Potential saving",   f"{days_saved:.1f} days",
              delta=f"{-days_saved:.1f} days" if days_saved > 0 else None)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    df_sim = pd.DataFrame(options)
    colors = ["#e74c3c" if r["Is Current"] else "#3498db" for r in options]
    fig = go.Figure(go.Bar(
        x=df_sim["Factory"],
        y=df_sim["Avg Lead Time"],
        marker_color=colors,
        text=df_sim["Avg Lead Time"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted avg lead time per factory (red = current, blue = alternatives)",
        yaxis_title="Lead Time (days)",
        xaxis_title="",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### Detailed comparison")
    df_display = df_sim.copy()
    df_display["Status"] = df_display["Is Current"].map({True: "Current", False: "Alternative"})
    df_display = df_display.drop(columns=["Is Current"])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – WHAT-IF ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "What-If Analysis":
    st.markdown("## What-If Analysis")
    st.markdown("Compare the current factory assignment against a specific alternative, by region.")

    col1, col2, col3 = st.columns(3)
    with col1:
        product  = st.selectbox("Product", sorted(PRODUCT_FACTORY.keys()))
    with col2:
        alt_factory = st.selectbox("Alternative factory", list(FACTORY_COORDS.keys()))
    with col3:
        ship = st.selectbox("Ship mode", df["Ship Mode"].unique().tolist())

    priority = st.slider(
        "Optimization priority (0 = pure speed, 100 = pure profit)",
        min_value=0, max_value=100, value=50, step=1,
        help="Moves the weight between minimising lead time vs. protecting profit margin"
    )
    priority_label = (
        "Speed focus"    if priority < 20 else
        "Speed-leaning"  if priority < 40 else
        "Balanced"       if priority < 60 else
        "Profit-leaning" if priority < 80 else
        "Profit focus"
    )
    st.caption(f"Priority: **{priority_label}**")

    current_factory = PRODUCT_FACTORY[product]
    product_rows    = df[df["Product Name"] == product]
    avg_sales = product_rows["Sales"].mean()
    avg_units = product_rows["Units"].mean()
    avg_cost  = product_rows["Cost"].mean()
    avg_profit = product_rows["Gross Profit"].mean()

    # Predict lead times for current and alternative across all regions
    current_lts = {}
    alt_lts     = {}
    for reg, rcoords in REGION_COORDS.items():
        cur_dist = haversine_distance(
            FACTORY_COORDS[current_factory]["lat"], FACTORY_COORDS[current_factory]["lng"],
            rcoords["lat"], rcoords["lng"]
        )
        alt_dist = haversine_distance(
            FACTORY_COORDS[alt_factory]["lat"], FACTORY_COORDS[alt_factory]["lng"],
            rcoords["lat"], rcoords["lng"]
        )
        current_lts[reg] = predict_lead_time(
            best_model, encoders, features,
            product, current_factory, reg, ship, cur_dist, avg_sales, avg_units, avg_cost
        )
        alt_lts[reg] = predict_lead_time(
            best_model, encoders, features,
            product, alt_factory, reg, ship, alt_dist, avg_sales, avg_units, avg_cost
        )

    avg_cur = np.mean(list(current_lts.values()))
    avg_alt = np.mean(list(alt_lts.values()))
    days_saved = avg_cur - avg_alt
    pct_saved  = (days_saved / avg_cur * 100) if avg_cur > 0 else 0

    # Blend score (weighted combination of speed and profit signals)
    speed_score  = max(0, days_saved * 2)          # More days saved = higher score
    profit_score = min(100, avg_profit * 3)         # Higher profit = higher score
    blend_score  = round((speed_score  * (100 - priority) / 100) +
                         (profit_score * priority / 100))

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current avg LT",    f"{avg_cur:.0f} days")
    c2.metric("Alternative avg LT", f"{avg_alt:.0f} days")
    c3.metric("Days saved",         f"{days_saved:.1f}",
              delta=f"{-days_saved:.1f}" if days_saved != 0 else None)
    c4.metric("Blend score",        str(blend_score))

    # Regional comparison chart
    regions = list(REGION_COORDS.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"Current ({current_factory})",
                         x=regions, y=[current_lts[r] for r in regions],
                         marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name=f"Alternative ({alt_factory})",
                         x=regions, y=[alt_lts[r] for r in regions],
                         marker_color="#2980b9"))
    fig.update_layout(barmode="group", xaxis_title="Region",
                      yaxis_title="Lead Time (days)",
                      title="Lead time comparison by region",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Summary text
    st.markdown("#### What-if summary")
    risk = "High" if avg_profit < 5 else "Medium" if avg_profit < 15 else "Low"
    verdict = (
        "Reassignment **recommended**. Meaningful improvement with acceptable profit stability."
        if days_saved > 10 and risk != "High" else
        "**Caution**: very thin profit margin. Proceed only if volume guarantees offset costs."
        if risk == "High" else
        "**Marginal** improvement. Consider only if current factory is at capacity."
        if days_saved > 0 else
        "Not recommended. Alternative factory shows **no improvement** for this product-region combo."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Product:** {product}  \n"
                f"**Current factory:** {current_factory}  \n"
                f"**Alternative factory:** {alt_factory}  \n"
                f"**Avg profit/order:** ${avg_profit:.2f}  \n"
                f"**Profit risk level:** {risk}")
    with col_b:
        st.info(f"**Verdict:** {verdict}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("## Factory Reassignment Recommendations")
    st.markdown("Ranked list of the most impactful factory reassignments based on predicted lead time reduction.")

    # Model performance table
    st.markdown("### Model performance comparison")
    model_df = pd.DataFrame([
        {
            "Model":  name,
            "MAE (days)": res["MAE"],
            "RMSE":       res["RMSE"],
            "R²":         res["R2"],
            "Selected":   "✅ Yes" if name == best_model_name else "No",
        }
        for name, res in model_results.items()
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    st.caption(f"Selected model: **{best_model_name}** (lowest MAE = smallest average error)")

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        div_filter  = st.selectbox("Filter by division", ["All"] + sorted(df["Division"].unique().tolist()))
    with col2:
        risk_filter = st.selectbox("Filter by risk level", ["All", "Low", "Medium", "High"])

    # Generate recommendations
    with st.spinner("Generating recommendations..."):
        recs = generate_recommendations(df, best_model, encoders, features, top_n=15)

    if div_filter  != "All":
        recs = [r for r in recs if r["Division"]    == div_filter]
    if risk_filter != "All":
        recs = [r for r in recs if r["Risk Level"]  == risk_filter]

    if not recs:
        st.warning("No recommendations match your filters.")
    else:
        recs_df = pd.DataFrame(recs)
        # Colour the risk column
        def highlight_risk(val):
            color = {"High": "#ffe0e0", "Medium": "#fff3cd", "Low": "#d4edda"}.get(val, "")
            return f"background-color: {color}"

        styled = recs_df.style.map(highlight_risk, subset=["Risk Level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Chart: days saved per product
        fig = px.bar(recs_df, x="Product", y="Days Saved",
                     color="Risk Level",
                     color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
                     title="Days saved by reassignment")
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=0, r=0, t=40, b=60))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 – RISK & IMPACT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Risk & Impact":
    st.markdown("## Risk & Impact Panel")
    st.markdown("Financial safety checks and high-risk reassignment warnings.")

    recs = generate_recommendations(df, best_model, encoders, features, top_n=15)
    risk_summary = get_risk_summary(recs)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High risk",         risk_summary["High"],
              delta="needs caution", delta_color="inverse")
    c2.metric("Medium risk",       risk_summary["Medium"])
    c3.metric("Low risk",          risk_summary["Low"],
              delta="safe to proceed", delta_color="normal")
    c4.metric("Total recommendations", len(recs))

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Bubble chart: profit vs. days saved
    with col1:
        st.markdown("#### Profit vs. lead time improvement")
        recs_df = pd.DataFrame(recs)
        fig = px.scatter(
            recs_df,
            x="Days Saved",
            y="Avg Profit ($)",
            size="Improvement (%)",
            color="Risk Level",
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
            hover_name="Product",
            title="Bubble size = improvement %",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Risk breakdown pie
    with col2:
        st.markdown("#### Risk distribution")
        labels = ["High risk", "Medium risk", "Low risk"]
        values = [risk_summary["High"], risk_summary["Medium"], risk_summary["Low"]]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=["#e74c3c", "#f39c12", "#27ae60"],
            hole=0.4,
        ))
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Alert cards
    st.markdown("#### Reassignment alerts")
    for r in recs:
        if r["Risk Level"] == "High":
            st.error(
                f"**HIGH RISK — {r['Product']}**  \n"
                f"Move from {r['Current Factory']} → {r['Recommended Factory']}  \n"
                f"Avg profit: **${r['Avg Profit ($)']:.2f}/order** — margin is very thin. "
                f"Any cost increase will erase profit entirely.  \n"
                f"Lead time saving: {r['Days Saved']} days ({r['Improvement (%)']}%)"
            )
        elif r["Risk Level"] == "Medium":
            st.warning(
                f"**MEDIUM RISK — {r['Product']}**  \n"
                f"Move from {r['Current Factory']} → {r['Recommended Factory']}  \n"
                f"Avg profit: **${r['Avg Profit ($)']:.2f}/order** — monitor volume closely.  \n"
                f"Lead time saving: {r['Days Saved']} days ({r['Improvement (%)']}%)"
            )
        else:
            st.success(
                f"**LOW RISK — {r['Product']}**  \n"
                f"Move from {r['Current Factory']} → {r['Recommended Factory']}  \n"
                f"Avg profit: **${r['Avg Profit ($)']:.2f}/order** — healthy margin.  \n"
                f"Lead time saving: {r['Days Saved']} days ({r['Improvement (%)']}%) ✅"
            )
