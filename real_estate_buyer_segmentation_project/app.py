import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
)

ROOT = Path(__file__).parent
clients_path = ROOT / "outputs/client_segment_assignments.csv"
profiles_path = ROOT / "outputs/segment_profiles.csv"
geo_path = ROOT / "outputs/geographic_segment_summary.csv"
inv_path = ROOT / "outputs/investment_summary.csv"
rec_path = ROOT / "outputs/property_recommendations_by_segment.csv"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data

def load_data():
    return (
        pd.read_csv(clients_path),
        pd.read_csv(profiles_path),
        pd.read_csv(geo_path),
        pd.read_csv(inv_path),
        pd.read_csv(rec_path),
    )


c, prof, geo, inv, recs = load_data()


# ------------------------------------------------------------
# DISPLAY FORMATTING
# Prevent Streamlit from showing long raw decimal values such as
# 43869854.349... in tables/charts.
# ------------------------------------------------------------
def number_column_config(df):
    """
    Build Streamlit column configurations from column names.
    The underlying data stays numeric, but the dashboard displays
    readable values with commas, sensible decimals, currency, and
    percentages.
    """
    config = {}

    for col in df.columns:
        name = str(col)
        n = name.lower()

        # Percentage / share columns: source values are expected to be 0-1.
        if (
            "share" in n
            or "percentage" in n
            or n.endswith("_pct")
            or n.endswith("_percent")
        ):
            config[col] = st.column_config.NumberColumn(
                name.replace("_", " ").title(),
                format="%.1f%%",
            )

        # Currency / money columns.
        elif any(
            word in n
            for word in [
                "total_investment",
                "avg_total_investment",
                "purchase_price",
                "avg_purchase_price",
                "price",
                "investment",
                "amount",
                "income",
                "value",
                "budget",
            ]
        ):
            config[col] = st.column_config.NumberColumn(
                name.replace("_", " ").title(),
                format="$%,.2f",
            )

        # Integer/count columns.
        elif (
            n in {
                "buyers",
                "clients",
                "client_count",
                "total_buyers",
                "count",
                "purchases",
                "client_id",
            }
            or n.endswith("_count")
        ):
            config[col] = st.column_config.NumberColumn(
                name.replace("_", " ").title(),
                format="%,d",
            )

        # Scores.
        elif "score" in n:
            config[col] = st.column_config.NumberColumn(
                name.replace("_", " ").title(),
                format="%.2f",
            )

        # Age / averages / areas / other measurements.
        elif (
            "age" in n
            or "average" in n
            or "avg_" in n
            or "area" in n
        ):
            config[col] = st.column_config.NumberColumn(
                name.replace("_", " ").title(),
                format="%,.2f",
            )

    return config


def show_dataframe(df, *, height=None):
    """Display a dataframe with readable numeric formatting."""
    kwargs = {
        "use_container_width": True,
        "hide_index": True,
        "column_config": number_column_config(df),
    }

    if height is not None:
        kwargs["height"] = height

    st.dataframe(df, **kwargs)


def money(value):
    """Format a numeric value as currency without raw floating-point noise."""
    if pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


def integer(value):
    """Format a count as an integer with thousands separators."""
    if pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}"


def decimal(value, digits=2):
    """Format a decimal without scientific notation or trailing noise."""
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"

# Make sure helper columns exist.
if "is_loan" not in c.columns:
    c["is_loan"] = (
        c["loan_applied"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
    )

if "is_investment" not in c.columns:
    c["is_investment"] = (
        c["acquisition_purpose"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"investment": 1, "home": 0})
        .fillna(0)
    )


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.caption(
    "Machine-learning buyer intelligence dashboard based on the supplied "
    "client and property datasets."
)


# ------------------------------------------------------------
# CASCADING FILTERS
#
# Country
#   -> Region
#       -> Acquisition purpose
#           -> Client type
#               -> Buyer segment
#
# Each lower-level filter is calculated from the currently selected
# values above it. Invalid selections are automatically removed from
# Streamlit session state.
# ------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # --------------------------------------------------------
    # 1. COUNTRY
    # --------------------------------------------------------
    country_options = sorted(c["country"].dropna().unique().tolist())

    country = st.multiselect(
        "Country",
        country_options,
        key="filter_country",
    )

    # --------------------------------------------------------
    # 2. REGION - depends on COUNTRY
    # --------------------------------------------------------
    if country:
        region_base = c[c["country"].isin(country)].copy()
    else:
        region_base = c.copy()

    region_options = sorted(
        region_base["region"].dropna().unique().tolist()
    )

    old_regions = st.session_state.get("filter_region", [])
    if not isinstance(old_regions, list):
        old_regions = list(old_regions) if old_regions else []

    st.session_state["filter_region"] = [
        value for value in old_regions if value in region_options
    ]

    region = st.multiselect(
        "Region",
        region_options,
        key="filter_region",
    )

    # --------------------------------------------------------
    # 3. ACQUISITION PURPOSE - depends on COUNTRY + REGION
    # --------------------------------------------------------
    purpose_base = region_base.copy()

    if region:
        purpose_base = purpose_base[
            purpose_base["region"].isin(region)
        ]

    purpose_options = sorted(
        purpose_base["acquisition_purpose"]
        .dropna()
        .unique()
        .tolist()
    )

    old_purposes = st.session_state.get("filter_purpose", [])
    if not isinstance(old_purposes, list):
        old_purposes = list(old_purposes) if old_purposes else []

    st.session_state["filter_purpose"] = [
        value for value in old_purposes if value in purpose_options
    ]

    purpose = st.multiselect(
        "Acquisition purpose",
        purpose_options,
        key="filter_purpose",
    )

    # --------------------------------------------------------
    # 4. CLIENT TYPE - depends on COUNTRY + REGION + PURPOSE
    # --------------------------------------------------------
    client_type_base = purpose_base.copy()

    if purpose:
        client_type_base = client_type_base[
            client_type_base["acquisition_purpose"].isin(purpose)
        ]

    client_type_options = sorted(
        client_type_base["client_type"]
        .dropna()
        .unique()
        .tolist()
    )

    old_client_types = st.session_state.get(
        "filter_client_type", []
    )
    if not isinstance(old_client_types, list):
        old_client_types = (
            list(old_client_types) if old_client_types else []
        )

    # IMPORTANT: this removes stale selections such as Company
    # when Company does not exist for the current filters.
    st.session_state["filter_client_type"] = [
        value
        for value in old_client_types
        if value in client_type_options
    ]

    ctype = st.multiselect(
        "Client type",
        client_type_options,
        key="filter_client_type",
    )

    # --------------------------------------------------------
    # 5. BUYER SEGMENT - depends on ALL PREVIOUS FILTERS
    # --------------------------------------------------------
    segment_base = client_type_base.copy()

    if ctype:
        segment_base = segment_base[
            segment_base["client_type"].isin(ctype)
        ]

    segment_options = sorted(
        segment_base["segment"].dropna().unique().tolist()
    )

    old_segments = st.session_state.get("filter_segment", [])
    if not isinstance(old_segments, list):
        old_segments = list(old_segments) if old_segments else []

    st.session_state["filter_segment"] = [
        value for value in old_segments if value in segment_options
    ]

    segment = st.multiselect(
        "Buyer segment",
        segment_options,
        key="filter_segment",
    )


# ------------------------------------------------------------
# APPLY FILTERS TO MAIN DATAFRAME
# ------------------------------------------------------------
f = c.copy()

if country:
    f = f[f["country"].isin(country)]

if region:
    f = f[f["region"].isin(region)]

if purpose:
    f = f[f["acquisition_purpose"].isin(purpose)]

if ctype:
    f = f[f["client_type"].isin(ctype)]

if segment:
    f = f[f["segment"].isin(segment)]


# ------------------------------------------------------------
# METRIC CARDS
#
# Regular HTML cards are used instead of st.metric so long currency
# values do not get shortened with "...".
# ------------------------------------------------------------
if f.empty:
    buyers_value = "—"
    investment_value = "—"
    purchase_price_value = "—"
    satisfaction_value = "—"
else:
    buyers_value = integer(f["client_id"].nunique())
    investment_value = money(f["total_investment"].sum())
    purchase_price_value = money(f["avg_purchase_price"].mean())
    satisfaction_value = decimal(f["satisfaction_score"].mean(), 2)

metric_css = """
<style>
.dashboard-metric {
    padding: 4px 0 18px 0;
    min-width: 0;
}
.dashboard-metric-label {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.dashboard-metric-value {
    font-size: 2.05rem;
    line-height: 1.15;
    font-weight: 500;
    white-space: nowrap;
    overflow: visible;
    letter-spacing: -0.02em;
}
</style>
"""
st.markdown(metric_css, unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="dashboard-metric">'
        f'<div class="dashboard-metric-label">Buyers</div>'
        f'<div class="dashboard-metric-value">{buyers_value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f'<div class="dashboard-metric">'
        f'<div class="dashboard-metric-label">Total investment</div>'
        f'<div class="dashboard-metric-value">{investment_value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f'<div class="dashboard-metric">'
        f'<div class="dashboard-metric-label">Avg purchase price</div>'
        f'<div class="dashboard-metric-value">{purchase_price_value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f'<div class="dashboard-metric">'
        f'<div class="dashboard-metric-label">Avg satisfaction</div>'
        f'<div class="dashboard-metric-value">{satisfaction_value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# STOP HERE IF THERE ARE NO MATCHING BUYERS
# ------------------------------------------------------------
if f.empty:
    st.warning(
        "No buyers match the selected filter combination. "
        "Please broaden one or more filters."
    )
    st.stop()


# ------------------------------------------------------------
# BUYER SEGMENTATION OVERVIEW
# ------------------------------------------------------------
st.subheader("Buyer Segmentation Overview")

dist = (
    f.groupby("segment")
    .size()
    .reset_index(name="buyers")
    .sort_values("buyers", ascending=False)
)

dist_fig = px.bar(
    dist,
    x="segment",
    y="buyers",
    text="buyers",
    title="Buyer Segment Distribution",
)
dist_fig.update_traces(
    texttemplate="%{y:,.0f}",
    hovertemplate="<b>%{x}</b><br>Buyers: %{y:,.0f}<extra></extra>",
)
dist_fig.update_layout(
    xaxis_title="Buyer segment",
    yaxis_title="Buyers",
)
st.plotly_chart(dist_fig, use_container_width=True)


# ------------------------------------------------------------
# INVESTOR BEHAVIOR DASHBOARD
# ------------------------------------------------------------
st.subheader("Investor Behavior Dashboard")

a, b = st.columns(2)

with a:
    behavior = (
        f.groupby("segment")
        .agg(
            buyers=("client_id", "count"),
            avg_purchases=("purchases", "mean"),
            avg_total_investment=("total_investment", "mean"),
            avg_purchase_price=("avg_purchase_price", "mean"),
            investment_share=("is_investment", "mean"),
            loan_share=("is_loan", "mean"),
        )
        .round(2)
    )

    show_dataframe(behavior)

with b:
    purpose_chart = (
        f.groupby(["segment", "acquisition_purpose"])
        .size()
        .unstack(fill_value=0)
    )

    purpose_long = (
        purpose_chart
        .reset_index()
        .melt(
            id_vars="segment",
            var_name="acquisition_purpose",
            value_name="buyers",
        )
    )

    purpose_fig = px.bar(
        purpose_long,
        x="segment",
        y="buyers",
        color="acquisition_purpose",
        barmode="group",
        text="buyers",
        title="Acquisition Purpose by Buyer Segment",
    )
    purpose_fig.update_traces(
        texttemplate="%{y:,.0f}",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Acquisition purpose: %{fullData.name}<br>"
            "Buyers: %{y:,.0f}<extra></extra>"
        ),
    )
    purpose_fig.update_layout(
        xaxis_title="Buyer segment",
        yaxis_title="Buyers",
    )
    st.plotly_chart(
        purpose_fig,
        use_container_width=True,
    )


# ------------------------------------------------------------
# GEOGRAPHIC BUYER ANALYSIS
# ------------------------------------------------------------
st.subheader("Geographic Buyer Analysis")

g = (
    f.groupby(["country", "region", "segment"])
    .size()
    .reset_index(name="buyers")
)

# Representative regional coordinates for visualization.
# These are regional visualization points, not property addresses.
REGION_COORDS = {
    "California": (36.7783, -119.4179),
    "Nevada": (38.8026, -116.4194),
    "Colorado": (39.5501, -105.7821),
    "Arizona": (34.0489, -111.0937),
    "Oregon": (43.8041, -120.5542),
    "Utah": (39.3210, -111.0937),
    "Washington": (47.4009, -121.4905),
    "Virginia": (37.4316, -78.6569),
    "Texas": (31.9686, -99.9018),
    "Florida": (27.6648, -81.5158),
    "New York": (42.1657, -74.9481),
    "Georgia": (32.1656, -82.9001),
    "Ohio": (40.4173, -82.9071),
    "Kansas": (39.0119, -98.4842),
    "Wyoming": (43.0760, -107.2900),
    "England": (52.3555, -1.1743),
    "Scotland": (56.4907, -4.2026),
    "Wales": (52.1307, -3.7837),
    "Northern Ireland": (54.7877, -6.4923),
    "Alberta": (53.9333, -116.5765),
    "British Columbia": (53.7267, -127.6476),
    "Manitoba": (53.7609, -98.8139),
    "Ontario": (51.2538, -85.3232),
    "Quebec": (52.9399, -73.5491),
    "Baja California": (30.8406, -115.2838),
    "Jalisco": (20.6597, -103.3496),
    "Mexico City": (19.4326, -99.1332),
    "Nuevo Leon": (25.5922, -99.9962),
    "Puebla": (19.0414, -98.2063),
    "Bavaria": (48.7904, 11.4979),
    "Berlin": (52.5200, 13.4050),
    "Hamburg": (53.5511, 9.9937),
    "Hesse": (50.6521, 9.1624),
    "North Rhine-Westphalia": (51.4332, 7.6616),
    "Brussels": (50.8503, 4.3517),
    "Flanders": (51.0543, 3.7174),
    "Wallonia": (50.5039, 4.4699),
    "Brittany": (48.2020, -2.9326),
    "Ile-de-France": (48.8499, 2.6370),
    "Normandy": (49.1829, 0.3707),
    "Occitanie": (43.8927, 3.2828),
    "Provence": (43.9352, 6.0679),
    "Capital Region": (55.6761, 12.5683),
    "Central Denmark": (56.3021, 9.3020),
    "North Denmark": (57.1180, 9.8492),
    "Southern Denmark": (55.4000, 10.4000),
    "Zealand": (55.4038, 11.3517),
    "Moscow Oblast": (55.1436, 38.8466),
    "Krasnodar Krai": (45.6415, 39.7056),
    "Novosibirsk": (55.0084, 82.9357),
    "Saint Petersburg": (59.9311, 30.3609),
    "Tatarstan": (55.7879, 49.1233),
    "New South Wales": (-31.2532, 146.9211),
    "Queensland": (-20.9176, 142.7028),
    "South Australia": (-30.0002, 136.2092),
    "Victoria": (-36.9848, 143.3906),
    "Western Australia": (-25.0423, 121.7310),
}

if not g.empty:
    idx = g.groupby(["country", "region"])["buyers"].idxmax()
    gm = g.loc[idx].copy()

    total_by_region = (
        g.groupby(["country", "region"])["buyers"]
        .sum()
        .reset_index(name="total_buyers")
    )

    gm = gm.merge(
        total_by_region,
        on=["country", "region"],
        how="left",
    )

    gm["latitude"] = gm["region"].map(
        lambda x: REGION_COORDS.get(x, (np.nan, np.nan))[0]
    )
    gm["longitude"] = gm["region"].map(
        lambda x: REGION_COORDS.get(x, (np.nan, np.nan))[1]
    )

    gm = gm.dropna(subset=["latitude", "longitude"])

    if not gm.empty:
        fig = px.scatter_geo(
            gm,
            lat="latitude",
            lon="longitude",
            size="total_buyers",
            color="segment",
            hover_name="region",
            hover_data={
                "country": True,
                "total_buyers": True,
                "segment": True,
                "buyers": True,
                "latitude": False,
                "longitude": False,
            },
            scope="world",
            projection="natural earth",
            title="Buyer Segments by Geographic Region",
            size_max=35,
        )
        fig.update_geos(showland=True, showcountries=True)

        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Country: %{customdata[0]}<br>"
                "Total buyers: %{customdata[1]:,.0f}<br>"
                "Segment: %{customdata[2]}<br>"
                "Segment buyers: %{customdata[3]:,.0f}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

show_dataframe(
    g.sort_values("buyers", ascending=False).head(50)
)


# ------------------------------------------------------------
# SEGMENT INSIGHTS
# ------------------------------------------------------------
st.subheader("Segment Insights")

for _, r in prof.iterrows():
    st.markdown(
        f"**{r.segment}** — {int(r.clients):,} buyers | "
        f"Avg age {r.avg_age:.1f} | "
        f"Avg purchases {r.avg_purchases:.1f} | "
        f"Avg purchase price ${r.avg_purchase_price:,.0f} | "
        f"Avg total investment ${r.avg_total_investment:,.0f} | "
        f"Loan share {r.loan_share:.0%} | "
        f"Investment-purpose share {r.investment_share:.0%}"
    )


# ------------------------------------------------------------
# PROPERTY RECOMMENDATIONS
# ------------------------------------------------------------
st.subheader("Property Recommendations by Segment")

if segment:
    rr = recs[recs["segment"].isin(segment)]
else:
    rr = recs

if not rr.empty:
    show_dataframe(
        rr.sort_values(["segment", "match_score"]).head(100)
    )
else:
    st.info("No property recommendations match the selected buyer segment.")

st.caption(
    "Recommendation score is a distance to the segment's observed average "
    "purchase price and average area among available listings; it is a "
    "rule-based matching aid, not a financial recommendation."
)
