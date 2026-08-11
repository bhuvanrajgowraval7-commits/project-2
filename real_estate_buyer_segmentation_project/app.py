
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Real Estate Buyer Intelligence", layout="wide")
ROOT=Path(__file__).parent
clients_path=ROOT/"outputs/client_segment_assignments.csv"
profiles_path=ROOT/"outputs/segment_profiles.csv"
geo_path=ROOT/"outputs/geographic_segment_summary.csv"
inv_path=ROOT/"outputs/investment_summary.csv"
rec_path=ROOT/"outputs/property_recommendations_by_segment.csv"

@st.cache_data
def load():
    return (pd.read_csv(clients_path), pd.read_csv(profiles_path),
            pd.read_csv(geo_path), pd.read_csv(inv_path), pd.read_csv(rec_path))

c,prof,geo,inv,recs=load()

# Compatibility: derive helper flags if they are not stored in the CSV.
if "is_loan" not in c.columns:
    c["is_loan"] = (
        c["loan_applied"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0}).fillna(0)
    )
if "is_investment" not in c.columns:
    c["is_investment"] = (
        c["acquisition_purpose"].astype(str).str.strip().str.lower()
        .map({"investment": 1, "home": 0}).fillna(0)
    )

st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Machine-learning buyer intelligence dashboard based on the supplied client and property datasets.")

with st.sidebar:
    st.header("Filters")
    country=st.multiselect("Country",sorted(c.country.dropna().unique()),default=[])
    region=st.multiselect("Region",sorted(c.region.dropna().unique()),default=[])
    purpose=st.multiselect("Acquisition purpose",sorted(c.acquisition_purpose.dropna().unique()),default=[])
    ctype=st.multiselect("Client type",sorted(c.client_type.dropna().unique()),default=[])
    segment=st.multiselect("Buyer segment",sorted(c.segment.dropna().unique()),default=[])

f=c.copy()
if country: f=f[f.country.isin(country)]
if region: f=f[f.region.isin(region)]
if purpose: f=f[f.acquisition_purpose.isin(purpose)]
if ctype: f=f[f.client_type.isin(ctype)]
if segment: f=f[f.segment.isin(segment)]

m1,m2,m3,m4=st.columns(4)
m1.metric("Buyers",f.client_id.nunique())
m2.metric("Total investment",f["total_investment"].sum())
m3.metric("Avg purchase price",f["avg_purchase_price"].mean())
m4.metric("Avg satisfaction",round(f["satisfaction_score"].mean(),2))

st.subheader("Buyer Segmentation Overview")
dist=f.groupby("segment").size().reset_index(name="buyers").sort_values("buyers",ascending=False)
st.bar_chart(dist.set_index("segment"))

st.subheader("Investor Behavior Dashboard")
a,b=st.columns(2)
with a:
    behavior=f.groupby("segment").agg(
        buyers=("client_id","count"), avg_purchases=("purchases","mean"),
        avg_total_investment=("total_investment","mean"),
        avg_purchase_price=("avg_purchase_price","mean"),
        investment_share=("is_investment","mean"), loan_share=("is_loan","mean")
    ).round(2)
    st.dataframe(behavior,use_container_width=True)
with b:
    st.bar_chart(f.groupby(["segment","acquisition_purpose"]).size().unstack(fill_value=0))

st.subheader("Geographic Buyer Analysis")

g=f.groupby(["country","region","segment"]).size().reset_index(name="buyers")

# Representative region coordinates for an interactive visualization.
# These are regional visualization points, not property addresses.
REGION_COORDS = {
    "California":(36.7783,-119.4179),"Nevada":(38.8026,-116.4194),
    "Colorado":(39.5501,-105.7821),"Arizona":(34.0489,-111.0937),
    "Oregon":(43.8041,-120.5542),"Utah":(39.3210,-111.0937),
    "Washington":(47.4009,-121.4905),"Virginia":(37.4316,-78.6569),
    "Texas":(31.9686,-99.9018),"Florida":(27.6648,-81.5158),
    "New York":(42.1657,-74.9481),"Georgia":(32.1656,-82.9001),
    "Ohio":(40.4173,-82.9071),"Kansas":(39.0119,-98.4842),
    "Wyoming":(43.076,-107.29),"England":(52.3555,-1.1743),
    "Scotland":(56.4907,-4.2026),"Wales":(52.1307,-3.7837),
    "Northern Ireland":(54.7877,-6.4923),"Alberta":(53.9333,-116.5765),
    "British Columbia":(53.7267,-127.6476),"Manitoba":(53.7609,-98.8139),
    "Ontario":(51.2538,-85.3232),"Quebec":(52.9399,-73.5491),
    "Baja California":(30.8406,-115.2838),"Jalisco":(20.6597,-103.3496),
    "Mexico City":(19.4326,-99.1332),"Nuevo Leon":(25.5922,-99.9962),
    "Puebla":(19.0414,-98.2063),"Bavaria":(48.7904,11.4979),
    "Berlin":(52.52,13.405),"Hamburg":(53.5511,9.9937),
    "Hesse":(50.6521,9.1624),"North Rhine-Westphalia":(51.4332,7.6616),
    "Brussels":(50.8503,4.3517),"Flanders":(51.0543,3.7174),
    "Wallonia":(50.5039,4.4699),"Brittany":(48.202,-2.9326),
    "Ile-de-France":(48.8499,2.637),"Normandy":(49.1829,0.3707),
    "Occitanie":(43.8927,3.2828),"Provence":(43.9352,6.0679),
    "Capital Region":(55.6761,12.5683),"Central Denmark":(56.3021,9.302),
    "North Denmark":(57.118,9.8492),"Southern Denmark":(55.4,10.4),
    "Zealand":(55.4038,11.3517),"Moscow Oblast":(55.1436,38.8466),
    "Krasnodar Krai":(45.6415,39.7056),"Novosibirsk":(55.0084,82.9357),
    "Saint Petersburg":(59.9311,30.3609),"Tatarstan":(55.7879,49.1233),
    "New South Wales":(-31.2532,146.9211),"Queensland":(-20.9176,142.7028),
    "South Australia":(-30.0002,136.2092),"Victoria":(-36.9848,143.3906),
    "Western Australia":(-25.0423,121.7310)
}

# Dominant segment per region, plus total regional buyer count.
idx=g.groupby(["country","region"])["buyers"].idxmax()
gm=g.loc[idx].copy()
tot=g.groupby(["country","region"])["buyers"].sum().reset_index(name="total_buyers")
gm=gm.merge(tot,on=["country","region"],how="left")
gm["latitude"]=gm["region"].map(lambda x: REGION_COORDS.get(x,(np.nan,np.nan))[0])
gm["longitude"]=gm["region"].map(lambda x: REGION_COORDS.get(x,(np.nan,np.nan))[1])
gm=gm.dropna(subset=["latitude","longitude"])

if not gm.empty:
    fig=px.scatter_geo(
        gm,lat="latitude",lon="longitude",size="total_buyers",color="segment",
        hover_name="region",
        hover_data={"country":True,"total_buyers":True,"segment":True,
                    "buyers":True,"latitude":False,"longitude":False},
        scope="world",projection="natural earth",
        title="Buyer Segments by Geographic Region",size_max=35
    )
    fig.update_geos(showland=True,showcountries=True)
    st.plotly_chart(fig,use_container_width=True)

st.dataframe(g.sort_values("buyers",ascending=False).head(50),use_container_width=True)

st.subheader("Segment Insights")
for _,r in prof.iterrows():
    st.markdown(f"**{r.segment}** — {int(r.clients):,} buyers | Avg age {r.avg_age:.1f} | "
                f"Avg purchases {r.avg_purchases:.1f} | Avg purchase price ${r.avg_purchase_price:,.0f} | "
                f"Avg total investment ${r.avg_total_investment:,.0f} | Loan share {r.loan_share:.0%} | "
                f"Investment-purpose share {r.investment_share:.0%}")

st.subheader("Property Recommendations by Segment")
if segment:
    rr=recs[recs.segment.isin(segment)]
else:
    rr=recs
st.dataframe(rr.sort_values(["segment","match_score"]).head(100),use_container_width=True)
st.caption("Recommendation score is a distance to the segment's observed average purchase price and average area among available listings; it is a rule-based matching aid, not a financial recommendation.")
