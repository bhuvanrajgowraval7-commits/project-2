
import argparse, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def parse_money(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]","",regex=True), errors="coerce")

def build_features(clients, properties):
    clients=clients.copy(); properties=properties.copy()
    today=pd.Timestamp("2026-08-11")
    clients["date_of_birth_parsed"]=pd.to_datetime(clients["date_of_birth"],errors="coerce",dayfirst=True)
    clients["age"]=((today-clients["date_of_birth_parsed"]).dt.days/365.2425).round(1)
    bad=(clients["age"]<18)|(clients["age"]>100)|clients["age"].isna()
    if bad.any():
        alt=pd.to_datetime(clients.loc[bad,"date_of_birth"],errors="coerce",dayfirst=False)
        clients.loc[bad,"date_of_birth_parsed"]=alt
        clients.loc[bad,"age"]=((today-alt).dt.days/365.2425).round(1)
    properties["transaction_date_parsed"]=pd.to_datetime(properties["transaction_date"],errors="coerce",dayfirst=True)
    properties["sale_price_num"]=parse_money(properties["sale_price"])
    properties["price_per_sqft"]=properties["sale_price_num"]/properties["floor_area_sqft"]
    joined=properties.merge(clients[["client_id"]],left_on="client_ref",right_on="client_id",how="left")
    agg=joined.groupby("client_id",dropna=False).agg(
        purchases=("listing_id","count"), total_investment=("sale_price_num","sum"),
        avg_purchase_price=("sale_price_num","mean"), avg_area_sqft=("floor_area_sqft","mean"),
        avg_price_per_sqft=("price_per_sqft","mean"), towers=("tower_number","nunique")
    ).reset_index()
    c=clients.merge(agg,on="client_id",how="left")
    for col in ["purchases","total_investment","avg_purchase_price","avg_area_sqft","avg_price_per_sqft","towers"]:
        c[col]=c[col].fillna(0)
    c["is_loan"]=c["loan_applied"].map({"Yes":1,"No":0}).fillna(0)
    c["is_investment"]=c["acquisition_purpose"].map({"Investment":1,"Home":0}).fillna(0)
    return c, properties

def train(c):
    num=["age","satisfaction_score","purchases","total_investment","avg_purchase_price","avg_area_sqft","avg_price_per_sqft"]
    cat=["client_type","acquisition_purpose","loan_applied"]
    pre=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])
    X=pre.fit_transform(c[num+cat])
    scores=[]
    for k in range(2,9):
        lab=KMeans(n_clusters=k,random_state=42,n_init=20).fit_predict(X)
        scores.append((k,float(silhouette_score(X,lab))))
    best_k=max(scores,key=lambda x:x[1])[0]
    model=KMeans(n_clusters=best_k,random_state=42,n_init=20).fit(X)
    c["cluster"]=model.labels_
    prof=c.groupby("cluster").agg(
        clients=("client_id","count"), avg_age=("age","mean"), avg_satisfaction=("satisfaction_score","mean"),
        avg_purchases=("purchases","mean"), avg_total_investment=("total_investment","mean"),
        avg_purchase_price=("avg_purchase_price","mean"), avg_area_sqft=("avg_area_sqft","mean"),
        avg_ppsqft=("avg_price_per_sqft","mean"), investment_share=("is_investment","mean"),
        loan_share=("is_loan","mean"), company_share=("client_type",lambda s:(s=="Company").mean())
    ).reset_index()
    labels={}
    labels[prof.set_index("cluster")["avg_purchases"].idxmax()]="Frequent / Portfolio Buyers"
    labels[prof.set_index("cluster")["avg_purchase_price"].idxmax()]="Premium Home Buyers"
    labels[prof.set_index("cluster")["avg_purchase_price"].idxmin()]="Value-Oriented Buyers"
    if len(set(labels.values()))<best_k:
        labels={i:f"Segment {i+1}" for i in range(best_k)}
    c["segment"]=c["cluster"].map(labels); prof["segment"]=prof["cluster"].map(labels)
    return c, prof, pre, model, scores, num+cat

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clients",required=True); ap.add_argument("--properties",required=True); ap.add_argument("--out",default="outputs")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    clients=pd.read_csv(args.clients); properties=pd.read_csv(args.properties)
    c,p=build_features(clients,properties)
    c,prof,pre,model,scores,features=train(c)
    c.to_csv(out/"client_segment_assignments.csv",index=False)
    prof.to_csv(out/"segment_profiles.csv",index=False)
    pd.DataFrame(scores,columns=["k","silhouette_score"]).to_csv(out/"cluster_evaluation.csv",index=False)
    joblib.dump({"preprocessor":pre,"model":model,"feature_columns":features,"segment_labels":dict(prof[["cluster","segment"]].values)},out/"buyer_segmentation_model.joblib")
    print(prof.round(2).to_string(index=False))

if __name__=="__main__":
    main()
