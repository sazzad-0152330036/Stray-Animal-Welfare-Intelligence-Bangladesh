import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA="data/v2_sari_scored.csv"
MODEL_RESULTS="results/model_comparison.csv"
CM_FILE="results/best_model_confusion_matrix.csv"
Path("visualization").mkdir(exist_ok=True)

df=pd.read_csv(DATA); models=pd.read_csv(MODEL_RESULTS); cm_df=pd.read_csv(CM_FILE,index_col=0)
NAVY="#17324D"; BLUE="#2878B5"; CYAN="#52B6D8"; TEAL="#2A9D8F"; GREEN="#55A868"
ORANGE="#F28E2B"; RED="#D9534F"; PURPLE="#8E6BBE"; PINK="#D66BA0"; GRAY="#7A869A"
plt.rcParams.update({"font.size":11,"axes.titlesize":15,"axes.titleweight":"bold","figure.facecolor":"white","axes.facecolor":"white"})

def clean(ax, axis="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis=axis,alpha=.18,linestyle="--"); ax.set_axisbelow(True)

# 1 SARI donut
order=["Low","Medium","High"]; risk=df["sari_v2_level"].value_counts().reindex(order,fill_value=0); total=risk.sum()
fig,ax=plt.subplots(figsize=(8,6))
wedges,_=ax.pie(risk.values,startangle=90,colors=[GREEN,ORANGE,RED],wedgeprops=dict(width=.38,edgecolor="white",linewidth=3))
ax.text(0,.08,f"{total}",ha="center",va="center",fontsize=25,fontweight="bold",color=NAVY)
ax.text(0,-.10,"articles",ha="center",va="center",color=GRAY)
ax.legend(wedges,[f"{n}: {c} ({c/total*100:.1f}%)" for n,c in risk.items()],title="SARI V2 Level",loc="center left",bbox_to_anchor=(.95,.5),frameon=False)
ax.set_title("SARI V2 Risk Profile — Bangladesh"); fig.tight_layout()
fig.savefig("visualization/final_sari_risk_donut.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 2 incident distribution
counts=df["ml_label"].value_counts().sort_values(); colors=[CYAN,BLUE,TEAL,PURPLE,ORANGE,RED,PINK,NAVY]
fig,ax=plt.subplots(figsize=(10,6.5)); bars=ax.barh(counts.index,counts.values,color=colors[:len(counts)])
ax.set_title("Incident-Type Distribution in the V2 ML Corpus"); ax.set_xlabel("Number of Articles"); ax.set_ylabel(""); clean(ax,"x")
for b,v in zip(bars,counts.values): ax.text(v+1,b.get_y()+b.get_height()/2,f"{v} ({v/counts.sum()*100:.1f}%)",va="center")
ax.set_xlim(0,max(counts.values)*1.27); fig.tight_layout()
fig.savefig("visualization/final_incident_distribution.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 3 model comparison
metrics=["Accuracy","Macro Precision","Macro Recall","Macro F1","Weighted F1"]; mc=[BLUE,TEAL,ORANGE,PURPLE,CYAN]
x=np.arange(len(models)); width=.15; fig,ax=plt.subplots(figsize=(12,6.5))
for i,(m,c) in enumerate(zip(metrics,mc)):
    vals=models[m].values; bars=ax.bar(x+(i-2)*width,vals,width,label=m,color=c)
    if m=="Macro F1":
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.018,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(models["Model"]); ax.set_ylim(0,1.02); ax.set_ylabel("Score")
ax.set_title("Machine-Learning Model Performance Comparison"); clean(ax)
ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.12)); fig.tight_layout()
fig.savefig("visualization/final_model_comparison.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 4 confusion matrix
cm=cm_df.values; labels=cm_df.index.tolist(); fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(cm,cmap="Blues")
ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels))); ax.set_xticklabels(labels,rotation=45,ha="right"); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("Reference Label"); ax.set_title("Linear SVM — Cross-Validated Confusion Matrix")
threshold=cm.max()/2 if cm.max() else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i,int(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>threshold else NAVY,fontweight="bold" if i==j else "normal")
fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Article Count"); fig.tight_layout()
fig.savefig("visualization/final_confusion_matrix.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 5 mean SARI
means=df.groupby("ml_label")["sari_v2_score"].mean().sort_values()
fig,ax=plt.subplots(figsize=(10,6)); cs=[GREEN if v<3 else ORANGE if v<6 else RED for v in means.values]
bars=ax.barh(means.index,means.values,color=cs); ax.axvline(3,ls="--",lw=1.5,color=ORANGE,label="Medium threshold (3)"); ax.axvline(6,ls="--",lw=1.5,color=RED,label="High threshold (6)")
ax.set_xlim(0,6.3); ax.set_xlabel("Mean SARI V2 Score"); ax.set_ylabel(""); ax.set_title("Average SARI V2 Score by Incident Type"); ax.legend(frameon=False); clean(ax,"x")
for b,v in zip(bars,means.values): ax.text(v+.06,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center")
fig.tight_layout(); fig.savefig("visualization/final_mean_sari_by_incident.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 6 yearly trend
if "year" in df.columns:
    t=df.copy(); t["year"]=pd.to_numeric(t["year"],errors="coerce"); yearly=t.dropna(subset=["year"]).groupby("year").size()
    if len(yearly):
        yearly.index=yearly.index.astype(int); fig,ax=plt.subplots(figsize=(11,5.5))
        ax.plot(yearly.index,yearly.values,marker="o",lw=2.5,color=BLUE); ax.fill_between(yearly.index,yearly.values,alpha=.12,color=CYAN)
        py,pv=yearly.idxmax(),yearly.max(); ax.scatter([py],[pv],s=90,color=RED,zorder=5)
        label = f"{py} partial: {pv} records" if py == 2026 else f"Peak: {pv} articles ({py})"
        ax.annotate(label,(py,pv),xytext=(12,-28),textcoords="offset points",fontweight="bold")        
        ax.set_title("Temporal Trend of Bangladesh Animal-Welfare News"); ax.set_xlabel("Year"); ax.set_ylabel("Number of Articles"); clean(ax)
        fig.tight_layout(); fig.savefig("visualization/final_yearly_trend.png",dpi=350,bbox_inches="tight"); plt.close(fig)

# 7 data refinement
stages=[("Expanded unique\nnews records",1094),("Animal-relevant\nrecords",690),("Bangladesh-primary\nrecords",250),("Final ML\ncorpus",len(df))]
names=[a for a,b in stages]; vals=[b for a,b in stages]; fc=[CYAN,BLUE,TEAL,NAVY]
fig,ax=plt.subplots(figsize=(10,6)); y=np.arange(len(stages)); bars=ax.barh(y,vals,color=fc); ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("Number of Articles"); ax.set_title("V2 Data Refinement Pipeline"); clean(ax,"x")
for i,(b,v) in enumerate(zip(bars,vals)):
    txt=f"{v}" if i==0 else f"{v} ({v/vals[0]*100:.1f}% of expanded data)"
    ax.text(v+15,b.get_y()+b.get_height()/2,txt,va="center",fontweight="bold",color=NAVY)
ax.set_xlim(0,max(vals)*1.25); fig.tight_layout()
fig.savefig("visualization/final_data_refinement_funnel.png",dpi=350,bbox_inches="tight"); plt.close(fig)

print("===== FINAL VISUALIZATION PACK COMPLETE =====")
for f in ["final_sari_risk_donut.png","final_incident_distribution.png","final_model_comparison.png","final_confusion_matrix.png","final_mean_sari_by_incident.png","final_yearly_trend.png","final_data_refinement_funnel.png"]:
    print("visualization/"+f)
print("\nNo ML score or SARI calculation was changed; only presentation was upgraded.")
