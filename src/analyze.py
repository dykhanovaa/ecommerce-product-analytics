from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'reports'
FIG = OUT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', palette='viridis')


def load_events():
    df = pd.read_csv(DATA / 'events.csv')
    df['event_time'] = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.tz_convert(None)
    df['transactionid'] = df.transactionid.astype('Int64')
    return df


def sessionize(df, gap='30min'):
    df = df.sort_values(['visitorid','timestamp']).copy()
    gap_ms = pd.Timedelta(gap).total_seconds() * 1000
    df['session_start'] = df.groupby('visitorid').timestamp.diff().gt(gap_ms).groupby(df.visitorid).cumsum()
    df['session_id'] = df.visitorid.astype(str) + '_' + df.session_start.astype(str)
    return df


def funnel(df):
    stages=(df[df.event.isin(['view','addtocart','transaction'])]
            .groupby(['session_id','event']).timestamp.min().unstack())
    stages=stages.reindex(columns=['view','addtocart','transaction'])
    f=stages.reset_index()
    f['visitorid']=f.session_id.str.rsplit('_',n=1).str[0].astype('int64')
    f['view']=stages['view'].notna().to_numpy()
    f['add_to_cart']=(stages['view'].notna() & stages['addtocart'].notna() & stages['addtocart'].ge(stages['view'])).to_numpy()
    f['purchase']=(stages['view'].notna() & stages['addtocart'].notna() & stages['transaction'].notna() & stages['addtocart'].ge(stages['view']) & stages['transaction'].ge(stages['addtocart'])).to_numpy()
    f=f[['session_id','visitorid','view','add_to_cart','purchase']]
    counts=f[['view','add_to_cart','purchase']].sum().astype(int)
    result=pd.DataFrame({'step':['View','Add to cart','Purchase'],'sessions':counts.values})
    result['conversion_from_previous']=result.sessions / result.sessions.shift(1)
    result.loc[0,'conversion_from_previous']=1.0
    result['conversion_from_view']=result.sessions / result.sessions.iloc[0]
    result.to_csv(OUT/'funnel.csv',index=False)
    return f,result


def category_map():
    cols=['timestamp','itemid','property','value']; pieces=[]
    for file in sorted(DATA.glob('item_properties_part*.csv')):
        for chunk in pd.read_csv(file, usecols=cols, chunksize=500_000):
            c=chunk[chunk.property.eq('categoryid')].dropna(subset=['value'])
            c['categoryid']=pd.to_numeric(c.value,errors='coerce')
            c=c.dropna(subset=['categoryid'])
            c['categoryid']=c.categoryid.astype('int64')
            pieces.append(c[['timestamp','itemid','categoryid']])
    all_categories=pd.concat(pieces,ignore_index=True)
    all_categories=all_categories.sort_values('timestamp').drop_duplicates('itemid',keep='last')
    return all_categories[['itemid','categoryid']]


def category_analysis(df, cats):
    events=df.merge(cats,on='itemid',how='left')
    v=events[events.event.eq('view')].groupby('categoryid').visitorid.nunique().rename('viewers')
    a=events[events.event.eq('addtocart')].groupby('categoryid').visitorid.nunique().rename('cart_users')
    p=events[events.event.eq('transaction')].groupby('categoryid').visitorid.nunique().rename('buyers')
    out=pd.concat([v,a,p],axis=1).fillna(0).astype(int).reset_index()
    out=out[out.viewers>=100].copy()
    out['view_to_cart']=out.cart_users/out.viewers
    out['view_to_purchase']=out.buyers/out.viewers
    out=out.sort_values('viewers',ascending=False)
    out.to_csv(OUT/'category_metrics.csv',index=False)
    return out


def repeat_and_time(df):
    tx=df[df.event.eq('transaction')].copy()
    user=tx.groupby('visitorid').agg(purchases=('transactionid','nunique'),first_purchase=('event_time','min'),last_purchase=('event_time','max')).reset_index()
    metrics=pd.DataFrame({'metric':['buyers','buyers_with_2plus_transactions','repeat_buyer_rate'],'value':[len(user),int((user.purchases>=2).sum()),float((user.purchases>=2).mean())]})
    metrics.to_csv(OUT/'repeat_purchase_metrics.csv',index=False)
    by_day=df.assign(date=df.event_time.dt.date).groupby(['date','event']).size().unstack(fill_value=0).reset_index()
    by_day.to_csv(OUT/'daily_activity.csv',index=False)
    return user,metrics,by_day


def charts(f, funnel_df, cats, by_day):
    fig,ax=plt.subplots(figsize=(8,4.5)); sns.barplot(data=funnel_df,x='step',y='sessions',ax=ax,color='#4676a9'); ax.set(title='Session funnel',xlabel='',ylabel='Sessions');
    for i,r in funnel_df.iterrows(): ax.text(i,r.sessions,f"{r.sessions:,.0f}\n({r.conversion_from_view:.1%} of views)",ha='center',va='bottom',fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'funnel.png',dpi=160); plt.close(fig)
    top=cats.sort_values('viewers',ascending=False).head(15).sort_values('view_to_purchase')
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(top.categoryid.astype(str),top.view_to_purchase,color='#3b8c79'); ax.set(title='View-to-purchase rate by category (top 15 by reach)',xlabel='Unique buyers / unique viewers',ylabel='Category ID'); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); fig.tight_layout(); fig.savefig(FIG/'category_conversion.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));
    for e in ['view','addtocart','transaction']:
        if e in by_day: ax.plot(pd.to_datetime(by_day.date),by_day[e],label=e,linewidth=1)
    ax.legend(); ax.set(title='Daily event activity',xlabel='',ylabel='Events'); fig.tight_layout(); fig.savefig(FIG/'daily_activity.png',dpi=160); plt.close(fig)


def main():
    df=load_events(); df=sessionize(df)
    f,fun=funnel(df)
    cats=category_analysis(df,category_map())
    users,repeat,days=repeat_and_time(df)
    charts(f,fun,cats,days)
    eligible=cats[cats.viewers>=100]
    table=np.array([eligible.buyers,eligible.viewers-eligible.buyers])
    chi2,p,_,_=chi2_contingency(table)
    summary={'events':int(len(df)),'visitors':int(df.visitorid.nunique()),'sessions':int(f.session_id.nunique()),'funnel_sessions':fun.sessions.tolist(),'view_to_cart':float(fun.loc[1,'conversion_from_previous']),'cart_to_purchase':float(fun.loc[2,'conversion_from_previous']),'view_to_purchase':float(fun.loc[2,'conversion_from_view']),'repeat_buyer_rate':float(repeat.loc[2,'value']),'category_chi2':float(chi2),'category_chi2_pvalue':float(p)}
    import json
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
