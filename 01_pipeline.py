import json
import os
import datetime
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

MAPA_SETORES_B3 = {
   'EMBJ3': 'Bens Industriais', 'POMO4': 'Bens Industriais', 'WEGE3': 'Bens Industriais',
   'FRAS3': 'Bens Industriais', 'RAPT4': 'Bens Industriais', 'TUPY3': 'Bens Industriais',
   'KEPL3': 'Bens Industriais', 'SHUL4': 'Bens Industriais', 'RAIL3': 'Bens Industriais',
   'DIRR3': 'Construção Civil', 'LAVV3': 'Construção Civil', 'TECN3': 'Construção Civil',
   'CYRE3': 'Construção Civil', 'EZTC3': 'Construção Civil', 'CURY3': 'Construção Civil',
   'PLPL3': 'Construção Civil', 'TRIS3': 'Construção Civil', 'JHSF3': 'Construção Civil',
   'CMIG4': 'Utilidade Pública', 'SBSP3': 'Utilidade Pública', 'EGIE3': 'Utilidade Pública',
   'CPLE6': 'Utilidade Pública', 'EQTL3': 'Utilidade Pública', 'TAEE11': 'Utilidade Pública',
   'SAPR11': 'Utilidade Pública', 'CSMG3': 'Utilidade Pública', 'NEOE3': 'Utilidade Pública',
   'GMAT3': 'Consumo', 'RIAA3': 'Consumo', 'MDIA3': 'Consumo', 'ABEV3': 'Consumo',
   'ARZZ3': 'Consumo', 'SMTO3': 'Consumo', 'SLCE3': 'Consumo', 'CAML3': 'Consumo',
   'ASAI3': 'Consumo', 'CRFB3': 'Consumo', 'ALPA4': 'Consumo', 'VULC3': 'Consumo',
   'TTEN3': 'Consumo', 'SBFG3': 'Consumo', 'TFCO4': 'Consumo', 'CSED3': 'Consumo',
   'AURA33': 'Materiais Básicos', 'VALE3': 'Materiais Básicos', 'GGBR4': 'Materiais Básicos',
   'CSNA3': 'Materiais Básicos', 'SUZB3': 'Materiais Básicos', 'KLBN11': 'Materiais Básicos',
   'UNIP6': 'Materiais Básicos', 'FESA4': 'Materiais Básicos', 'EUCA4': 'Materiais Básicos',
   'PSSA3': 'Financeiro', 'BBAS3': 'Financeiro', 'ITUB4': 'Financeiro', 'BBDC4': 'Financeiro',
   'SANB11': 'Financeiro', 'CXSE3': 'Financeiro', 'BBSE3': 'Financeiro', 'B3SA3': 'Financeiro',
   'PETR4': 'Petróleo e Gás', 'PRIO3': 'Petróleo e Gás', 'RECV3': 'Petróleo e Gás',
   'UGPA3': 'Petróleo e Gás', 'VBBR3': 'Petróleo e Gás', 'BRAV3': 'Petróleo e Gás',
   'RADL3': 'Saúde', 'FLRY3': 'Saúde', 'HYPE3': 'Saúde', 'RDOR3': 'Saúde', 'PNVL3': 'Saúde',
   'TOTS3': 'Tecnologia', 'INTB3': 'Tecnologia', 'LWSA3': 'Tecnologia'
}

PARAMS = {
   "min_roe": 10.0, "max_div_patrimonio": 1.0, "min_liquidez": 500_000,
   "min_cresc_5a": 5.0, "min_margem_liq": 0.0, "n_top_assets": 12,
   "max_ativos_por_setor": 2, "janela_momentum": 126, "state_file": "./portfolio_state.json",
   "taxa_selic_ano": 0.1075
}

def carregar_dados_silver():
   folder_path = "../datasets/silver"
   latest_file = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])[-1]
   data = []
   with open(os.path.join(folder_path, latest_file), 'r', encoding='utf-8') as f:
       for line in f: data.append(json.loads(line))
   df = pd.DataFrame(data)
   df['ticker_puro'] = df['ticker'].astype(str).str.replace(".SA", "", regex=False).str.strip()
   df['setor'] = df['ticker_puro'].map(MAPA_SETORES_B3).fillna('Outros')
  
   df_filtered = df[
       (df['roe'] >= PARAMS["min_roe"]) &
       (df['div_liq_patrim'] <= PARAMS["max_div_patrimonio"]) &
       (df['liq_2meses'] >= PARAMS["min_liquidez"]) &
       (df['cresc_rec_5a'] >= PARAMS["min_cresc_5a"]) &
       (df['mrg_liq'] > PARAMS["min_margem_liq"])
   ].copy()
   df_filtered['radical'] = df_filtered['ticker_puro'].str[:4]
   return df_filtered.sort_values('liq_2meses', ascending=False).drop_duplicates('radical')

def selecionar_top_momentum(returns_window, df_silver):
   mapa_ticker_setor = dict(zip(df_silver['ticker_puro'].apply(lambda x: x + ".SA"), df_silver['setor']))
   valid_tickers = [t for t in returns_window.columns if t in mapa_ticker_setor]
   ret_sub = returns_window[valid_tickers]
  
   w_mom = min(PARAMS["janela_momentum"], len(ret_sub) - 1)
   ret_acumulado = (1 + ret_sub.iloc[-w_mom:]).prod() - 1
   vol_diaria = ret_sub.iloc[-w_mom:].std()
   score_momentum = ret_acumulado / (vol_diaria + 1e-6)
  
   df_scores = pd.DataFrame({
       'score': score_momentum,
       'setor': [mapa_ticker_setor.get(t, 'Outros') for t in score_momentum.index]
   }).sort_values(by='score', ascending=False)
  
   selecionados, contagem_setores = [], {}
   for ticker, row in df_scores.iterrows():
       setor = row['setor']
       if contagem_setores.get(setor, 0) < PARAMS["max_ativos_por_setor"]:
           selecionados.append(ticker)
           contagem_setores[setor] = contagem_setores.get(setor, 0) + 1
           if len(selecionados) == PARAMS["n_top_assets"]:
               break
              
   if len(selecionados) < PARAMS["n_top_assets"]:
       restantes = [t for t in df_scores.index if t not in selecionados]
       selecionados.extend(restantes[:(PARAMS["n_top_assets"] - len(selecionados))])
   return selecionados, df_scores

def executar_pipeline():
   df_silver = carregar_dados_silver()
   csv_price = pd.read_csv("../datasets/snapshots_dados/historico_precos_tela.csv", index_col=0, parse_dates=True)
   tickers_validos = [t + ".SA" for t in df_silver['ticker_puro'].tolist() if (t + ".SA") in csv_price.columns]
   returns_full = csv_price[tickers_validos].ffill().bfill().pct_change().dropna()
  
   top_ativos, df_scores = selecionar_top_momentum(returns_full, df_silver)
   ret_treino_top = returns_full[top_ativos].dropna()
  
   port_hrp = rp.HCPortfolio(returns=ret_treino_top)
   w_hrp = port_hrp.optimization(model='HRP', codependence='pearson', rm='MV', linkage='ward')
   pesos = dict(zip(w_hrp.index, w_hrp['weights']))
  
   port_mkv = rp.Portfolio(returns=ret_treino_top)
   port_mkv.assets_stats(method_mu='hist', method_cov='hist')
   w_mkv = port_mkv.optimization(model='Classic', rm='MV', obj='Sharpe', rf=PARAMS["taxa_selic_ano"]/252, hist=True)
  
   os.makedirs("./temp_charts", exist_ok=True)
   with PdfPages('relatorio_graficos_hrp.pdf') as pdf:

       df_comparativo = pd.DataFrame({'HRP': w_hrp['weights'] * 100, 'Markowitz (Max Sharpe)': w_mkv['weights'] * 100}).sort_values(by='HRP', ascending=False)
       ax = df_comparativo.plot(kind='bar', figsize=(14, 6), color=['#1F4E78', '#C00000'], width=0.8)
       plt.title("Distribuição de Pesos Alocados: HRP vs. Markowitz", fontsize=14, fontweight='bold')
       for p in ax.patches:
           if p.get_height() > 0.5: ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=9)
       plt.tight_layout(); plt.savefig('./temp_charts/04_distribuicao_pesos.png', dpi=300); pdf.savefig(); plt.close()

       corr_matrix = ret_treino_top.corr()
       dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
       fig, axes = plt.subplots(1, 2, figsize=(16, 6))
       sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
       axes[0].set_title("Matriz de Correlação Linear")
       sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="viridis_r", ax=axes[1])
       axes[1].set_title("Matriz de Distância Angular")
       plt.tight_layout(); plt.savefig('./temp_charts/01_corr_dist.png', dpi=300); pdf.savefig(); plt.close()

       dist_array = ssd.squareform(dist_matrix)
       linkage_matrix = sch.linkage(dist_array, method='ward')
       plt.figure(figsize=(14, 6))
       sch.dendrogram(linkage_matrix, labels=corr_matrix.columns, leaf_rotation=45)
       plt.title("Dendrograma de Agrupamento Hierárquico")
       plt.tight_layout(); plt.savefig('./temp_charts/02_dendrograma.png', dpi=300); pdf.savefig(); plt.close()

   state = {"data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ativos": {}}
   for t, w in pesos.items():
       s = df_scores.loc[t, 'setor']
       state["ativos"][t] = {"setor": s, "peso_alvo": float(w), "status": "EM_CARTEIRA", "peso_acao": float(w), "peso_cdi": 0.0, "dias_saida": 0, "dias_reentrada": 0}
      
   with open(PARAMS["state_file"], 'w', encoding='utf-8') as f:
       json.dump(state, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
   executar_pipeline()
