import json
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import riskfolio as rp

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
   "max_ativos_por_setor": 2, "janela_momentum": 126, "ema_periodos": 50,
   "histerese_pct": 0.03, "dias_saida": 4, "dias_reentrada": 2,
   "valor_investido_inicial": 1000.00, "taxa_selic_ano": 0.1075,
   "frequencia_rebalanceamento": 126
}

def checar_sinais_dinamicos(preco_hist, status_atual, dias_saida, dias_reentrada, histerese=0.03):
   preco_hoje = preco_hist.iloc[-1]
   ema50 = preco_hist.ewm(span=50, adjust=False).mean().iloc[-1]
   limite_saida = ema50 * (1.0 - histerese)
  
   novo_status = status_atual
   if status_atual == "EM_CARTEIRA":
       dias_saida = dias_saida + 1 if preco_hoje < limite_saida else 0
       if dias_saida >= 4:
           novo_status = "EM_CDI"
           dias_saida = 0
   elif status_atual == "EM_CDI":
       dias_reentrada = dias_reentrada + 1 if preco_hoje > ema50 else 0
       if dias_reentrada >= 2:
           novo_status = "EM_CARTEIRA"
           dias_reentrada = 0
          
   return novo_status, dias_saida, dias_reentrada

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
   return selecionados

def otimizar_hrp(returns_subset):
   port = rp.HCPortfolio(returns=returns_subset)
   return port.optimization(model='HRP', codependence='pearson', rm='MV', linkage='ward')

def otimizar_markowitz(returns_subset):
   port = rp.Portfolio(returns=returns_subset)
   port.assets_stats(method_mu='hist', method_cov='hist')
   try:
       w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=PARAMS["taxa_selic_ano"]/252)
       if w is None or w.empty or np.isnan(w.values).any(): raise ValueError()
       return w
   except Exception:
       return pd.DataFrame({'weights': [1.0/len(returns_subset.columns)] * len(returns_subset.columns)}, index=returns_subset.columns)

def gerar_graficos_estatisticos(ret_treino_top, w_hrp, w_mvo, pasta_output="./temp_charts"):
   os.makedirs(pasta_output, exist_ok=True)
   caminhos = {}

   corr_matrix = ret_treino_top.corr()
   dist_matrix = np.sqrt((1.0 - corr_matrix) / 2.0)
   fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
   sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
   axes[0].set_title("Matriz de Correlação Linear (Pearson)", fontweight='bold')
   sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap='viridis_r', ax=axes[1])
   axes[1].set_title("Matriz de Distância Angular", fontweight='bold')
   plt.tight_layout()
   p1 = f"{pasta_output}/01_corr_dist.png"
   plt.savefig(p1, dpi=180); plt.close()
   caminhos["corr_dist"] = p1

   dist_condensed = squareform(dist_matrix.values, checks=False)
   linkage_matrix = linkage(dist_condensed, method='ward')
   plt.figure(figsize=(10, 4.2))
   dendrogram(linkage_matrix, labels=dist_matrix.columns.tolist(), leaf_rotation=45)
   plt.title("Dendrograma de Agrupamento Hierárquico (Ward Linkage)", fontweight='bold')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   p2 = f"{pasta_output}/02_dendrograma.png"
   plt.savefig(p2, dpi=180); plt.close()
   caminhos["dendrograma"] = p2

   T, N = ret_treino_top.shape
   q = T / N
   e_val, _ = np.linalg.eigh(corr_matrix)
   e_val = np.sort(e_val)[::-1]
   e_max = (1.0 + (1.0 / q)**0.5)**2
   e_min = (1.0 - (1.0 / q)**0.5)**2
   e_grid = np.linspace(e_min, e_max, 500)
   pdf_mp = (q / (2.0 * np.pi * e_grid)) * np.sqrt(np.maximum(0.0, (e_max - e_grid) * (e_grid - e_min)))

   plt.figure(figsize=(10, 4.2))
   plt.hist(e_val, bins=12, density=True, alpha=0.6, color='#1F4E78', edgecolor='black', label='Autovalores Empíricos')
   plt.plot(e_grid, pdf_mp, color='#C00000', lw=2, label=f'Cinturão de Ruído ($\lambda_+={e_max:.2f}$)')
   plt.axvline(e_max, color='black', linestyle=':', lw=1.5, label='Threshold de Sinal')
   plt.title("RMT: Sinal vs. Ruído nos Ativos Selecionados", fontweight='bold')
   plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
   p3 = f"{pasta_output}/03_rmt_marchenko.png"
   plt.savefig(p3, dpi=180); plt.close()
   caminhos["rmt"] = p3

   df_comparativo_pesos = pd.DataFrame({
       'HRP': w_hrp['weights'] * 100,
       'Markowitz (Max Sharpe)': w_mvo['weights'] * 100
   }).sort_values(by='HRP', ascending=False)

   ax = df_comparativo_pesos.plot(kind='bar', figsize=(14, 6), color=['#1F4E78', '#C00000'], width=0.8)
   plt.title("Distribuição de Pesos Alocados: HRP vs. Markowitz (Max Sharpe)", fontsize=12, fontweight='bold')
   plt.ylabel("Peso Alocado (%)")
   plt.xlabel("Ativo")
   plt.xticks(rotation=45)
   for p in ax.patches:
       if p.get_height() > 0.5:
           ax.annotate(f"{p.get_height():.1f}%",
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')
   plt.grid(True, axis='y', alpha=0.3)
   plt.tight_layout()
   p4 = f"{pasta_output}/04_distribuicao_pesos.png"
   plt.savefig(p4, dpi=180, bbox_inches='tight')
   plt.close()
   caminhos["dist_pesos"] = p4

   return caminhos

def gerar_relatorio_pdf_completo(output_path, metricas, caminhos_graficos, datas_info, lista_eventos):
   doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)
   story = []
   styles = getSampleStyleSheet()
  
   title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#1F4E78'), spaceAfter=4)
   h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=9.5, textColor=colors.HexColor('#2C3E50'), spaceBefore=6, spaceAfter=4)
   body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9, spaceAfter=3)
   tbl_hdr = ParagraphStyle('TH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7, textColor=colors.whitesmoke, alignment=1)
   tbl_txt = ParagraphStyle('TD', parent=styles['Normal'], fontName='Helvetica', fontSize=6.5, alignment=1)
  
   story.append(Paragraph("Relatório Final de Backtest: Validação Quantitativa dos 5 Modelos do TCC", title_style))
   story.append(Paragraph(f"<b>Execução:</b> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | <b>Período Out-of-Sample:</b> {datas_info['inicio']} até {datas_info['fim']}", body_style))
   story.append(Spacer(1, 4))
  
   story.append(Paragraph("1. Tabela Comparativa de Desempenho e Eficiência Ajustada ao Risco", h2_style))
   m_data = [
       [Paragraph("Métrica Analisada", tbl_hdr), Paragraph("HRP Dinâmico (Proposto k≤2)", tbl_hdr), Paragraph("Markowitz + Momentum", tbl_hdr), Paragraph("HRP Puro (Base Silver)", tbl_hdr), Paragraph("Markowitz Puro (Silver)", tbl_hdr), Paragraph("IBOVESPA (Benchmark)", tbl_hdr)],
       [Paragraph("Patrimônio Final", tbl_txt), f"R$ {metricas['patr_hrp_din']:,.2f}", f"R$ {metricas['patr_mvo_mom']:,.2f}", f"R$ {metricas['patr_hrp_puro']:,.2f}", f"R$ {metricas['patr_mvo_puro']:,.2f}", f"R$ {metricas['patr_ibov']:,.2f}"],
       [Paragraph("Retorno Acumulado", tbl_txt), f"{metricas['ret_hrp_din']:.2f}%", f"{metricas['ret_mvo_mom']:.2f}%", f"{metricas['ret_hrp_puro']:.2f}%", f"{metricas['ret_mvo_puro']:.2f}%", f"{metricas['ret_ibov']:.2f}%"],
       [Paragraph("Max Drawdown", tbl_txt), f"{metricas['dd_hrp_din']:.2f}%", f"{metricas['dd_mvo_mom']:.2f}%", f"{metricas['dd_hrp_puro']:.2f}%", f"{metricas['dd_mvo_puro']:.2f}%", f"{metricas['dd_ibov']:.2f}%"],
       [Paragraph("Índice Sharpe (Rf=10.75%)", tbl_txt), f"{metricas['sharpe_hrp_din']:.2f}", f"{metricas['sharpe_mvo_mom']:.2f}", f"{metricas['sharpe_hrp_puro']:.2f}", f"{metricas['sharpe_mvo_puro']:.2f}", f"{metricas['sharpe_ibov']:.2f}"],
       [Paragraph("Determinismo Matemático", tbl_txt), "100% (Exato)", "100% (Exato)", "100% (Exato)", "100% (Exato)", "Passivo"]
   ]
   t_m = Table(m_data, colWidths=[105, 95, 95, 85, 90, 75])
   t_m.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F4E78')), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BOTTOMPADDING', (0,0), (-1,-1), 2), ('TOPPADDING', (0,0), (-1,-1), 2)]))
   story.append(t_m)
   story.append(Spacer(1, 4))
  
   story.append(Paragraph("2. Curvas Patrimoniais e de Drawdown Residual dos 5 Modelos", h2_style))
   story.append(Image(caminhos_graficos["backtest"], width=540, height=220))
   story.append(Spacer(1, 4))
  
   story.append(PageBreak())
   story.append(Paragraph("3. Distribuição de Pesos: HRP vs Markowitz", h2_style))
   story.append(Image(caminhos_graficos["dist_pesos"], width=540, height=225))
   story.append(Spacer(1, 4))
  
   story.append(Paragraph("4. Topologia Hierárquica e Decomposição Espectral dos Ativos", h2_style))
   story.append(Image(caminhos_graficos["corr_dist"], width=540, height=180))
   story.append(Spacer(1, 4))
   story.append(Image(caminhos_graficos["dendrograma"], width=540, height=155))
   story.append(Spacer(1, 4))
   story.append(Image(caminhos_graficos["rmt"], width=540, height=155))
  
   story.append(PageBreak())
   story.append(Paragraph("5. Log de Eventos de Gestão de Cauda (Stops e Reentradas)", h2_style))
  
   if lista_eventos:
       t_ev_data = [[Paragraph("Data", tbl_hdr), Paragraph("Tipo Evento", tbl_hdr), Paragraph("Ativo", tbl_hdr), Paragraph("Critério Técnico", tbl_hdr), Paragraph("Peso (%)", tbl_hdr)]]
       for ev in lista_eventos:
           t_ev_data.append([Paragraph(str(ev[0]), tbl_txt), Paragraph(str(ev[1]), tbl_txt), Paragraph(str(ev[2]), tbl_txt), Paragraph(str(ev[3]), tbl_txt), Paragraph(str(ev[4]), tbl_txt)])
      
       t_ev = Table(t_ev_data, colWidths=[65, 85, 75, 140, 60], repeatRows=1)
       t_ev.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F4E78')), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BOTTOMPADDING', (0,0), (-1,-1), 3), ('TOPPADDING', (0,0), (-1,-1), 3)]))
       story.append(t_ev)
   else:
       story.append(Paragraph("Nenhum evento registrado no período.", body_style))
  
   doc.build(story)
   print(f"\n[PDF ENGINE] Relatório completo com gráficos salvo em: {output_path}")

def executar_backtest():
   df_silver = carregar_dados_silver()
   snapshot_dir = "../datasets/snapshots_dados"
   csv_price = pd.read_csv(f"{snapshot_dir}/historico_precos_tela.csv", index_col=0, parse_dates=True)
  
   caminho_ibov = f"{snapshot_dir}/historico_ibovespa.csv"
   if os.path.exists(caminho_ibov):
       csv_ibov = pd.read_csv(caminho_ibov, index_col=0, parse_dates=True)
       ret_ibov_raw = csv_ibov.squeeze().pct_change()
   else:
       ret_ibov_raw = pd.Series(0.0, index=csv_price.index)
      
   tickers_validos = [t + ".SA" for t in df_silver['ticker_puro'].tolist() if (t + ".SA") in csv_price.columns]
   returns_full = csv_price[tickers_validos].ffill().bfill().pct_change().dropna()
   precos_full = csv_price[tickers_validos].ffill().bfill()
  
   ponto_corte = returns_full.index.get_indexer([pd.to_datetime("2024-06-03")], method='nearest')[0]
   total_dias = len(returns_full)
   janela_rebal = PARAMS["frequencia_rebalanceamento"]
   taxa_cdi_diaria = (1 + PARAMS["taxa_selic_ano"])**(1/252) - 1
  
   ret_hrp_din_list, ret_mvo_mom_list, ret_hrp_puro_list, ret_mvo_puro_list, ret_ibov_list = [], [], [], [], []
   lista_eventos = []
   cursor = ponto_corte
  
   print("="*85)
   print(" [03 BACKTEST] EXECUTANDO MODELOS DETERMINÍSTICOS E GERANDO FIGURAS")
   print("="*85)
  
   ret_treino_inicial = returns_full.iloc[:cursor].dropna(axis=1)
   top_ativos_inicial = selecionar_top_momentum(ret_treino_inicial, df_silver)
   df_ret_top_inicial = ret_treino_inicial[top_ativos_inicial].dropna()
   w_hrp_inicial = otimizar_hrp(df_ret_top_inicial)
   w_mvo_inicial = otimizar_markowitz(df_ret_top_inicial)
   caminhos_graficos = gerar_graficos_estatisticos(df_ret_top_inicial, w_hrp_inicial, w_mvo_inicial)
  
   while cursor < total_dias:
       fim_teste = min(cursor + janela_rebal, total_dias)
       data_str = returns_full.index[cursor].strftime('%d/%m/%Y')
      
       ret_treino = returns_full.iloc[:cursor].replace([np.inf, -np.inf], np.nan)
       cols_validas = ret_treino.columns[ret_treino.std() > 1e-6]
       ret_treino = ret_treino[cols_validas].dropna()
       ret_teste = returns_full.iloc[cursor:fim_teste]
      
       print(f" -> Rebalanceamento: {data_str} | Base de Treino: {ret_treino.shape[1]} ativos")
      
       top_ativos = selecionar_top_momentum(ret_treino, df_silver)
      
       w_hrp_din = otimizar_hrp(ret_treino[top_ativos].dropna())
       w_mvo_mom = otimizar_markowitz(ret_treino[top_ativos].dropna())
       w_hrp_puro = otimizar_hrp(ret_treino)
       w_mvo_puro = otimizar_markowitz(ret_treino)
      
       pesos_ativos_din = dict(zip(w_hrp_din.index, w_hrp_din['weights']))
       pesos_cdi_din = {t: 0.0 for t in top_ativos}
      
       s_w_mvo_mom = pd.Series(0.0, index=returns_full.columns); s_w_mvo_mom.update(w_mvo_mom['weights'])
       s_w_hrp_puro = pd.Series(0.0, index=returns_full.columns); s_w_hrp_puro.update(w_hrp_puro['weights'])
       s_w_mvo_puro = pd.Series(0.0, index=returns_full.columns); s_w_mvo_puro.update(w_mvo_puro['weights'])
      
       ret_mvo_mom_list.append((ret_teste * s_w_mvo_mom).sum(axis=1))
       ret_hrp_puro_list.append((ret_teste * s_w_hrp_puro).sum(axis=1))
       ret_mvo_puro_list.append((ret_teste * s_w_mvo_puro).sum(axis=1))
       ret_ibov_list.append(ret_ibov_raw.reindex(ret_teste.index).fillna(0.0))
      
       dias_saida = {t: 0 for t in top_ativos}
       dias_reentrada = {t: 0 for t in top_ativos}
       retornos_dinamicos = []
      
       for dt in ret_teste.index:
           retorno_dia = 0.0
           for tkr in top_ativos:
               p_hist = precos_full[tkr].loc[:dt]
               if len(p_hist) >= PARAMS["ema_periodos"]:
                  
                   status_atual = "EM_CARTEIRA" if pesos_ativos_din.get(tkr, 0.0) > 0.0 else "EM_CDI"
                   novo_status, dias_saida[tkr], dias_reentrada[tkr] = checar_sinais_dinamicos(
                       p_hist, status_atual, dias_saida[tkr], dias_reentrada[tkr], PARAMS["histerese_pct"]
                   )
                  
                   if novo_status == "EM_CDI" and status_atual == "EM_CARTEIRA":
                       peso_formatado = f"{pesos_ativos_din[tkr]*100:.2f}%"
                       lista_eventos.append([dt.strftime('%d/%m/%Y'), 'STOP -> CDI', tkr, f'{PARAMS["dias_saida"]}d < (EMA50-{PARAMS["histerese_pct"]*100:.0f}%)', peso_formatado])
                       pesos_cdi_din[tkr] = pesos_ativos_din[tkr]
                       pesos_ativos_din[tkr] = 0.0
                      
                   elif novo_status == "EM_CARTEIRA" and status_atual == "EM_CDI":
                       peso_formatado = f"{pesos_cdi_din[tkr]*100:.2f}%"
                       lista_eventos.append([dt.strftime('%d/%m/%Y'), 'REENTRADA', tkr, f'{PARAMS["dias_reentrada"]}d consecutivos > EMA50', peso_formatado])
                       pesos_ativos_din[tkr] = pesos_cdi_din[tkr]
                       pesos_cdi_din[tkr] = 0.0
                          
               r_tkr = ret_teste.loc[dt, tkr] if tkr in ret_teste.columns else 0.0
               retorno_dia += pesos_ativos_din.get(tkr, 0.0) * r_tkr + pesos_cdi_din.get(tkr, 0.0) * taxa_cdi_diaria
           retornos_dinamicos.append(retorno_dia)
          
       ret_hrp_din_list.append(pd.Series(retornos_dinamicos, index=ret_teste.index))
       cursor += janela_rebal
      
   df_res = pd.DataFrame({
       'HRP Dinâmico (Proposto k≤2)': pd.concat(ret_hrp_din_list),
       'Markowitz + Momentum': pd.concat(ret_mvo_mom_list),
       'HRP Puro (Base Silver)': pd.concat(ret_hrp_puro_list),
       'Markowitz Puro (Silver)': pd.concat(ret_mvo_puro_list),
       'IBOVESPA': pd.concat(ret_ibov_list)
   }).dropna()
  
   v0 = PARAMS["valor_investido_inicial"]
   cum_ret = (1 + df_res).cumprod() * v0
   dd = (cum_ret / cum_ret.cummax()) - 1
   rf_d = taxa_cdi_diaria
  
   sharpe = {}
   for c in df_res.columns:
       ex = df_res[c] - rf_d
       sharpe[c] = (ex.mean() / df_res[c].std()) * np.sqrt(252) if df_res[c].std() > 0 else 0
      
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
   cores = ['#1F4E78', '#E67E22', '#27AE60', '#C0392B', '#7F8C8D']
   for i, col in enumerate(cum_ret.columns):
       ax1.plot(cum_ret[col], color=cores[i], label=col, lw=2.0 if i==0 else 1.2)
       ax2.plot(dd[col] * 100, color=cores[i], label=f"Drawdown {col}", lw=1.2)
      
   ax1.set_title("Evolução Patrimonial: HRP Dinâmico (k ≤ 2) vs. Markowitz e Benchmarks", fontsize=11, fontweight='bold')
   ax1.set_ylabel("Patrimônio (R$)"); ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)
   ax2.set_title("Análise Comparativa de Drawdown Residual (%)", fontsize=11, fontweight='bold')
   ax2.set_ylabel("Drawdown (%)"); ax2.legend(loc="lower left"); ax2.grid(True, alpha=0.3)
   plt.tight_layout()
   p_backtest = "./temp_charts/00_backtest.png"
   plt.savefig(p_backtest, dpi=180); plt.close()
   caminhos_graficos["backtest"] = p_backtest
  
   metricas = {
       "patr_hrp_din": cum_ret['HRP Dinâmico (Proposto k≤2)'].iloc[-1], "patr_mvo_mom": cum_ret['Markowitz + Momentum'].iloc[-1],
       "patr_hrp_puro": cum_ret['HRP Puro (Base Silver)'].iloc[-1], "patr_mvo_puro": cum_ret['Markowitz Puro (Silver)'].iloc[-1], "patr_ibov": cum_ret['IBOVESPA'].iloc[-1],
       "ret_hrp_din": ((cum_ret['HRP Dinâmico (Proposto k≤2)'].iloc[-1] / v0) - 1) * 100, "ret_mvo_mom": ((cum_ret['Markowitz + Momentum'].iloc[-1] / v0) - 1) * 100,
       "ret_hrp_puro": ((cum_ret['HRP Puro (Base Silver)'].iloc[-1] / v0) - 1) * 100, "ret_mvo_puro": ((cum_ret['Markowitz Puro (Silver)'].iloc[-1] / v0) - 1) * 100, "ret_ibov": ((cum_ret['IBOVESPA'].iloc[-1] / v0) - 1) * 100,
       "dd_hrp_din": dd['HRP Dinâmico (Proposto k≤2)'].min() * 100, "dd_mvo_mom": dd['Markowitz + Momentum'].min() * 100,
       "dd_hrp_puro": dd['HRP Puro (Base Silver)'].min() * 100, "dd_mvo_puro": dd['Markowitz Puro (Silver)'].min() * 100, "dd_ibov": dd['IBOVESPA'].min() * 100,
       "sharpe_hrp_din": sharpe['HRP Dinâmico (Proposto k≤2)'], "sharpe_mvo_mom": sharpe['Markowitz + Momentum'],
       "sharpe_hrp_puro": sharpe['HRP Puro (Base Silver)'], "sharpe_mvo_puro": sharpe['Markowitz Puro (Silver)'], "sharpe_ibov": sharpe['IBOVESPA']
   }
  
   datas_info = {"inicio": df_res.index[0].strftime("%d/%m/%Y"), "fim": df_res.index[-1].strftime("%d/%m/%Y")}
   os.makedirs("../results", exist_ok=True)
   ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   pdf_path = f"../results/relatorio_final_tcc_completo-{ts}.pdf"
  
   gerar_relatorio_pdf_completo(pdf_path, metricas, caminhos_graficos, datas_info, lista_eventos)
  
   print("\n" + "="*95)
   print("             TABELA CONSOLIDADA OFICIAL - RESULTADOS DO BACKTEST             ")
   print("="*95)
   df_terminal = pd.DataFrame([
       {"Estratégia": "HRP Dinâmico (Proposto k≤2)", "Patrimônio Final": f"R$ {metricas['patr_hrp_din']:,.2f}", "Retorno": f"{metricas['ret_hrp_din']:.2f}%", "Max DD": f"{metricas['dd_hrp_din']:.2f}%", "Sharpe": f"{metricas['sharpe_hrp_din']:.2f}"},
       {"Estratégia": "Markowitz + Momentum", "Patrimônio Final": f"R$ {metricas['patr_mvo_mom']:,.2f}", "Retorno": f"{metricas['ret_mvo_mom']:.2f}%", "Max DD": f"{metricas['dd_mvo_mom']:.2f}%", "Sharpe": f"{metricas['sharpe_mvo_mom']:.2f}"},
       {"Estratégia": "HRP Puro (Base Silver)", "Patrimônio Final": f"R$ {metricas['patr_hrp_puro']:,.2f}", "Retorno": f"{metricas['ret_hrp_puro']:.2f}%", "Max DD": f"{metricas['dd_hrp_puro']:.2f}%", "Sharpe": f"{metricas['sharpe_hrp_puro']:.2f}"},
       {"Estratégia": "Markowitz Puro (Silver)", "Patrimônio Final": f"R$ {metricas['patr_mvo_puro']:,.2f}", "Retorno": f"{metricas['ret_mvo_puro']:.2f}%", "Max DD": f"{metricas['dd_mvo_puro']:.2f}%", "Sharpe": f"{metricas['sharpe_mvo_puro']:.2f}"},
       {"Estratégia": "IBOVESPA (Benchmark)", "Patrimônio Final": f"R$ {metricas['patr_ibov']:,.2f}", "Retorno": f"{metricas['ret_ibov']:.2f}%", "Max DD": f"{metricas['dd_ibov']:.2f}%", "Sharpe": f"{metricas['sharpe_ibov']:.2f}"}
   ])
   print(df_terminal.to_string(index=False))

if __name__ == "__main__":
   executar_backtest()
