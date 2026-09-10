import json
import pandas as pd
import datetime
import os

PARAMS = {
   "ema_periodos": 50,
   "histerese_pct": 0.03,
   "dias_saida": 4,
   "dias_reentrada": 2,
   "state_file": "./portfolio_state.json"
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

def monitorar_diario():
   if not os.path.exists(PARAMS["state_file"]):
       print("Arquivo de estado não encontrado.")
       return

   with open(PARAMS["state_file"], 'r', encoding='utf-8') as f:
       state = json.load(f)

   csv_price = pd.read_csv("../datasets/snapshots_dados/historico_precos_tela.csv", index_col=0, parse_dates=True)
  
   for tkr, dados in state["ativos"].items():
       if tkr not in csv_price.columns:
           continue
          
       p_hist = csv_price[tkr].dropna()
       if len(p_hist) < PARAMS["ema_periodos"]:
           continue
          
       novo_status, dados["dias_saida"], dados["dias_reentrada"] = checar_sinais_dinamicos(
           p_hist,
           dados["status"],
           dados["dias_saida"],
           dados["dias_reentrada"],
           PARAMS["histerese_pct"]
       )
      
       if novo_status == "EM_CDI" and dados["status"] == "EM_CARTEIRA":
           dados["peso_cdi"] = dados["peso_acao"]
           dados["peso_acao"] = 0.0
           dados["status"] = "EM_CDI"
           print(f"[STOP ACIONADO] {tkr} migrado para CDI.")
          
       elif novo_status == "EM_CARTEIRA" and dados["status"] == "EM_CDI":
           dados["peso_acao"] = dados["peso_cdi"]
           dados["peso_cdi"] = 0.0
           dados["status"] = "EM_CARTEIRA"
           print(f"[REENTRADA] {tkr} retornou para a carteira.")

   state["data_atualizacao"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   with open(PARAMS["state_file"], 'w', encoding='utf-8') as f:
       json.dump(state, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
   monitorar_diario()
