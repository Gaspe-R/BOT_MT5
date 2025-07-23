# --- Imports & initialisation ---
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import traceback
import ta
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta

# --- Initialisation MT5 ---

def initialize_mt5(symbol):
    if not mt5.initialize():
        raise RuntimeError("❌ Échec de la connexion à MetaTrader 5")
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"❌ Symbole {symbol} non trouvé dans MT5")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"❌ Impossible d’activer le symbole {symbol}")

# --- Données marché ---

def get_market_data(symbol, timeframe=mt5.TIMEFRAME_M5, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("❌ Données indisponibles pour la paire")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    rsi_value = df['rsi'].iloc[-2]

    st.write(f"📉 RSI (confirmé) : {rsi_value:.2f}")

    return df

# --- Indicateurs techniques ---

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_indicators(df):
    df['rsi'] = compute_rsi(df['close'])
    df['atr'] = compute_atr(df)
    df['ema_tendance'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema3'] = df['close'].ewm(span=3, adjust=False).mean()
    return df

# --- Détection signal ---

def detect_scalping_signal(df):
    price = df['close'].iloc[-1]
    ema8 = df['ema_tendance'].iloc[-2]
    ema5 = df['ema5'].iloc[-2]
    ema3 = df['ema3'].iloc[-2]
    rsi = df['rsi'].iloc[-1]
    close_prev = df['close'].iloc[-2]
    close_2prev = df['close'].iloc[-3]
    open_prev = df['open'].iloc[-2]
    open_2prev = df['open'].iloc[-3]

    bullish = (
    (close_prev > open_prev and price >= close_prev) or  # Bougie haussière + prix actuel au-dessus ou égal
    (close_prev > close_2prev and open_prev > open_2prev)  # Deux bougies consécutives qui montent
)

    bearish = (
    (close_prev < open_prev and price <= close_prev) or
    (close_prev < close_2prev and open_prev < open_2prev)
)

    # --- Signal d'achat ---

    if price > ema8:
        if ema3 < ema5 or ema5 < ema8:
            return None, "Pas de signal BUY : tendance haussière pas clairement établie"
        if rsi >= 70:
            return None, f"RSI trop élevé pour achat ({rsi:.2f})"
        if not bullish:
            return None, "Pas de confirmation haussière par chandelier"
        return "BUY", f"Signal BUY détecté | RSI: {rsi:.2f}"

    # --- Signal de vente ---

    elif price < ema8:
        if ema3 > ema5 or ema5 > ema8:
            return None, "Pas de signal SELL : tendance baissière pas clairement établie"
        if rsi <= 30:
            return None, f"RSI trop bas pour vente ({rsi:.2f})"
        if not bearish:
            return None, "Pas de confirmation baissière par chandelier "
        return "SELL", f"Signal SELL détecté | RSI: {rsi:.2f}"

    return None, "Pas de signal : Prix proche de ema8"

# --- Vérification de la marge disponible ---

def check_margin(symbol, lot):
    info = mt5.account_info()
    if info is None:
        return False, "Impossible de récupérer les infos compte"

    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    if tick is None or symbol_info is None:
        return False, "Données symboles indisponibles"

    margin_required = symbol_info.margin_initial * lot
    margin_free = info.margin_free

    if margin_free < margin_required:
        return False, f"Pas assez de marge libre. Requise: {margin_required}, Disponible: {margin_free}"
    return True, "Marge suffisante"


# --- Envoi ordre initial ---

def send_order(symbol, signal, lot=0.01, sl_factor=None, tp_factor=None):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("❌ Tick indisponible")
        return None

    price = tick.ask if signal == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        mt5.symbol_select(symbol, True)

    digits = info.digits
    df = get_market_data(symbol)
    atr = compute_atr(df).iloc[-1]
    if atr is None or np.isnan(atr):
        print("❌ ATR non disponible")
        return None

    # Ajustement des facteurs SL / TP dynamiquement
    if atr > 100:
        sl_factor = sl_factor or 1.5
        tp_factor = tp_factor or 1.0
    elif atr > 30:
        sl_factor = sl_factor or 2.0
        tp_factor = tp_factor or 1.5
    else:
        sl_factor = sl_factor or 2.5
        tp_factor = tp_factor or 2.0

    sl = price - sl_factor * atr if signal == "BUY" else price + sl_factor * atr
    tp = price + tp_factor * atr if signal == "BUY" else price - tp_factor * atr

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": round(price, digits),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": 20,
        "magic": 10102207,
        "comment": "Scalping dynamique ATR",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is not None and result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"⚠️ Ordre refusé | Retcode: {result.retcode} - {result.comment}")
    return result

    

# --- Trailing stop ---

def update_trailing_stop(symbol, trailing_factor=1.5):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    df = get_market_data(symbol)
    atr = compute_atr(df).iloc[-1]
    tick = mt5.symbol_info_tick(symbol)
    digits = mt5.symbol_info(symbol).digits
    if tick is None:
        st.error("Tick data non disponible pour le symbole.")
        return None
    price = tick.ask if signal == "BUY" else tick.bid
    
    for pos in positions:
        price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = price - trailing_factor * atr
            if new_sl > pos.sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": round(new_sl, digits),
                    "tp": pos.tp,
                })
        else:
            new_sl = price + trailing_factor * atr
            if new_sl < pos.sl or pos.sl == 0.0:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": round(new_sl, digits),
                    "tp": pos.tp,
                })
def explain_retcode(retcode):
    messages = {
        10027: "🚫 Erreur 10027 : Pas assez de marge pour cette position",
        10006: "⛔ Erreur 10006 : Pas de prix valide (market fermé ?)",
        10030: "🔁 Erreur 10030 : Requote requise",
        10013: "🔒 Erreur 10013 : Trading désactivé",
        10009: "✅ Succès : ordre exécuté",
        10021: "📛 Erreur 10021 : Paramètre invalide (volume, prix, etc.)",
        # Ajoute d'autres codes ici si nécessaire
    }
    return messages.get(retcode, f"❓ Erreur inconnue : Retcode {retcode}")

def get_account_info():
    info = mt5.account_info()
    if info is None:
        return None

    return {
        "Solde": f"{info.balance:.2f} €", 
        "Marge libre": f"{info.margin_free:.2f} €",
        "Marge utilisée": f"{info.margin:.2f} €",
        "Profit flottant": f"{info.profit:.2f} €",
        "Niveau de marge": f"{info.margin_level:.2f}%" if info.margin_level else "N/A",
        "Ordres ouverts": info.positions if hasattr(info, 'positions') else 'N/A'
    }

# --- Interface Streamlit ---

st.title("🤖 ScalBot MT5 by Yassine J. 🤖")
st_autorefresh(interval=10000, key="refresh")

symbol = st.selectbox("📈 Choisir une paire", ["BTCUSD", "XAUUSD"])

# --- Choix du timeframe ---

timeframe_str = st.selectbox("🕒 Timeframe", ["M1", "M5", "M15"])

timeframe_map = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
}
timeframe = timeframe_map[timeframe_str]

lot = st.number_input("💼 Taille du lot", min_value=0.01, max_value=10.0, value=0.01, step=0.01)
max_orders = st.number_input("🔢 Max ordres ouverts :", 1, 10, 1)

default_sl, default_tp = 2.0, 1.0
if "BTC" in symbol:
    default_sl, default_tp = 3.0, 1.5
elif "XAU" in symbol:
    default_sl, default_tp = 2.5, 1.2

sl_atr_factor = st.slider("🎯 Facteur ATR - Stop Loss", 0.5, 10.0, default_sl, step=0.1)
tp_atr_factor = st.slider("🎯 Facteur ATR - Take Profit", 0.1, 5.0, default_tp, step=0.1)
trailing_atr_factor = st.slider("🏃 Facteur ATR - Trailing Stop", 0.5, 10.0, default_sl, step=0.1)
enable_trailing = st.checkbox("🔁 Activer trailing stop", value=False)

if "logs" not in st.session_state:
    st.session_state.logs = []
if "bot_running" not in st.session_state:
    st.session_state.bot_running = False

if st.button("🛑 Stopper le bot"):
    st.session_state.bot_running = False
    st.success("🛑 Bot arrêté manuellement")

if st.button("▶ Démarrer le bot"):
    try:
        initialize_mt5(symbol)
        st.session_state.bot_running = True
        st.success("✅ Bot démarré manuellement")
    except Exception as e:
        st.error(f"Erreur initialisation MT5 : {e}")

# --- Traitement principal ---

if st.session_state.bot_running:
    try:
        initialize_mt5(symbol)
        df = get_market_data(symbol)
        df = compute_indicators(df)

        signal, reason = detect_scalping_signal(df)
        current_positions = mt5.positions_get(symbol=symbol)
        current_count = len(current_positions) if current_positions else 0
        log_time = datetime.now().strftime('%H:%M:%S')

        if signal and current_count < max_orders:
            result = send_order(
    symbol,
    signal,
    lot,
    sl_factor=sl_atr_factor,
    tp_factor=tp_atr_factor,
    timeframe=timeframe   # 👉 ici tu transmets ce que tu as sélectionné dans Streamlit
)

            if result is not None and result.retcode == 10009:
                st.success(f"✅ Ordre exécuté : {signal}")
                st.session_state.logs.insert(0, f"[{log_time}] ✅ SIGNAL {signal} exécuté | {reason}")
            else:
                retcode = result.retcode if result else None
                msg = explain_retcode(retcode)
                st.error(f"❌ MT5 Retcode: {retcode} — {msg}")
                st.session_state.logs.insert(0, f"[{log_time}] ❌ Erreur MT5 : {msg}")
        else:
            st.info(f"ℹ️ Pas de signal : {reason}")
            st.session_state.logs.insert(0, f"[{log_time}] ℹ️ {reason}")

        if enable_trailing:
            update_trailing_stop(symbol, trailing_factor=trailing_atr_factor)

    except Exception as e:
        error_trace = traceback.format_exc()
        st.session_state.logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] 💥 Erreur : {e}\n{error_trace}")

# --- Affichage logs ---

st.text_area("📋 Logs", value="\n".join(st.session_state.logs[:15]), height=200)

st.markdown("### 🧾 Informations Compte MT5")
account_data = get_account_info()
if account_data:
    for key, value in account_data.items():
        st.write(f"🔹 {key}: {value}")
else:
    st.warning("⚠️ Impossible de récupérer les informations du compte.")




################################################################################################################################################



# --- Backtest Streamlit Section ---

from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

st.subheader("🧪 Backtest")

# 1. Sélection des dates par l'utilisateur
today = datetime.now()
default_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input("📆 Date de début", default_start)
with col2:
    to_date = st.date_input("📅 Date de fin", today)

# Conversion en datetime avec heure 00:00
from_dt = datetime.combine(from_date, datetime.min.time())
to_dt = datetime.combine(to_date, datetime.max.time())

if st.button("🔁 Lancer le backtest"):
    try:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, from_dt, to_dt)
        if rates is None or len(rates) == 0:
            st.error("❌ Aucune donnée reçue pour le backtest.")
            st.stop()

        df_bt = pd.DataFrame(rates)
        df_bt['time'] = pd.to_datetime(df_bt['time'], unit='s')
        df_bt = compute_indicators(df_bt)


        buy_count, sell_count = 0, 0
        win_count, loss_count = 0, 0
        results = []
        total_profit = 0.0

        # Multi positions : on garde une liste de positions ouvertes
        open_positions = []

        lot_size_factor = 100000
        if "XAU" in symbol:
            lot_size_factor = 100
        elif "BTC" in symbol:
            lot_size_factor = 1

        lot_value = 0.02  # lot défini dans l'interface

        max_positions = 5  # nombre max de positions ouvertes en même temps
        trailing_atr_factor = 0  # trailing stop en ATR (à ajuster ou mettre à 0 pour désactiver)

        for i in range(20, len(df_bt) - 1):
            sub_df = df_bt.iloc[:i+1]
            current_time = df_bt['time'].iloc[i]
            atr = df_bt['atr'].iloc[i]
            if np.isnan(atr):
                continue
            close_price = df_bt['close'].iloc[i]
            high = df_bt['high'].iloc[i]
            low = df_bt['low'].iloc[i]

            # 1) Entrée de nouvelles positions si on n’a pas atteint le max
            if len(open_positions) < max_positions:
                signal, _ = detect_scalping_signal(sub_df)
                if signal:
                    entry_price = close_price
                    if signal == 'BUY':
                        sl = entry_price - sl_atr_factor * atr
                        tp = entry_price + tp_atr_factor * atr
                        buy_count += 1
                    else:
                        sl = entry_price + sl_atr_factor * atr
                        tp = entry_price - tp_atr_factor * atr
                        sell_count += 1

                    open_positions.append({
                        'type': signal,
                        'entry': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'entry_time': current_time
                    })

            # 2) Gestion des positions ouvertes
            positions_to_close = []
            for idx, pos in enumerate(open_positions):
                # Trailing stop update
                if trailing_atr_factor > 0:
                    if pos['type'] == 'BUY':
                        new_sl = max(pos['sl'], high - trailing_atr_factor * atr)
                        pos['sl'] = new_sl
                    elif pos['type'] == 'SELL':
                        new_sl = min(pos['sl'], low + trailing_atr_factor * atr)
                        pos['sl'] = new_sl

                # Check TP/SL
                if pos['type'] == 'BUY':
                    if high >= pos['tp']:
                        win_count += 1
                        profit = (pos['tp'] - pos['entry']) * lot_value * lot_size_factor
                        total_profit += profit
                        results.append((current_time, 'BUY', '✅'))
                        positions_to_close.append(idx)
                    elif low <= pos['sl']:
                        loss_count += 1
                        profit = (pos['sl'] - pos['entry']) * lot_value * lot_size_factor
                        total_profit += profit
                        results.append((current_time, 'BUY', '❌'))
                        positions_to_close.append(idx)

                elif pos['type'] == 'SELL':
                    if low <= pos['tp']:
                        win_count += 1
                        profit = (pos['entry'] - pos['tp']) * lot_value * lot_size_factor
                        total_profit += profit
                        results.append((current_time, 'SELL', '✅'))
                        positions_to_close.append(idx)
                    elif high >= pos['sl']:
                        loss_count += 1
                        profit = (pos['entry'] - pos['sl']) * lot_value * lot_size_factor
                        total_profit += profit
                        results.append((current_time, 'SELL', '❌'))
                        positions_to_close.append(idx)

            # On ferme les positions
            for idx in sorted(positions_to_close, reverse=True):
                open_positions.pop(idx)

        total_trades = win_count + loss_count
        success_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        st.write(f"Symbole : {symbol}")
        st.write(f" Nb de potition: {max_positions:.2f}")
        st.write(f"Total BUY: {buy_count} | SELL: {sell_count}")
        st.write(f"✅ Gagnants: {win_count} | ❌ Perdants: {loss_count}")
        st.write(f"🎯 Taux de réussite: {success_rate:.2f}%")
        st.write(f"💰 Profit / Perte total estimé: {total_profit:.2f} €")

        # Graphique des signaux (idem code initial)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_bt['time'],
            open=df_bt['open'],
            high=df_bt['high'],
            low=df_bt['low'],
            close=df_bt['close'],
            name='Prix'
        ))
        for i in range(20, len(df_bt) - 1):
            sub_df = df_bt.iloc[:i+1]
            signal, _ = detect_scalping_signal(sub_df)
            if signal:
                time = sub_df['time'].iloc[-1]
                price = sub_df['close'].iloc[-1]
                color = 'green' if signal == 'BUY' else 'red'
                symbol_mark = '✅' if signal == 'BUY' else '🔻'
                fig.add_trace(go.Scatter(
                    x=[time],
                    y=[price],
                    mode='text',
                    text=[symbol_mark],
                    textposition='top center',
                    textfont=dict(color=color, size=14),
                    name=signal
                ))
        fig.update_layout(
            title="📈 Signaux de trading détectés",
            xaxis_title="Temps",
            yaxis_title="Prix",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur backtest: {e}")
