# Bot de trading systématique — paper trading Alpaca uniquement

Architecture modulaire, typée et testée pour un bot de trading **restreint à
l'API Paper Trading d'Alpaca**. Aucun ordre réel ne peut être transmis : l'URL de
trading est validée à la construction de la configuration et le client
d'exécution refuse tout ce qui n'est pas `https://paper-api.alpaca.markets`.

## Les quatre garanties structurelles

1. **Découplage total** — `data` → `strategies` → `risk` → `execution`. La
   stratégie ne connaît ni le solde, ni la taille de position, ni le courtier ;
   elle ne produit qu'un `Signal` (BUY / SELL / HOLD).
2. **Aucun droit d'ordre pour la stratégie** — seul le `RiskEngine` peut créer un
   `OrderIntent`, et tout ordre porte un `ApprovalToken` dont le champ `issuer`
   référence un objet privé du module `risk.engine`. Un jeton forgé ailleurs est
   rejeté (`OrderIntent.__post_init__`), et toute modification de quantité, de
   sens ou de symbole après approbation casse l'invariant et lève `ValueError`.
3. **Parité backtest / live** — le backtest exécute les **mêmes appels** au
   moteur de risque que la boucle live : entrées classées par momentum, stops,
   trailing, prises de profit partielles, écrémage des lignes surpondérées,
   reconstitution du plancher de cash, financement du Bouclier et cash sweep.
   La différence tient au modèle de barres (exécution à l'ouverture `t+1`,
   stop sur le `low`), pas aux décisions. Le filtre d'actualités, lui, n'est
   pas consulté en backtest : il est injectable (`news_hook`) et inerte par
   défaut, faute de données point-in-time — l'équivalent de
   `NEWS_FILTER_ENABLED=false`.
4. **Déterminisme** — pas d'aléa dans le moteur : à barres identiques, deux
   backtests produisent la même courbe d'équité à l'unité flottante près (test
   dédié). Comme un fournisseur réseau peut réviser des prix entre deux
   téléchargements (yfinance « répare » certaines barres), les barres sont
   mises en cache disque et relues telles quelles : deux exécutions du même
   backtest donnent alors des résultats **bit à bit identiques**. Les
   identifiants de décision sont dérivés par SHA-256 des entrées, ce qui rend
   l'audit reproductible et les ordres idempotents côté Alpaca.

## Flux d'une décision

```
yfinance / Alpaca bars ──► normalisation OHLCV (UTC, qualité vérifiée)
        │
        ├─► Strategy.prepare()      EMA 20/50, RSI(14), ATR(14)
        ├─► Strategy.compute_signals()  →  Signal(BUY|SELL|HOLD)
        │
        ▼
RiskEngine.evaluate_entry()   ── dimensionnement, stop ATR, plafonds,
        │                        circuit breakers, jeton d'approbation
        ▼
RiskDecision(approved, code, reason, OrderIntent, metrics)   ── audit JSONL
        │
        ▼
AlpacaPaperClient.submit()  ── vérifie le jeton, refuse les rejeux (idempotence),
        │                       transmet un MarketOrderRequest (TIF=DAY)
        ▼
wait_for_fill()  ──► fill enregistré, stop persisté dans state/state.json
```

## Installation

```bash
cd /home/tom/Documents/trader
uv sync                     # crée .venv (Python 3.12) et installe les dépendances
cp .env.example .env        # renseignez vos clés PAPER Alpaca
```

`requirements.txt` est généré par `uv export` : versions exactes, y compris
transitives.

## Configuration

Tout passe par l'environnement ou `.env` (`trading_bot/config.py`, Pydantic
Settings). Les valeurs par défaut sont prudentes :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `SYMBOLS` | `AAPL,MSFT` | univers (CSV) |
| `BENCHMARK_SYMBOL` | `SPY` | benchmark de régime et de force relative (jamais tradé) |
| `TIMEFRAME` | `1Day` | `1Min` … `1Day` |
| `POLL_INTERVAL_SECONDS` | `60` | cadence de la boucle live |
| `RISK_PER_TRADE_PCT` | `0.01` | risque par trade si le stop est touché |
| `MAX_POSITION_PCT` | `0.20` | poids maximal d'une position |
| `MAX_GROSS_EXPOSURE_PCT` | `0.80` | exposition brute maximale |
| `MAX_OPEN_POSITIONS` | `5` | positions simultanées |
| `BULL_GROSS_EXPOSURE_PCT` | `0.90` | plafond d'exposition en régime haussier (SPY > SMA 200) |
| `BULL_MAX_POSITION_PCT` | `0.30` | plafond de position en régime haussier |
| `ATR_STOP_MULTIPLE` | `2.0` | distance du stop initial (× ATR) |
| `TRAILING_ATR_MULTIPLE` | `3.0` | distance du trailing stop (0 = désactivé) |
| `TRAILING_ACTIVATION_R` | `1.0` | gain en R avant activation du trailing |
| `MAX_DAILY_LOSS_PCT` | `0.03` | circuit breaker journalier |
| `MAX_DRAWDOWN_PCT` | `0.15` | arrêt définitif (drawdown depuis le pic) |
| `FRACTIONAL_SHARES` | `true` | fractions d'action (Alpaca paper) |
| `DRY_RUN` | `false` | analyse et journalise sans transmettre d'ordre |

Toute configuration incohérente (par exemple `MAX_DAILY_LOSS_PCT` ≥
`MAX_DRAWDOWN_PCT`) échoue au démarrage, jamais en séance.

## Utilisation

```bash
./tradingbot web                      # INTERFACE WEB LOCALE (recommandée)
./tradingbot check                    # lecture seule : compte, horloge, données, signal
./tradingbot backtest --symbols AAPL,MSFT --start 2022-01-01 \
    --output state/backtest.json --csv state/equity.csv --trades state/trades.csv
./tradingbot backtest --strategy donchian_breakout --entry 20 --exit-channel 10 --sma-long 200
./tradingbot backtest --strategy relative_momentum   # benchmark SPY téléchargé automatiquement
./tradingbot backtest --refresh       # retélécharge les barres et réécrit state/cache/
./tradingbot live --once --dry-run --strategy donchian_breakout
./tradingbot live                     # boucle paper (arrêt propre sur SIGINT/SIGTERM)
```

Trois moteurs de signaux sont disponibles via `--strategy` (défaut :
`ema_cross`) :

| Stratégie | Règle d'entrée | Règle de sortie |
| :-- | :-- | :-- |
| `ema_cross` | croisement EMA 20/50, RSI 50-70 | croisement EMA inverse |
| `donchian_breakout` | clôture > plus haut 20 j achevés, SMA 200 au-dessus | clôture < plus bas 10 j |
| `relative_momentum` | z-score momentum 60/120 j > SPY sur les deux horizons | momentum 60 j négatif ou sous SPY |
| `shield_sword` | Top 3 momentum composite, au rebalancement mensuel | sortie du Top 3 au rebalancement suivant |

`shield_sword` est une allocation Core-Satellite : `SHIELD_ALLOCATION_PCT` de
l'équité part dans l'actif du Bouclier (`SHIELD_SYMBOL`, SGOV par défaut), sans
stop ni trailing ; le reste est réparti sur les `SWORD_TOP_N` valeurs les plus
vigoureuses. Le rebalancement tombe le **premier jour de bourse du mois**
(`SWORD_REBALANCE_DAYS` = 20 séances) : cette ancre calendaire garantit que la
même date est un rebalancement en backtest et en live, où la fenêtre de barres
glisse à chaque cycle. Les signaux sont volontairement sans état : un actif
retenu reçoit BUY à chaque rebalancement, et c'est le moteur de risque qui
refuse un doublon (`POSITION_ALREADY_OPEN`), pas la stratégie.

`relative_momentum` exige un benchmark (SPY par défaut) : il est téléchargé
en backtest et récupéré chaque cycle en live — sans lui, la stratégie échoue
bruyamment plutôt que d'émettre un signal différent du backtest.

**Allocation dynamique** : à chaque séance, les candidats à l'entrée sont
classés par z-score de momentum (60/120 j) et le budget de liquidités va
d'abord aux actifs les plus vigoureux. Quand SPY clôture au-dessus de sa
SMA 200, les plafonds passent de 20 %/80 % (position/exposition brute) à
30 %/90 % ; en cas de doute sur le régime, le profil de base s'applique.

## Automatisation quotidienne

Les barres analysées sont journalières : le bot n'a besoin que d'un cycle
par jour de marché, à **15:55 heure de New York** (5 minutes avant la
clôture). Le fuseau est fixé côté planificateur, jamais en heure locale de
Paris : le décalage change pendant les semaines de bascule d'heure.

```bash
./deploy/install.sh            # installe et active le timer systemd utilisateur
./deploy/install.sh --status   # prochaine exécution, état du service
journalctl --user -u tradingbot.service -f   # suivi des cycles
```

`deploy/tradingbot.timer` déclenche `live --once` via
`OnCalendar=Mon..Fri *-*-* 15:55:00 America/New_York` (les fériés sont
filtrés par le bot : cycle ignoré si le marché est fermé). `Persistent=false`
est volontaire : une exécution rejouée au démarrage tomberait en pleine
séance sur des signaux de la veille. Alternative cronie :
`deploy/tradingbot.cron` (utilise `CRON_TZ`).

**Notifications** : renseignez `WEBHOOK_URL` dans `.env` pour recevoir le
récapitulatif JSON de chaque cycle exécuté (équité, signaux, ordres, halts,
erreurs) sur Discord ou Telegram (`WEBHOOK_FORMAT`, `WEBHOOK_CHAT_ID`).
L'envoi est best-effort : un webhook muet n'interrompt jamais le trading,
et les cycles ignorés (marché fermé) ne notifient pas.

## Interface web locale

`./tradingbot web` ouvre un tableau de bord sur
`http://127.0.0.1:8484`, pensé pour les non-financiers :

* **Tableau de bord** — valeur du compte (argent fictif), positions avec
  gains/pertes et filets de sécurité, état du pilotage automatique,
  diagnostic du jour en un clic ;
* **Tester sur le passé** — lance un backtest depuis le navigateur,
  graphique de la valeur du compte, résultats expliqués en français
  (pire chute, frais, temps en marché, comparaison buy & hold) ;
* **Piloter** — active/modifie le timer quotidien (choix de la stratégie),
  lance un cycle immédiat en répétition générale ou en vraie passe paper ;
* **Réglages** — univers de symboles et webhook Discord/Telegram sans
  éditer `.env` à la main (validation + restauration automatique) ;
* **Comprendre** — glossaire : paper trading, stop, ATR, drawdown,
  coupe-circuit, momentum.

Sécurité : écoute sur 127.0.0.1 uniquement, requêtes croix-site refusées
(en-tête `Origin`), hôte validé, et le mode paper reste verrouillé par
l'architecture — aucun bouton ne peut engager d'argent réel. L'onglet
« Piloter » sait aussi lever l'arrêt permanent du coupe-circuit (avec
confirmation et justification), évitant d'éditer `state/state.json` à la
main.

Les barres téléchargées sont mises en cache dans `state/cache/` (CSV,
`float_format="%.17g"` pour un aller-retour sans perte). C'est ce cache qui rend
les backtests relançables à l'identique ; `--refresh` force un
retéléchargement, `--no-cache` désactive complètement le cache.

`python -m trading_bot.main <commande>` fonctionne de la même façon, y compris
depuis un shell dont le `PYTHONPATH` pointe ailleurs (le lanceur le neutralise).

## Modèle de risque

Dimensionnement d'une entrée longue :

```
stop_distance   = ATR(14) × ATR_STOP_MULTIPLE
budget_risque   = equity × RISK_PER_TRADE_PCT
qty_risque      = budget_risque / stop_distance
capital_max     = min(equity × MAX_POSITION_PCT,
                      exposition_brute_restante,
                      min(cash, buying_power))
qty             = floor(min(qty_risque, capital_max / prix), pas autorisé)
```

Si la quantité tombe sous `MIN_ORDER_NOTIONAL`, l'entrée est refusée (motif
`BELOW_MIN_NOTIONAL` dans l'audit). Chaque refus est enregistré avec un code
machine-lisible : `HALTED`, `POSITION_ALREADY_OPEN`, `MAX_POSITIONS_REACHED`,
`GROSS_EXPOSURE_LIMIT`, `INSUFFICIENT_BUYING_POWER`, `INVALID_STOP`, etc.

Stops :

* **stop initial** = prix de décision − `ATR_STOP_MULTIPLE` × ATR(14) ;
* **trailing** — après `TRAILING_ACTIVATION_R` R de gain, le stop monte au
  minimum au prix d'entrée (mise à breakeven), puis suit
  `prix − TRAILING_ATR_MULTIPLE × ATR` ; il ne redescend jamais ;
* le relèvement est calculé **après la clôture** de la barre et prend effet à la
  barre suivante (aucun aller-retour rétroactif).

Circuit breakers :

* perte journalière ≥ `MAX_DAILY_LOSS_PCT` → arrêt pour la séance, remis à zéro
  à la séance suivante, avec liquidation immédiate des positions ;
* drawdown depuis le pic d'équité ≥ `MAX_DRAWDOWN_PCT` → arrêt **définitif**,
  levée manuelle explicite uniquement (`RiskEngine.reset_permanent_halt`).

L'état de risque (pic d'équité, équity de début de séance, arrêt en cours,
stops des positions ouvertes) est persisté de façon atomique dans
`state/state.json` ; chaque décision et chaque ordre sont écrits dans
`state/audit.jsonl`. Après un redémarrage, les positions sont réconciliées avec
le courtier et les stops restaurés — une position inconnue reçoit un stop ATR
d'office (avertissement dans les logs).

## Modèle de backtest

* signal de la barre `t` exécuté à l'ouverture de la barre `t+1` ;
* prix d'exécution = ouverture ± `slippage` (0,1 % par défaut) ;
* commission = 0,1 % du notionnel par côté ;
* stop vérifié sur le `low` de la barre (ou à l'ouverture en cas de gap) ;
* pas de réentrée dans la même barre après une sortie ;
* métriques produites : rendement total, CAGR, Sharpe, Sortino, max drawdown,
  Calmar, taux de réussite, profit factor, espérance, frais, exposition,
  circuit breakers, comparaison buy & hold.

Les signaux sont calculés de façon vectorisée (pandas), l'exécution barre par
barre pour rester fidèle aux stops et aux breakers — c'est ce qui garantit que
les résultats du backtest sont reproductibles.

## Qualité

```
337 tests passés            (env -u PYTHONPATH .venv/bin/python -m pytest -q)
couverture                  79 % des instructions
mypy --strict               (26 fichiers source, 0 erreur)
ruff check / ruff format    (0 violation)
```

La suite de tests est **entièrement hors ligne** : courtier simulé (compte,
positions, prix, fills), clients Alpaca injectés, données synthétiques
déterministes. Un marqueur `network` existe pour de futurs tests d'intégration
réseau.

## Limites connues

* **Stops différents selon le mode** — en paper live, le bot dépose des ordres
  stop-loss protecteurs chez Alpaca. Avant une vente au marché, il annule les
  ordres ouverts du symbole et relit le carnet ; si l'annulation ne peut pas
  être confirmée, la vente est bloquée. Le remplacement d'un stop suit la même
  règle. Si une vente est remplie mais que la réconciliation post-vente échoue,
  le cycle consigne une erreur et émet un log critique. En backtest, les stops
  restent simulés à partir du `low` (ou de l'ouverture en cas de gap) des barres ;
  aucun ordre courtier n'est placé.
* **Fraîcheur des prix** — la boucle live utilise le dernier trade connu du flux
  configuré (`IEX` par défaut, 15 min de retard possible en heures creuses) avec
  repli sur la dernière clôture en cas d'indisponibilité.
* **Univers actions uniquement**, long-only, une position par symbole, pas de
  vente à découvert ni d'effet de levier (le sizing plafonne sur les liquidités).
* **Validation** — les 337 tests et les backtests historiques (yfinance,
  AAPL+MSFT 2022-2024, cache disque) ont tourné sur cette machine. La boucle live n'a
  **pas** été exécutée contre un compte Alpaca réel dans cet environnement :
  elle est validée contre des doublures de
  `TradingClient`/`StockHistoricalDataClient` (`tests/test_execution.py`,
  `tests/test_main.py`). Lancez `./tradingbot check` avec vos clés paper pour
  valider la connectivité avant le premier cycle.

## Structure

```
trading_bot/
├── config.py              # Settings (Pydantic), RiskConfig, BacktestConfig, garde-fous paper
├── clock.py               # horloge alignée sur l'heure en ligne (offset monotonic)
├── data/
│   ├── loader.py          # normalisation OHLCV, yfinance, cache disque, indicateurs (pandas-ta + natif)
│   └── __init__.py
├── strategies/
│   ├── base.py            # Signal, SignalType, contrat Strategy (analyse pure)
│   ├── ema_cross.py       # EMA 20/50 filtré par RSI
│   ├── donchian_breakout.py  # cassure de canal 20 j + filtre SMA 200
│   ├── relative_momentum.py  # momentum 60/120 j vs benchmark SPY
│   ├── shield_sword.py    # Bouclier + Épée, rebalancement mensuel calendaire
│   └── __init__.py
├── risk/
│   ├── engine.py          # dimensionnement, stops ATR, trailing, breakers, jetons
│   └── __init__.py
├── backtest/
│   ├── runner.py          # moteur déterministe (frais, slippage, stops, métriques)
│   └── __init__.py
├── notifications/
│   ├── webhook.py         # récapitulatif Discord/Telegram, best-effort (urllib)
│   └── __init__.py
├── web/
│   ├── app.py             # serveur HTTP local (stdlib), API JSON, tâches de fond
│   ├── jobs.py            # gestionnaire de tâches (backtest, cycle, diagnostic)
│   └── static/index.html  # page unique : tableau de bord, graphique SVG, glossaire
├── execution/
│   ├── alpaca_client.py   # wrapper Alpaca Paper (alpaca-py), idempotence
│   └── __init__.py
└── main.py                # boucle live, persistance d'état, audit, CLI
tests/                     # 337 tests hors ligne
tradingbot                 # lanceur (uv, PYTHONPATH neutralisé)
```
