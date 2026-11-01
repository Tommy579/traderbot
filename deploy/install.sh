#!/usr/bin/env bash
# Installe le timer systemd UTILISATEUR du TradingBot (aucun privilège requis).
#
#   ./deploy/install.sh          # installe puis active le timer
#   ./deploy/install.sh --status # affiche l'etat du timer et du service
#
# Le service exécute `live --once` a 15:55 America/New_York chaque jour ouvrable.
# Desinstallation :
#   systemctl --user disable --now tradingbot.timer
#   rm ~/.config/systemd/user/tradingbot.{service,timer}

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    echo "ERREUR : ${PROJECT_DIR}/.venv/bin/python est introuvable." >&2
    echo "Creez le venv d'abord (uv sync) puis relancez l'installation." >&2
    exit 1
fi

if [[ "${1:-}" == "--status" ]]; then
    systemctl --user list-timers tradingbot.timer || true
    systemctl --user status tradingbot.service || true
    exit 0
fi

# Stratégie du cycle quotidien (par défaut : ema_cross).
STRATEGY="ema_cross"
if [[ "${1:-}" == "--strategy" && -n "${2:-}" ]]; then
    STRATEGY="${2}"
    shift 2
fi

mkdir -p "${UNIT_DIR}"
for unit in tradingbot.service tradingbot.timer; do
    sed -e "s|@PROJECT_DIR@|${PROJECT_DIR}|g" \
        -e "s|live --once|live --once --strategy ${STRATEGY}|" \
        "${PROJECT_DIR}/deploy/${unit}" > "${UNIT_DIR}/${unit}"
done

systemctl --user daemon-reload
systemctl --user enable --now tradingbot.timer

echo "Timer installe et active :"
systemctl --user list-timers tradingbot.timer
echo ""
echo "Prochaine exécution ci-dessus (15:55 America/New_York, jours ouvrables)."
echo "Journal des cycles : journalctl --user -u tradingbot.service -f"
