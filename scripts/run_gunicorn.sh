#!/bin/bash

# === 1. Verificar si existe archivo .env ===
if [ -f .env ]; then
  echo "[INFO] Cargando variables de entorno desde .env..."
  export $(grep -v '^#' .env | xargs)
else
  echo "[WARN] No se encontró el archivo .env. Usando valores por defecto."
fi

# === 2. Configuración por defecto (si alguna variable no está en .env) ===
: "${PORT:=5813}"               # Puerto por defecto 5000
: "${HOST:=0.0.0.0}"            # Escuchar en todas las interfaces
: "${WORKERS:=4}"               # Número de workers
: "${TIMEOUT:=120}"             # Timeout de los workers

# === 3. Definir el módulo Flask (ubicación del app) ===
APP_MODULE="ui.ui_app:app"      # ui/ui_app.py → app = Flask(...)

# === 4. Crear carpeta de logs si no existe ===
mkdir -p logs

# === 5. Mostrar configuración ===
echo "=========================================="
echo "[INFO] Iniciando Gunicorn con configuración:"
echo "App:         $APP_MODULE"
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "Workers:     $WORKERS"
echo "Timeout:     $TIMEOUT"
echo "=========================================="

# === 6. Lanzar Gunicorn ===
exec gunicorn "$APP_MODULE" \
  --workers "$WORKERS" \
  --bind "${HOST}:${PORT}" \
  --timeout "$TIMEOUT" \
  --log-level info \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
