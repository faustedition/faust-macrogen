#!/bin/bash
#
# Sets up the graphviewer for production outside Docker. Requires uv
# (https://docs.astral.sh/uv/) and Graphviz with development headers
# (Debian: graphviz libgraphviz-dev build-essential pkg-config).

set -euo pipefail

# The path to the productive python environment:
PRODUCTION_ENV=${PRODUCTION_ENV:-"$PWD/venv"}

echob () {
  tput bold 2>/dev/null || true
  echo "$@"
  tput sgr0 2>/dev/null || true
}

echob "Installing the locked environment to $PRODUCTION_ENV ..."
UV_PROJECT_ENVIRONMENT="$PRODUCTION_ENV" uv sync --locked --no-dev --no-editable --extra production

if [[ ! -r run-server ]]
then
  cat > run-server.sh <<EOF
#!/bin/sh
cd "$PWD"
if test -r .env
then
    . ./.env
fi
exec $PRODUCTION_ENV/bin/gunicorn "\$@"
EOF
  chmod 755 ./run-server.sh
fi

echo
echo

config=gunicorn.conf.py 
if [[ -r "$config" ]]
then
  echob "You might now want to have a look at the existing config file, $config"
else
  cat > "$config" <<EOF
worker_class = 'uvicorn.workers.UvicornWorker'
proc_name = 'faust-macrogen-graphviewer'
wsgi_app = 'graphviewer.gvfa:app'
bind = '127.0.0.1:5000'
EOF
  echob "I have created a config file at $config. Edit it"
fi
echob "and run $PWD/run-server.sh to start the server for testing."
echo 

unit=faust-macrogen.service
if [[ -r "$unit" ]]
then
  echob "You might now want to have a look at the existing service file, $config"
else
  cat > "$unit" <<EOF
[Unit]
Description=gunicorn faust/macrogen
After=network.target

[Service]
WorkingDirectory=$PWD
User=$USER
Group=nogroup
ExecStart=$PWD/run-server.sh
ExecReload=/bin/kill -s HUP \$MAINPID

[Install]
WantedBy=default.target
EOF
  echob "I have created a systemd unit file at $unit. Edit and install it"
fi
echob "and run sudo systemctl enable --now `basename $unit` to enable the service."
