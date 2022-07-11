#!/bin/bash

### The following environment variables can be used to adjust processing

# 1. We're downloading and installing micromamba here, if required:
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-"$PWD/mamba"}

# 2. The path to the productive python environment:
PRODUCTION_ENV=${PRODUCTION_ENV:-"$PWD/venv"}

#   if present, use this file to initialize the production environment:
ENVIRONMENT_FILE=${ENVIRONMENT_FILE:-"environment.yml"}

#   otherwise, just install python in the given version:
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

# 3. finally, run pip with the following argument to install the project:
PIP_SPEC="${PIP_SPEC:-.[production]}"
###

export MAMBA_ROOT_PREFIX PRODUCTION_ENV
micromamba="$MAMBA_ROOT_PREFIX/bin/micromamba"

echob () {
  tput bold
  echo "$@"
  tput sgr0
}

if [[ -x "$micromamba" ]]; then
  echob "Micromamba environment present at $MAMBA_ROOT_PREFIX"
else
  echob "Downloading micromamba to $MAMBA_ROOT_PREFIX ..."
  mkdir "$MAMBA_ROOT_PREFIX" && wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C "$MAMBA_ROOT_PREFIX" -xj bin/micromamba
fi

if [[ -r ${ENVIRONMENT_FILE} ]]
then
  echob "Found environment.yml, installing its contents to $PRODUCTION_ENV"
  $micromamba create --no-rc --yes -p "$PRODUCTION_ENV" -f "${ENVIRONMENT_FILE}"
elif [[ -x "${PRODUCTION_ENV}/bin/python${PYTHON_VERSION}" ]]
then
  echob "Python $PYTHON_VERSION is already installed in $PRODUCTION_ENV"
else
  echob "Using micromamba to install Python $PYTHON_VERSION to $PRODUCTION_ENV ..."
  $micromamba create --no-rc --yes -p "$PRODUCTION_ENV" python=3.10 wheel -c conda-forge
fi

if [[ -n "$PIP_SPEC" ]]
then
  echob "Using pip to install the project ($PIP_SPEC) ..."
  $micromamba --no-rc -p "$PRODUCTION_ENV" run pip install "$PIP_SPEC"
fi

if [[ ! -r run-server ]]
then
  cat > run-server.sh <<EOF
#!/bin/sh
cd "$PWD"
if test -r .env
then
    . ./.env
fi
exec $PRODUCTION_ENV/bin/gunicorn
EOF
  chmod 755 ./run-server.sh
  echob "You may now want to create an .env file to configure stuff"
  echob "and run $PWD/run-server.sh to start the server."
fi

