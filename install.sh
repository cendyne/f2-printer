#!/bin/bash
set -e

if [ ! -d "$HOME/.cargo" ]; then
echo "Installing rust, in case"
curl https://sh.rustup.rs -sSf | sh
fi

source "$HOME/.cargo/env"

if [ ! -d "$HOME/.pyenv" ]; then
echo "Installing pyenv"
curl https://pyenv.run | bash
fi

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "Installing Python 3.10.10"

pyenv install --skip-existing 3.10.10

# it is fine if the following has errors while searching
set +e
search=$(pyenv virtualenvs | grep "threedotten")
# restore error condition
set -e

if [ -z "$search" ]; then
echo "Setting up environment"
pyenv virtualenv 3.10 threedotten
else
echo "threedotten environment already set up"
fi
echo "Activating environment"
pyenv activate threedotten

echo "Installing application installer"
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies"
poetry config virtualenvs.in-project true
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install || true

echo ""
echo ""

# it is fine if the following has errors while searching
set +e

if [ -f "$HOME/.bashrc" ]; then
  search=$(grep "PYENV_ROOT" "$HOME/.bashrc")
  if [ -z "$search" ]; then
cat << EOF >> "$HOME/.bashrc"
export PATH="\$HOME/.local/bin:\$PATH"
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv virtualenv-init -)"
EOF
echo "Installed pyenv into bash, please begin a new session"
  fi
fi

if [ -f "$HOME/.zshrc" ]; then
  search=$(grep "PYENV_ROOT" "$HOME/.zshrc")
  if [ -z "$search" ]; then
cat << EOF >> "$HOME/.zshrc"
export PATH="\$HOME/.local/bin:\$PATH"
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv virtualenv-init -)"
EOF
echo "Installed pyenv into zsh, please begin a new session"
  fi
fi

architecture=$(uname -m)
if [[ "$architecture" == "x86_64" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www.hidglobal.com/drivers/41707"
elif [[ "$architecture" == "x86" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www.hidglobal.com/drivers/41707"
else
  echo "This platform $architecture does not support HID FARGO DTC1250E drivers"
fi
