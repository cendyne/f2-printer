#!/bin/bash
set -e

if [ ! -d "~/.rustup" ]; then
echo "Installing rust, in case"
curl https://sh.rustup.rs -sSf | sh
fi

source "$HOME/.cargo/env"

if [ ! -d "~/.pyenv" ];
echo "Installing pyenv"

curl https://pyenv.run | bash
fi

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "Installing Python 3.10.10"

pyenv install 3.10.10

echo "Setting up environment"
pyenv virtualenv 3.10 threedotten
echo "Activating environment"
pyenv activate threedotten
echo "Installing application installer"
pip3 install poetry
echo "Installing dependencies"
poetry install

echo ""
echo ""

architecture=$(uname -m)
if [[ "$architecture" == "x86_64" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www3.hidglobal.com/drivers/24365"
elif [[ "$architecture" == "x86" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www3.hidglobal.com/drivers/24363"
else
  echo "This platform $architecture does not support HID FARGO DTC1250E drivers"
fi
