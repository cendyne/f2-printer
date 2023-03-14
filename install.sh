#!/bin/bash
sudo apt install libffi-dev libssl-dev git curl
curl https://pyenv.run | bash
curl https://sh.rustup.rs -sSf | sh

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.10.10
pyenv virtualenv 3.10 threedotten
pyenv activate threedotten
pip3 install poetry
poetry install

architecture=$(uname -m)
if [[ "$architecture" == "x86_64" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www3.hidglobal.com/drivers/24365"
elif [[ "$architecture" == "x86" ]]; then
  echo "Download LINUX 64 BIT DRIVER at https://www3.hidglobal.com/drivers/24363"
else
  echo "This platform $architecture does not support HID FARGO DTC1250E drivers"
fi
