#!/bin/sh

pip_ids_project_deps="dask[complete] pyarrow numpy pandas matplotlib imblearn tensorflow optuna" 

echo ---Hello!---
echo ---This is the setup script for the Intrusion Detection System Project---
echo ---Author : Fakhry Husssein Tatanaki---
echo A Python virtual enviroment will be created and dependencies for the project will be installed


while [ "$ans" != 'y' -o "$ans" != 'n' ]; do
echo -n '--Continue?-- (y/n)? '
  read ans
  if [ "$ans" == "n" ]; then
    exit 0
  fi

  if [ "$ans" == "y" ]; then
    break
  fi
done

python_version="$(python --version 2> /dev/null | grep -o '[0-9]' | head -n1)"
python_exec=python

if [ "$?" -ne 0 -o "$python_version" != "3" ];then
    python3 --version 2> /dev/null
    python_exec=python3
fi

if [ "$?" -ne 0  ]; then
  echo python 3 does not seem to be installed, exiting
  exit 1
fi

$python_exec -m venv venv
if [ "$?" -eq 0 ]; then
  echo virtual env created, installing dependencies
else
  echo could not create a virtual enviroment, is the venv module installed?
  exit 1
fi

source venv/bin/activate

python -m pip install -U pip $pip_ids_project_deps

if [ "$?" -eq 0 ]; then
  echo $'\n\n'
  echo ------------------------------------------------------------
  echo All ready! type the following in the terminal to use the venv
  echo ------------------------------------------------------------
  echo source venv/bin/activate
  echo ------------------------------------------------------------
else
  echo could not install dependencies
  exit 1
fi



