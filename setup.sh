#!/bin/sh

current_dir="$(pwd)"
macOS=false

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      printf 'Installing on macOS...'
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf 'Installing on Linux...'
      ;;
    Windows)
      printf 'Installing on Windows...'
      ;;
    *)
      printf 'Only Linux, macOS and Windows systems are currently supported.'
      exit 1
      ;;
  esac
}

install_python_packages() {
  printf '\nInstall Python packages\n'
  python3 -m pip install -r requirements.txt
  case "$(uname -s)" in
  Linux|Darwin)
    python3 -m pip install cvxpy==1.1.20
    ;;
  Windows)
    python3 -m pip install cvxpy==1.0.26
    ;;
  *)
    printf 'Only Linux, macOS and Windows systems are currently supported.'
    exit 1
    ;;
  esac
}

install_oasis() {
  printf '\nInstalling OASIS...'
  cd "$current_dir" || exit 1
  printf '\nInstalling Gurobi...'
  conda config --add channels http://conda.anaconda.org/gurobi
  conda install gurobi -y
  printf '\nInstalling Mosek...'
  conda install -c mosek mosek -y
  git clone https://github.com/j-friedrich/OASIS.git oasis
  cd oasis || exit 1
  python3 setup.py build_ext --inplace
  python3 -m pip install -e .
}

install_elephant() {
  printf '\nInstalling Elephant...'
  cd "$current_dir" || exit 1
  git clone git://github.com/NeuralEnsemble/elephant.git elephant
  cd elephant || exit 1
  python3 setup.py install
}

check_requirements
install_python_packages
install_oasis
install_elephant

printf '\nSetup completed.'
