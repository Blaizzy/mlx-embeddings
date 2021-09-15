#!/bin/bash

# HOMEBREW_LIBRARY is set by bin/brew
# HOMEBREW_PREFIX is set by extend/ENV/super.rb
# shellcheck disable=SC2154
if [[ -z "${HOMEBREW_LIBRARY}" || -z "${HOMEBREW_PREFIX}" ]]
then
  echo "${0##*/}: This program is internal and must be run via brew." >&2
  exit 1
fi

# HOMEBREW_PREFIX is set by extend/ENV/super.rb
# shellcheck disable=SC2154
SHFMT="${HOMEBREW_PREFIX}/opt/shfmt/bin/shfmt"

if [[ ! -x "${SHFMT}" ]]
then
  echo "${0##*/}: Please install shfmt by running \`brew install shfmt\`." >&2
  exit 1
fi

SHFMT_ARGS=()
INPLACE=''
while [[ $# -gt 0 ]]
do
  arg="$1"
  if [[ "${arg}" == "--" ]]
  then
    shift
    break
  fi
  if [[ "${arg}" == "-w" ]]
  then
    shift
    INPLACE=1
    continue
  fi
  SHFMT_ARGS+=("${arg}")
  shift
done

FILES=()
for file in "$@"
do
  if [[ -f "${file}" ]]
  then
    FILES+=("${file}")
  else
    echo "${0##*/}: File \"${file}\" does not exist." >&2
    exit 1
  fi
done

if [[ "${#FILES[@]}" == 0 ]]
then
  exit
fi

"${SHFMT}" "${SHFMT_ARGS[@]}" "${FILES[@]}"
