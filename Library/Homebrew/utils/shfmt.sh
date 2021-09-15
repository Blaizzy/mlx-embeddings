#!/bin/bash

onoe() {
  echo "$*" >&2
}

odie() {
  onoe "$@"
  exit 1
}

# HOMEBREW_PREFIX is set by extend/ENV/super.rb
# shellcheck disable=SC2154
if [[ -z "${HOMEBREW_PREFIX}" ]]
then
  odie "${0##*/}: This program is internal and must be run via brew."
fi

# HOMEBREW_PREFIX is set by extend/ENV/super.rb
# shellcheck disable=SC2154
SHFMT="${HOMEBREW_PREFIX}/opt/shfmt/bin/shfmt"

if [[ ! -x "${SHFMT}" ]]
then
  odie "${0##*/}: Please install shfmt by running \`brew install shfmt\`."
fi

if [[ ! -x "$(command -v diff)" ]]
then
  odie "${0##*/}: Please install diff by running \`brew install diffutils\`."
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

check_read_and_write() {
  [[ -f "$1" && -w "$1" ]]
}

###
### Custom shell script styling
###

# Check pattern:
# '^\t+'
#
# Replace tabs with 2 spaces instead
#
no_tabs() {
  local file="$1"

  # TODO: use bash built-in regex match syntax instead
  if grep -qE '^\t+' "${file}"
  then
    # TODO: add line number
    onoe "Indent by tab detected."
    return 1
  fi
}

# Check pattern:
# for var in ... \
#            ...; do
#
# Use the followings instead (keep for statements only one line):
# ARRAY=(
#   ...
#   ...
# )
# for var in "${ARRAY[@]}"
# do
#
no_multiline_for_statements() {
  local file="$1"
  check_read_and_write "${file}" || return 1

  # TODO: use bash built-in regex match syntax instead
  if grep -qE '^\s*for .*\\\(#.*\)\?$' "${file}"
  then
    # TODO: add line number
    onoe "Multi-line for statement detected."
    return 1
  fi
}

# Check pattern:
# IFS=$'\n'
#
# Use the followings instead:
# while IFS='' read -r line
# do
#   ...
# done < <(command)
#
no_IFS_newline() {
  local file="$1"
  check_read_and_write "${file}" || return 1


  # TODO: use bash built-in regex match syntax instead
  if grep -qE "^[^#]*IFS=\\\$'\\\\n'" "${file}"
  then
    # TODO: add line number
    onoe "Pattern \`IFS=\$'\\\\n'\` detected."
    return 1
  fi
}

# TODO: Wrap `then` to a separated line
# before:                   after:
# if [[ ... ]]; then        if [[ ... ]]
#                           then
#
# before:                   after:
# if [[ ... ]] ||           if [[ ... ]] ||
#   [[ ... ]]; then           [[ ... ]]
#                           then
#
wrap_then() {
  local file="$1"
  check_read_and_write "${file}" || return 1

  true
}

# Probably merge into the above function
# TODO: Wrap `do` to a separated line
# before:                   after:
# for var in ...; do        for var in ...do
#                           do
#
wrap_do() {
  local file="$1"
  check_read_and_write "${file}" || return 1

  true
}

# TODO: Align multiline if condition (indent with 3 spaces or 6 spaces (start with "-"))
# before:                   after:
# if [[ ... ]] ||           if [[ ... ]] ||
#   [[ ... ]]                  [[ ... ]]
# then                      then
#
# before:                   after:
# if [[ -n ... || \         if [[ -n ... || \
#   -n ... ]]                     -n ... ]]
# then                      then
#
align_multiline_if_condition() {
  local file="$1"
  check_read_and_write "${file}" || return 1

  true
}

# TODO: It's hard to align multiline switch cases
align_multiline_switch_cases() {
  true
}

format() {
  local file="$1"
  # shellcheck disable=SC2155
  local tempfile="$(dirname "${file}")/.${file##*/}.temp"
  local retcode=0

  cp -af "${file}" "${tempfile}"

  # Format with `shfmt` first
  "${SHFMT}" -w "${SHFMT_ARGS[@]}" "${tempfile}"

  # Fail fast when forbidden patterns detected
  if ! no_tabs "${tempfile}" ||
     ! no_multiline_for_statements "${tempfile}" ||
     ! no_IFS_newline "${tempfile}"
  then
    rm -f "${tempfile}" 2>/dev/null
    return 1
  fi

  # Tweak it with custom shell script styles
  wrap_then "${tempfile}"
  wrap_do "${tempfile}"
  align_multiline_if_condition "${tempfile}"

  if ! diff -q "${file}" "${tempfile}"
  then
    # Show differences
    diff -d -C 1 --color=auto "${file}" "${tempfile}"
    if [[ -n "${INPLACE}" ]]; then
      cp -af "${tempfile}" "${file}"
    fi
    retcode=1
  else
    # File is identical between code formations (good styling)
    retcode=0
  fi
  rm -f "${tempfile}" 2>/dev/null
  return "${retcode}"
}

for file in "${FILES[@]}"
do
  # TODO: catch return values
  format "${file}"
done
