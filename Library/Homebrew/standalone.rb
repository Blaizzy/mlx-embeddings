# typed: true
# frozen_string_literal: true

# This file should be the first `require` in all entrypoints outside the `brew` environment.

require_relative "standalone/load_path"
require_relative "standalone/sorbet"
