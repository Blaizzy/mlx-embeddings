# typed: strict
# frozen_string_literal: true

require "active_support/core_ext/array/conversions"

require "rubocop-performance"
require "rubocop-rails"
require "rubocop-rspec"
require "rubocop-sorbet"

require_relative "io_read"
require_relative "shell_commands"

require_relative "formula_desc"
require_relative "components_order"
require_relative "components_redundancy"
require_relative "dependency_order"
require_relative "homepage"
require_relative "text"
require_relative "caveats"
require_relative "checksum"
require_relative "patches"
require_relative "conflicts"
require_relative "options"
require_relative "urls"
require_relative "lines"
require_relative "livecheck"
require_relative "class"
require_relative "uses_from_macos"
require_relative "files"
require_relative "keg_only"
require_relative "version"
require_relative "deprecate_disable"
require_relative "bottle"

require_relative "rubocop-cask"
