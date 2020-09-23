# typed: strict
# frozen_string_literal: true

if ENV["HOMEBREW_TESTS_COVERAGE"]
  require "sorbet-runtime"
else
  require "utils/sorbet/stubs"
end
