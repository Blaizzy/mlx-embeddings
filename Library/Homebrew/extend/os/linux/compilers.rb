# typed: strict
# frozen_string_literal: true

class CompilerSelector
  sig { returns(String) }
  def self.preferred_gcc
    # gcc-5 is the lowest gcc version we support on Linux.
    # gcc-5 is the default gcc in Ubuntu 16.04 (used for our CI)
    "gcc@5"
  end
end
