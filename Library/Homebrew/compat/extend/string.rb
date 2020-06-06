# frozen_string_literal: true

class String
  module Compat
    # String.chomp, but if result is empty: returns nil instead.
    # Allows `chuzzle || foo` short-circuits.
    def chuzzle
      odeprecated "chuzzle", "chomp.presence"
      s = chomp
      s unless s.empty?
    end
  end

  prepend Compat
end
