# frozen_string_literal: true

class String
  module Compat
    def chuzzle
      odisabled ".chuzzle", "&.chomp.presence"
    end
  end

  prepend Compat
end
