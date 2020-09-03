# frozen_string_literal: true

class NilClass
  module Compat
    def chuzzle
      odisabled ".chuzzle", "&.chomp.presence"
    end
  end

  prepend Compat
end
