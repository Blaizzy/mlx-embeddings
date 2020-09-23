# typed: true
# frozen_string_literal: true

require "compat/global"

module Homebrew
  module CLI
    class Parser
      module Compat
        def parse(*)
          args = super
          Homebrew.args = args.dup
          args
        end
      end

      prepend Compat
    end
  end
end
