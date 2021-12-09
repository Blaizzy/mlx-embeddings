# frozen_string_literal: true

require "rubocop"
require_relative "has_sigil"

module RuboCop
  module Cop
    module Sorbet
      # This cop makes the Sorbet `false` sigil mandatory in all files.
      class FalseSigil < HasSigil
        def minimum_strictness
          "false"
        end
      end
    end
  end
end
