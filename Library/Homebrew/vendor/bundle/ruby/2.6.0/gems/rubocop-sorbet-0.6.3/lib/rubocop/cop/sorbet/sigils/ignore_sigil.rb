# frozen_string_literal: true

require "rubocop"
require_relative "has_sigil"

module RuboCop
  module Cop
    module Sorbet
      # This cop makes the Sorbet `ignore` sigil mandatory in all files.
      class IgnoreSigil < HasSigil
        def minimum_strictness
          "ignore"
        end
      end
    end
  end
end
