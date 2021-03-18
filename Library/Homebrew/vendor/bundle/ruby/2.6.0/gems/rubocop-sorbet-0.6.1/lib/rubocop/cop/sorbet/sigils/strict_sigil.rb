# frozen_string_literal: true

require 'rubocop'
require_relative 'has_sigil'

module RuboCop
  module Cop
    module Sorbet
      # This cop makes the Sorbet `strict` sigil mandatory in all files.
      class StrictSigil < HasSigil
        def minimum_strictness
          'strict'
        end
      end
    end
  end
end
