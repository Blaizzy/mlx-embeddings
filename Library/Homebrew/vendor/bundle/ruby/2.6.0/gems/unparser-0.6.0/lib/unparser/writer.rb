# frozen_string_literal: true

module Unparser
  module Writer
    include Generation, NodeHelpers

    def self.included(descendant)
      descendant.class_eval do
        include Anima.new(:buffer, :comments, :node, :local_variable_scope)

        extend DSL
      end
    end
  end # Writer
end # Unparser
