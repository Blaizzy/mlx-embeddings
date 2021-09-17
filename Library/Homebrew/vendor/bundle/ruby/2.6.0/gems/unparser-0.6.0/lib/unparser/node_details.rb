# frozen_string_literal: true

module Unparser
  module NodeDetails
    include Constants, NodeHelpers

    def self.included(descendant)
      descendant.class_eval do
        include Adamantium, Concord.new(:node)

        extend DSL
      end
    end

    private

    def children
      node.children
    end
  end # NodeDetails
end # Unparser
