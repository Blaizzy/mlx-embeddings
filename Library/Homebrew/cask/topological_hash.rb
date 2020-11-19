# typed: true
# frozen_string_literal: true

require "tsort"

module Cask
  # Topologically sortable hash map.
  class TopologicalHash < Hash
    include TSort

    private

    def tsort_each_node(&block)
      each_key(&block)
    end

    def tsort_each_child(node, &block)
      fetch(node).each(&block)
    end
  end
end
