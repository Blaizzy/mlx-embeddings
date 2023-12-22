# typed: true
# frozen_string_literal: true

module RuboCop
  module Cop
    module Cask
      class ArrayAlphabetization < Base
        extend AutoCorrector

        SINGLE_MSG = "Remove the `[]` around a single `zap trash` path".freeze
        NON_ALPHABETICAL_MSG = "The `zap trash` paths should be in alphabetical order".freeze

        def on_send(node)
          return if node.method_name != :zap

          node.each_descendant(:pair).each do |pair|
            pair.each_descendant(:array).each do |array|
              if array.children.length == 1
                add_offense(array, message: SINGLE_MSG) do |corrector|
                  corrector.replace(array.source_range, array.children.first.source)
                end
              end

              next if array.children.length <= 1

              sorted_array = array.children.sort_by { |child| child.source.downcase }
              next if sorted_array.map(&:source) == array.children.map(&:source)

              add_offense(array, message: NON_ALPHABETICAL_MSG) do |corrector|
                array.children.each_with_index do |child, index|
                  corrector.replace(child.source_range, sorted_array[index].source)
                end
              end
            end
          end
        end
      end
    end
  end
end
