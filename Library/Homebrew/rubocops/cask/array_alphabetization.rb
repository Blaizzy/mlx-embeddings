# typed: true
# frozen_string_literal: true

module RuboCop
  module Cop
    module Cask
      class ArrayAlphabetization < Base
        extend AutoCorrector

        def on_send(node)
          return if node.method_name != :zap

          node.each_descendant(:pair).each do |pair|
            pair.each_descendant(:array).each do |array|
              if array.children.length == 1
                add_offense(array, message: "Remove the `[]` around a single `zap trash` path") do |corrector|
                  corrector.replace(array.source_range, array.children.first.source)
                end
              end

              array.each_descendant(:str).each_cons(2) do |first, second|
                next if first.source < second.source

                add_offense(second, message: "The `zap trash` paths should be in alphabetical order") do |corrector|
                  corrector.insert_before(first.source_range, second.source)
                  corrector.insert_before(second.source_range, first.source)
                  # Using `corrector.replace` here trips the clobbering detection.
                  corrector.remove(first.source_range)
                  corrector.remove(second.source_range)
                end
              end
            end
          end
        end
      end
    end
  end
end
