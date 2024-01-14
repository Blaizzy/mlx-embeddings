# typed: true
# frozen_string_literal: true

module RuboCop
  module Cop
    module Cask
      class ArrayAlphabetization < Base
        extend AutoCorrector

        def on_send(node)
          return unless [:zap, :uninstall].include?(name = node.method_name)

          node.each_descendant(:pair).each do |pair|
            symbols = pair.children.select(&:sym_type?).map(&:value)
            # For `zap`s, we only care about `trash` arrays.
            next if name == :zap && !symbols.include?(:trash)
            # Don't order `uninstall` arrays that contain commands.
            next if name == :uninstall && (symbols & [:signal, :script, :early_script]).any?

            pair.each_descendant(:array).each do |array|
              if array.children.length == 1
                add_offense(array, message: "Avoid single-element arrays by removing the []") do |corrector|
                  corrector.replace(array.source_range, array.children.first.source)
                end
              end

              next if array.children.length <= 1

              sorted_array = array.children.sort_by { |child| child.source.downcase }
              next if sorted_array.map(&:source) == array.children.map(&:source)

              add_offense(array, message: "The array elements should be ordered alphabetically") do |corrector|
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
