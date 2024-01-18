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
            next if name == :uninstall && (symbols & [:signal, :script, :early_script, :args, :input]).any?

            pair.each_descendant(:array).each do |array|
              if array.children.length == 1
                add_offense(array, message: "Avoid single-element arrays by removing the []") do |corrector|
                  corrector.replace(array.source_range, array.children.first.source)
                end
              end

              next if array.children.length <= 1

              comments = find_inline_comments(array.source)
              array_with_comments = array.children.dup
              array.children.map(&:source).each_with_index do |child, index|
                comment = comments.find { |c| c.include?(child) }
                next unless comment

                p comment.strip
                # Add the comment to the main array.
                array_with_comments[index] = comment.strip
              end

              sorted_array = array_with_comments.sort_by { |child| child.to_s.downcase }
              next if sorted_array == array_with_comments

              add_offense(array, message: "The array elements should be ordered alphabetically") do |corrector|
                array.children.each_with_index do |child, index|
                  p sorted_array[index]
                  corrector.replace(child.source_range, sorted_array[index])
                end
              end
            end
          end
        end

        def find_inline_comments(source)
          comments = []
          source.each_line do |line|
            # Comments are naively detected by looking for lines that include a `#` surrounded by spaces.
            comments << line if line.include?(" # ")
          end

          # Remove lines that are only comments, we don't want to move those.
          comments.reject { |comment| comment.strip.start_with?("# ") }
        end
      end
    end
  end
end
