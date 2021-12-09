# frozen_string_literal: true

require "pathname"

module RuboCop
  module Cop
    module Sorbet
      # This cop makes sure that RBI files are always located under the defined allowed paths.
      #
      # Options:
      #
      # * `AllowedPaths`: A list of the paths where RBI files are allowed (default: ["sorbet/rbi/**"])
      #
      # @example
      #   # bad
      #   # lib/some_file.rbi
      #   # other_file.rbi
      #
      #   # good
      #   # sorbet/rbi/some_file.rbi
      #   # sorbet/rbi/any/path/for/file.rbi
      class ForbidRBIOutsideOfAllowedPaths < RuboCop::Cop::Cop
        include RangeHelp

        def investigate(processed_source)
          paths = allowed_paths

          if paths.nil?
            add_offense(
              nil,
              location: source_range(processed_source.buffer, 1, 0),
              message: "AllowedPaths expects an array"
            )
            return
          elsif paths.empty?
            add_offense(
              nil,
              location: source_range(processed_source.buffer, 1, 0),
              message: "AllowedPaths cannot be empty"
            )
            return
          end

          # When executed the path to the source file is absolute.
          # We need to remove the exec path directory prefix before matching with the filename regular expressions.
          rel_path = processed_source.file_path.sub("#{Dir.pwd}/", "")

          add_offense(
            nil,
            location: source_range(processed_source.buffer, 1, 0),
            message: "RBI file path should match one of: #{paths.join(", ")}"
          ) if paths.none? { |pattern| File.fnmatch(pattern, rel_path) }
        end

        private

        def allowed_paths
          paths = cop_config["AllowedPaths"]
          return nil unless paths.is_a?(Array)
          paths.compact
        end
      end
    end
  end
end
