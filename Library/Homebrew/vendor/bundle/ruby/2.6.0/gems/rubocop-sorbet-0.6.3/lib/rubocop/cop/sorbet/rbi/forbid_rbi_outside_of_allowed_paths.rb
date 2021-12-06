# frozen_string_literal: true

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
          add_offense(
            nil,
            location: source_range(processed_source.buffer, 1, 0),
            message: message
          ) if allowed_paths.none? { |pattern| File.fnmatch(pattern, processed_source.file_path) }
        end

        private

        def allowed_paths
          cop_config["AllowedPaths"]&.compact || []
        end

        def message
          if allowed_paths.empty?
            "RBI files should be located in an allowed path, but AllowedPaths is empty or nil"
          else
            "RBI file path should match one of: #{allowed_paths.join(", ")}"
          end
        end
      end
    end
  end
end
