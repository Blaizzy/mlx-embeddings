# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop makes sure that RBI files are always located under sorbet/rbi/.
      #
      # @example
      #   # bad
      #   lib/some_file.rbi
      #   other_file.rbi
      #
      #   # good
      #   sorbet/rbi/some_file.rbi
      #   sorbet/rbi/any/path/for/file.rbi
      class ForbidRBIOutsideOfSorbetDir < RuboCop::Cop::Cop
        include RangeHelp

        PATH_REGEXP = %r{sorbet/rbi}

        def investigate(processed_source)
          add_offense(
            nil,
            location: source_range(processed_source.buffer, 1, 0),
            message: "RBI files are only accepted in the sorbet/rbi/ directory."
          ) unless processed_source.file_path =~ PATH_REGEXP
        end
      end
    end
  end
end
