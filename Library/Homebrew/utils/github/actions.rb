# frozen_string_literal: true

require "utils/tty"

module GitHub
  # Helper functions for interacting with GitHub Actions.
  #
  # @api private
  module Actions
    def self.escape(string)
      # See https://github.community/t/set-output-truncates-multiline-strings/16852/3.
      string.gsub("%", "%25")
            .gsub("\n", "%0A")
            .gsub("\r", "%0D")
    end

    # Helper class for formatting annotations on GitHub Actions.
    class Annotation
      def self.path_relative_to_workspace(path)
        workspace = Pathname(ENV.fetch("GITHUB_WORKSPACE", Dir.pwd)).realpath
        path = Pathname(path)
        return path unless path.exist?

        path.realpath.relative_path_from(workspace)
      end

      def initialize(type, message, file: nil, line: nil, column: nil)
        raise ArgumentError, "Unsupported type: #{type.inspect}" unless [:warning, :error].include?(type)

        @type = type
        @message = Tty.strip_ansi(message)
        @file = self.class.path_relative_to_workspace(file) if file
        @line = Integer(line) if line
        @column = Integer(column) if column
      end

      def to_s
        file = "file=#{Actions.escape(@file.to_s)}" if @file
        line = "line=#{@line}" if @line
        column = "col=#{@column}" if @column

        metadata = [*file, *line, *column].join(",").presence&.prepend(" ")

        "::#{@type}#{metadata}::#{Actions.escape(@message)}"
      end

      # An annotation is only relevant if the corresponding `file` is relative to
      # the `GITHUB_WORKSPACE` directory or if no `file` is specified.
      def relevant?
        return true if @file.nil?

        @file.descend.next.to_s != ".."
      end
    end
  end
end
