# typed: true
# frozen_string_literal: true

require "utils/tty"

module GitHub
  # Helper functions for interacting with GitHub Actions.
  #
  # @api private
  module Actions
    extend T::Sig

    sig { params(string: String).returns(String) }
    def self.escape(string)
      # See https://github.community/t/set-output-truncates-multiline-strings/16852/3.
      string.gsub("%", "%25")
            .gsub("\n", "%0A")
            .gsub("\r", "%0D")
    end

    # Helper class for formatting annotations on GitHub Actions.
    class Annotation
      extend T::Sig

      ANNOTATION_TYPES = [:notice, :warning, :error].freeze

      sig { params(path: T.any(String, Pathname)).returns(T.nilable(Pathname)) }
      def self.path_relative_to_workspace(path)
        workspace = Pathname(ENV.fetch("GITHUB_WORKSPACE", Dir.pwd)).realpath
        path = Pathname(path)
        return path unless path.exist?

        path.realpath.relative_path_from(workspace)
      end

      sig {
        params(
          type:       Symbol,
          message:    String,
          title:      T.nilable(String),
          file:       T.nilable(T.any(String, Pathname)),
          line:       T.nilable(Integer),
          end_line:   T.nilable(Integer),
          column:     T.nilable(Integer),
          end_column: T.nilable(Integer),
        ).void
      }
      def initialize(type, message, title: nil, file: nil, line: nil, end_line: nil, column: nil, end_column: nil)
        raise ArgumentError, "Unsupported type: #{type.inspect}" if ANNOTATION_TYPES.exclude?(type)

        @type = type
        @message = Tty.strip_ansi(message)
        @title = Tty.strip_ansi(title) if title
        @file = self.class.path_relative_to_workspace(file) if file
        @line = Integer(line) if line
        @end_line = Integer(end_line) if end_line
        @column = Integer(column) if column
        @end_column = Integer(end_column) if end_column
      end

      sig { returns(String) }
      def to_s
        metadata = @type.to_s

        if @file
          metadata << " file=#{Actions.escape(@file.to_s)}"

          if @line
            metadata << ",line=#{@line}"
            metadata << ",endLine=#{@end_line}" if @end_line

            if @column
              metadata << ",col=#{@column}"
              metadata << ",endColumn=#{@end_column}" if @end_column
            end
          end
        end

        if @title
          metadata << if @file
            ","
          else
            " "
          end
          metadata << "title=#{Actions.escape(@title)}"
        end

        "::#{metadata}::#{Actions.escape(@message)}"
      end

      # An annotation is only relevant if the corresponding `file` is relative to
      # the `GITHUB_WORKSPACE` directory or if no `file` is specified.
      sig { returns(T::Boolean) }
      def relevant?
        return true if @file.nil?

        @file.descend.next.to_s != ".."
      end
    end
  end
end
