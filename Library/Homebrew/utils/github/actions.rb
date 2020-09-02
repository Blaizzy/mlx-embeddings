# frozen_string_literal: true

module GitHub
  # Helper functions for interacting with GitHub Actions.
  #
  # @api private
  module Actions
    def self.escape(string)
      string.gsub(/\r/, "%0D")
            .gsub(/\n/, "%0A")
            .gsub(/]/, "%5D")
            .gsub(/;/, "%3B")
    end

    # Helper class for formatting annotations on GitHub Actions.
    class Annotation
      def initialize(type, message, file: nil, line: nil, column: nil)
        raise ArgumentError, "Unsupported type: #{type.inspect}" unless [:warning, :error].include?(type)

        @type = type
        @message = String(message)
        @file = Pathname(file) if file
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
    end
  end
end
