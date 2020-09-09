# frozen_string_literal: true

require "rspec/core"
require "rspec/core/formatters/base_formatter"

# TODO: Replace with `rspec-github` when https://github.com/Drieam/rspec-github/pull/4 is merged.
module RSpec
  module Github
    class Formatter < RSpec::Core::Formatters::BaseFormatter
      RSpec::Core::Formatters.register self, :example_failed, :example_pending

      def self.escape(string)
        # See https://github.community/t/set-output-truncates-multiline-strings/16852/3.
        string.gsub("%", "%25")
              .gsub("\n", "%0A")
              .gsub("\r", "%0D")
      end

      def self.relative_path(path)
        if (workspace = ENV["GITHUB_WORKSPACE"])
          workspace = "#{File.realpath(workspace)}#{File::SEPARATOR}"
          absolute_path = File.realpath(path)

          return absolute_path.delete_prefix(workspace) if absolute_path.start_with?(workspace)
        end

        path
      end

      def example_failed(failure)
        file, line = failure.example.location.split(":")
        file = self.class.relative_path(file)

        description = failure.example.full_description
        message = failure.message_lines.join("\n")
        annotation = "#{description}\n\n#{message}"

        output.puts "\n::error file=#{file},line=#{line}::#{self.class.escape(annotation)}"
      end

      def example_pending(pending)
        file, line = pending.example.location.split(":")
        file = self.class.relative_path(file)

        description = pending.example.full_description
        message = if pending.example.skip
          "Skipped: #{pending.example.execution_result.pending_message}"
        else
          "Pending: #{pending.example.execution_result.pending_message}"
        end
        annotation = "#{description}\n\n#{message}"

        output.puts "\n::warning file=#{file},line=#{line}::#{self.class.escape(annotation)}"
      end
    end
  end
end
