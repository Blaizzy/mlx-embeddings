# frozen_string_literal: true

require "rspec/core"
require "rspec/core/formatters/base_formatter"

# TODO: Replace with `rspec-github` when https://github.com/Drieam/rspec-github/pull/4 is merged.
module RSpec
  module Github
    class Formatter < RSpec::Core::Formatters::BaseFormatter
      RSpec::Core::Formatters.register self, :example_failed, :example_pending

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
        output.puts "\n::error file=#{file},line=#{line}::#{failure.message_lines.join("%0A")}"
      end

      def example_pending(pending)
        file, line = pending.example.location.split(":")
        file = self.class.relative_path(file)
        output.puts "\n::warning file=#{file},line=#{line}::#{pending.example.full_description}"
      end
    end
  end
end
