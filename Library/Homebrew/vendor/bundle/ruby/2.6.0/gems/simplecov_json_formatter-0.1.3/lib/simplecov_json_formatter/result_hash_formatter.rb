# frozen_string_literal: true

require 'simplecov_json_formatter/source_file_formatter'

module SimpleCovJSONFormatter
  class ResultHashFormatter
    def initialize(result)
      @result = result
    end

    def format
      @result.files.each do |source_file|
        formatted_result[:coverage][source_file.filename] =
          format_source_file(source_file)
      end

      formatted_result
    end

    private

    def formatted_result
      @formatted_result ||= {
        meta: {
          simplecov_version: SimpleCov::VERSION
        },
        coverage: {}
      }
    end

    def format_source_file(source_file)
      source_file_formatter = SourceFileFormatter.new(source_file)
      source_file_formatter.format
    end
  end
end
