# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the formula JSON API.
    #
    # @api private
    module Formula
      class << self
        extend T::Sig

        sig { returns(String) }
        def formula_api_path
          "formula"
        end
        alias generic_formula_api_path formula_api_path

        sig { returns(String) }
        def cached_formula_json_file
          HOMEBREW_CACHE_API/"#{formula_api_path}.json"
        end

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "#{formula_api_path}/#{name}.json"
        end

        sig { returns(Array) }
        def all_formulae
          @all_formulae ||= begin
            curl_args = %w[--compressed --silent https://formulae.brew.sh/api/formula.json]
            if cached_formula_json_file.exist?
              last_modified = cached_formula_json_file.mtime.utc
              last_modified = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
              curl_args = ["--time-cond", last_modified, *curl_args] unless cached_formula_json_file.empty?
            end
            curl_download(*curl_args, to: HOMEBREW_CACHE_API/"#{formula_api_path}.json", max_time: 5)

            json_formulae = JSON.parse(cached_formula_json_file.read)

            json_formulae.to_h do |json_formula|
              [json_formula["name"], json_formula.except("name")]
            end
          end
        end
      end
    end
  end
end
