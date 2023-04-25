# typed: true
# frozen_string_literal: true

require "extend/cachable"

module Homebrew
  module API
    # Helper functions for using the formula JSON API.
    #
    # @api private
    module Formula
      class << self
        include Cachable

        private :cache

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "formula/#{name}.json"
        end

        sig { returns(T::Boolean) }
        def download_and_cache_data!
          json_formulae, updated = Homebrew::API.fetch_json_api_file "formula.jws.json",
                                                                     target: HOMEBREW_CACHE_API/"formula.jws.json"

          cache["aliases"] = {}
          cache["formulae"] = json_formulae.to_h do |json_formula|
            json_formula["aliases"].each do |alias_name|
              cache["aliases"][alias_name] = json_formula["name"]
            end

            [json_formula["name"], json_formula.except("name")]
          end

          updated
        end
        private :download_and_cache_data!

        sig { returns(Hash) }
        def all_formulae
          unless cache.key?("formulae")
            json_updated = download_and_cache_data!
            write_names_and_aliases(regenerate: json_updated)
          end

          cache["formulae"]
        end

        sig { returns(Hash) }
        def all_aliases
          unless cache.key?("aliases")
            json_updated = download_and_cache_data!
            write_names_and_aliases(regenerate: json_updated)
          end

          cache["aliases"]
        end

        sig { params(regenerate: T::Boolean).void }
        def write_names_and_aliases(regenerate: false)
          download_and_cache_data! unless cache.key?("formulae")

          return unless Homebrew::API.write_names_file(all_formulae.keys, "formula", regenerate: regenerate)

          (HOMEBREW_CACHE_API/"formula_aliases.txt").open("w") do |file|
            all_aliases.each do |alias_name, real_name|
              file.puts "#{alias_name}|#{real_name}"
            end
          end
        end
      end
    end
  end
end
