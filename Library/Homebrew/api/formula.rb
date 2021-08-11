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

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "#{formula_api_path}/#{name}.json"
        end
      end
    end
  end
end

require "extend/os/api/formula"
