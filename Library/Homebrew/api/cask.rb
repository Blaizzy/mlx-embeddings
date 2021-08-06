# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the cask JSON API.
    #
    # @api private
    module Cask
      extend T::Sig

      module_function

      sig { params(name: String).returns(Hash) }
      def fetch(name)
        Homebrew::API.fetch "cask/#{name}.json", json: true
      end
    end
  end
end
