# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the cask JSON API.
    #
    # @api private
    module Cask
      class << self
        extend T::Sig

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "cask/#{name}.json"
        end
      end
    end
  end
end
