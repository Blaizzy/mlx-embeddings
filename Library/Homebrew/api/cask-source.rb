# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the cask source API.
    #
    # @api private
    module CaskSource
      class << self
        extend T::Sig

        sig { params(token: String, git_head: T.nilable(String)).returns(Hash) }
        def fetch(token, git_head: nil)
          token = token.sub(%r{^homebrew/cask/}, "")
          Homebrew::API.fetch_source token, git_head: git_head
        end

        sig { params(token: String).returns(T::Boolean) }
        def available?(token)
          fetch token
          true
        rescue ArgumentError
          false
        end
      end
    end
  end
end
