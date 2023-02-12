# typed: true
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the cask JSON API.
    #
    # @api private
    module Cask
      class << self
        extend T::Sig

        sig { params(token: String).returns(Hash) }
        def fetch(token)
          Homebrew::API.fetch "cask/#{token}.json"
        end

        sig { params(token: String, git_head: T.nilable(String)).returns(String) }
        def fetch_source(token, git_head: nil)
          Homebrew::API.fetch_file_source "Casks/#{token}.rb", repo: "Homebrew/homebrew-cask", git_head: git_head
        end

        sig { returns(Hash) }
        def all_casks
          @all_casks ||= begin
            json_casks = Homebrew::API.fetch_json_api_file "cask.json",
                                                           target: HOMEBREW_CACHE_API/"cask.json"

            json_casks.to_h do |json_cask|
              [json_cask["token"], json_cask.except("token")]
            end
          end
        end
      end
    end
  end
end
