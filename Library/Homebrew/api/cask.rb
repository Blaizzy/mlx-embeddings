# typed: true
# frozen_string_literal: true

require "extend/cachable"

module Homebrew
  module API
    # Helper functions for using the cask JSON API.
    #
    # @api private
    module Cask
      class << self
        include Cachable

        private :cache

        sig { params(token: String).returns(Hash) }
        def fetch(token)
          Homebrew::API.fetch "cask/#{token}.json"
        end

        sig {
          params(token: String, path: T.any(String, Pathname), git_head: String,
                 sha256: T.nilable(String)).returns(String)
        }
        def fetch_source(token, path:, git_head:, sha256: nil)
          Homebrew::API.fetch_homebrew_cask_source token, path: path, git_head: git_head, sha256: sha256
        end

        sig { returns(T::Boolean) }
        def download_and_cache_data!
          json_casks, updated = Homebrew::API.fetch_json_api_file "cask.jws.json",
                                                                  target: HOMEBREW_CACHE_API/"cask.jws.json"

          cache["casks"] = json_casks.to_h do |json_cask|
            [json_cask["token"], json_cask.except("token")]
          end

          updated
        end
        private :download_and_cache_data!

        sig { returns(Hash) }
        def all_casks
          unless cache.key?("casks")
            json_updated = download_and_cache_data!
            write_names(regenerate: json_updated)
          end

          cache["casks"]
        end

        sig { params(regenerate: T::Boolean).void }
        def write_names(regenerate: false)
          download_and_cache_data! unless cache.key?("casks")

          Homebrew::API.write_names_file(all_casks.keys, "cask", regenerate: regenerate)
        end
      end
    end
  end
end
