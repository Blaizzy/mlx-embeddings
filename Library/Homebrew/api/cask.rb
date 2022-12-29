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

        MAX_RETRIES = 3

        sig { returns(String) }
        def cached_cask_json_file
          HOMEBREW_CACHE_API/"cask.json"
        end

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "cask/#{name}.json"
        end

        sig { returns(Hash) }
        def all_casks
          @all_casks ||= begin
            retry_count = 0

            url = "https://formulae.brew.sh/api/cask.json"
            json_casks = begin
              curl_args = %W[--compressed --silent #{url}]
              if cached_cask_json_file.exist? && !cached_cask_json_file.empty?
                curl_args.prepend("--time-cond", cached_cask_json_file)
              end
              curl_download(*curl_args, to: cached_cask_json_file, max_time: 5)

              JSON.parse(cached_cask_json_file.read)
            rescue JSON::ParserError
              cached_cask_json_file.unlink
              retry_count += 1
              odie "Cannot download non-corrupt #{url}!" if retry_count > MAX_RETRIES

              retry
            end

            json_casks.to_h do |json_cask|
              [json_cask["token"], json_cask.except("token")]
            end
          end
        end
      end
    end
  end
end
