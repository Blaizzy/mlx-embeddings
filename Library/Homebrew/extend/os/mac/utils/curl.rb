# typed: true
# frozen_string_literal: true

module Utils
  module Curl
    undef return_value_for_empty_http_status_code

    def return_value_for_empty_http_status_code(url_type, url)
      # Hack around https://github.com/Homebrew/brew/issues/3199
      return if MacOS.version == :el_capitan

      generic_return_value_for_empty_http_status_code url_type, url
    end
  end
end
