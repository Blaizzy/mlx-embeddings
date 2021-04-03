# typed: false
# frozen_string_literal: true

class CurlDownloadStrategy
  module Compat
    def _fetch(*args, **options)
      unless options.key?(:timeout)
        odeprecated "#{self.class}#_fetch"
        options[:timeout] = nil
      end
      super(*args, **options)
    end
  end

  prepend Compat
end

class CurlPostDownloadStrategy
  prepend Compat
end
