# typed: false
# frozen_string_literal: true

class AbstractDownloadStrategy
  module CompatFetch
    def fetch(timeout: nil)
      super()
    end
  end

  module Compat_Fetch # rubocop:disable Naming/ClassAndModuleCamelCase
    def _fetch(*args, **options)
      options[:timeout] = nil unless options.key?(:timeout)

      begin
        super
      rescue ArgumentError => e
        raise unless e.message.include?("timeout")

        odisabled "`def _fetch` in a subclass of `CurlDownloadStrategy`"
        options.delete(:timeout)
        super(*args, **options)
      end
    end
  end

  class << self
    def method_added(method)
      if method == :fetch && instance_method(method).arity.zero?
        odisabled "`def fetch` in a subclass of `#{self}`",
                  "`def fetch(timeout: nil, **options)` and output a warning " \
                  "when `options` contains new unhandled options"

        class_eval do
          prepend CompatFetch
        end
      elsif method == :_fetch
        class_eval do
          prepend Compat_Fetch
        end
      end

      super
    end
  end
end
