# typed: false
# frozen_string_literal: true

class AbstractDownloadStrategy
  module Compat
    def fetch(timeout: nil)
      super()
    end
  end

  class << self
    def method_added(method)
      if method == :fetch && instance_method(method).arity.zero?
        odeprecated "`def fetch` in a subclass of #{self}",
                    "`def fetch(timeout: nil, **options)` and output a warning " \
                    "when `options` contains new unhandled options"

        class_eval do
          prepend Compat
        end
      end

      super
    end
  end
end
